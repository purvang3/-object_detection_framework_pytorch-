import argparse
import os
import random
import time
from pathlib import Path

import misc
import numpy as np
import torch
import torchvision
import torch.backends.cudnn as cudnn
from build.dataset import build_dataset
from build.eval import evaluate
from build.losses import MultiBoxLoss
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from utils import *


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        if isinstance(dataset, torchvision.datasets.CocoDetection):
            break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco


def collate_fn(batch):
    """
    Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).
    This describes how to combine these tensors of different sizes. We use lists.
    Note: this need not be defined in this Class, can be standalone.
    :param batch: an iterable of N sets from __getitem__()
    :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
    """

    images = list()
    boxes = list()
    labels = list()
    ids = list()

    for b in batch:
        images.append(b[0])
        boxes.append(b[1]['boxes'])
        labels.append(b[1]['labels'])
        ids.append(b[1]['image_id'])

    images = torch.stack(images, dim=0)

    return images, boxes, labels, ids


def main(args):
    misc.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    train_dataset = build_dataset(image_set='train', args=args)
    eval_dataset = build_dataset(image_set='eval', args=args)

    if args.distributed:
        sampler_train = DistributedSampler(train_dataset, shuffle=True)
        sampler_eval = DistributedSampler(eval_dataset, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(train_dataset)
        sampler_eval = torch.utils.data.SequentialSampler(eval_dataset)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    batch_sampler_eval = torch.utils.data.BatchSampler(
        sampler_eval, 1, drop_last=False)

    train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler_train,
                              collate_fn=collate_fn, num_workers=args.num_workers)

    eval_loader = DataLoader(eval_dataset, batch_sampler=batch_sampler_eval,
                             collate_fn=collate_fn, num_workers=args.num_workers)

    base_ds = get_coco_api_from_dataset(eval_dataset)

    if args.backbone == "vgg300":
        from build.models.ssd import SSD300
        model = SSD300(args=args)
    elif args.backbone == "inception_v3":
        from build.models.inception import Inception_V3
        model = Inception_V3(args=args, aux_logits=False, init_weights=True)
    elif args.backbone == "resnet50":
        from build.models.resnet import ResNetModel
        model = ResNetModel(args=args, aux_logits=False, init_weights=False)
    elif args.backbone == "mlp_mixure":
        from build.models.mlp_mixure import Model
        model = Model(args=args)
    else:
        raise ValueError("provide architecture")

    model.to(device)
    criterion = MultiBoxLoss(args=args, priors_cxcy=model.priors_cxcy)
    criterion.to(device)

    if not args.lr_drop:
        args.lr_drop = 40000

    param_dicts = [
        {'params': model.parameters(), "lr": args.lr},
    ]

    if args.optimizer == "adam":
        optimizer = torch.optim.AdamW(param_dicts,
                                      lr=args.lr,
                                      weight_decay=args.weight_decay)

    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params=param_dicts,
                                    lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    else:
        raise ValueError("provide optimizer")

    if args.lr_scheduler == "exponetial_lr":
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer,
                                                              gamma=args.decay_lr_to)
    elif args.lr_scheduler == "cosine_annealing_lr":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                                  T_max=args.iterations,
                                                                  eta_min=0)

    elif args.lr_scheduler == "reduce_on_plateau_lr":
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                                  mode='min',
                                                                  factor=0.008,
                                                                  patience=4,
                                                                  threshold=1e-4,
                                                                  threshold_mode='rel',
                                                                  cooldown=3,
                                                                  min_lr=0,
                                                                  eps=1e-8)

    elif args.lr_scheduler == "cosine_annealing_warm_start_lr":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer,
                                                                            T_0=5)

    # TODO support for cyclicLR

    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                       step_size=args.lr_drop,
                                                       gamma=args.decay_lr_to)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    output_dir = Path(args.output_dir)
    if args.resume:
        print("*** resuming from checkpoint")
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')

        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            import copy
            p_groups = copy.deepcopy(optimizer.param_groups)
            optimizer.load_state_dict(checkpoint['optimizer'])
            for pg, pg_old in zip(optimizer.param_groups, p_groups):
                pg['lr'] = pg_old['lr']
                pg['initial_lr'] = pg_old['initial_lr']
            print(optimizer.param_groups)
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            # todo: this is a hack for doing experiment that resume from checkpoint and also modify lr scheduler (e.g., decrease lr in advance).
            args.override_resumed_lr_drop = True
            if args.override_resumed_lr_drop:
                print(
                    'Warning: (hack) args.override_resumed_lr_drop is set to True, so args.lr_drop would override lr_drop in resumed lr_scheduler.')
                lr_scheduler.step_size = args.lr_drop
                lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
            lr_scheduler.step(lr_scheduler.last_epoch)
            args.start_epoch = checkpoint['epoch'] + 1
            args.iter_count = checkpoint['iter_count'] + 1

    # check the resumed model
    if args.eval:
        APs, mAP, coco_evaluator = evaluate(test_loader=eval_loader,
                                            base_ds=base_ds,
                                            model=model_without_ddp,
                                            criterion=criterion,
                                            args=args)
        if args.output_dir:
            torch.save(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        return

    # Epochs
    for epoch in range(args.start_epoch, args.total_epochs):

        # One epoch's training
        model_without_ddp.train()  # training mode enables dropout
        criterion.train()

        batch_time = AverageMeter()  # forward prop. + back prop. time
        data_time = AverageMeter()  # data loading time
        losses = AverageMeter()  # loss
        mAP_meter = AverageMeter()

        start = time.time()

        # Batches
        for i, (images, boxes, labels, ids) in enumerate(train_loader):
            data_time.update(time.time() - start)

            args.iter_count += 1
            l2_regularization_loss = 0.0
            gradient_norm = 0.0
            loss = 0.0

            # Move to default device
            images = images.to(device)  # (batch_size (N), 3, 300, 300)
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]

            # Forward prop.
            predicted_locs, predicted_scores = model_without_ddp(images)  # (N, 8732, 4), (N, 8732, n_classes)

            # Loss
            loss, loss_dict = criterion(predicted_locs, predicted_scores, boxes, labels, args)  # scalar

            # Backward prop.
            optimizer.zero_grad()
            loss.backward()

            if args.l1_l2_reg:
                l1_regularization_loss = 0
                l1_reg_obj = torch.nn.L1Loss(size_average=False)

            for name, param in model.named_parameters():
                try:
                    gradient_norm += torch.linalg.norm(param.grad)
                    l2_regularization_loss += torch.linalg.norm(param)
                    if args.l1_l2_reg:
                        l1_regularization_loss += l1_reg_obj(param)
                except Exception as e:
                    print(f"*** {name} {param.grad}")
                    continue

            l2_regularization_loss = args.weight_decay * l2_regularization_loss

            if args.l1_l2_reg:
                l1_regularization_loss = args.weight_decay * l1_regularization_loss
                args.writer.add_scalar("Training_Loss/L1_regularization_loss", l1_regularization_loss, args.iter_count)
                args.writer.flush()
                loss += l1_regularization_loss

            # Clip gradients, if necessary
            if args.grad_norm_clip > 0.0:
                # clip_gradient(optimizer, args.grad_norm_clip)  # for param value clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm_clip,
                                               norm_type=2.0)  # for grad norm clipping

            # Update model
            optimizer.step()

            lr_scheduler.step()

            losses.update(loss.item(), images.size(0))

            batch_time.update(time.time() - start)

            args.writer.add_scalar("training_stats/avg_data_time", data_time.avg, args.iter_count)
            args.writer.add_scalar("training_stats/avg_batch_time", batch_time.avg, args.iter_count)
            args.writer.add_scalar("lr", lr_scheduler.get_last_lr()[0], args.iter_count)
            args.writer.add_scalar("Training_Loss/L2_regularization_loss", l2_regularization_loss, args.iter_count)
            args.writer.add_scalar("Training_Loss/gradient_norm", gradient_norm, args.iter_count)
            args.writer.add_scalar("Training_Loss/total_avg_loss", losses.avg, args.iter_count)

            if True:
                args.writer.add_scalar("training_stats/batch_time", batch_time.val, args.iter_count)
                args.writer.add_scalar("training_stats/data_time", data_time.val, args.iter_count)
                args.writer.add_scalar("Training_Loss/total_loss", loss, args.iter_count)

            args.writer.flush()
            # Print status
            if args.iter_count % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i + 1, len(train_loader),
                                                                      batch_time=batch_time,
                                                                      data_time=data_time, loss=losses))

            if args.iter_count % args.eval_freq == 0 and args.iter_count != 0:
                APs, mAP, coco_evaluator = evaluate(test_loader=eval_loader,
                                                    base_ds=base_ds,
                                                    model=model_without_ddp,
                                                    criterion=criterion,
                                                    args=args)

                mAP_meter.update(mAP, args.batch_size)
                # args.writer.add_scalar("eval_mAP/val", mAP_meter.val, args.iter_count)
                # args.writer.add_scalar("eval_mAP/avg", mAP_meter.avg, args.iter_count)

                # for key, val in APs.items():
                #     args.writer.add_scalar("AP/" + key, val, args.iter_count)

                for name, weight in model_without_ddp.named_parameters():
                    args.writer.add_histogram("WEIGHT/" + name, weight, args.iter_count)
                    args.writer.add_histogram("GRAD/" + f'{name}.grad', weight.grad, args.iter_count)

            args.writer.flush()

            start = time.time()

            del predicted_locs, predicted_scores, images, boxes, labels  # free some memory

            # Save checkpoint
            if args.iter_count % args.eval_freq == 0 and args.iter_count != 0:
                save_checkpoint(model=model_without_ddp,
                                optimizer=optimizer,
                                lr_scheduler=lr_scheduler,
                                epoch=epoch,
                                iter_count=args.iter_count,
                                filename=os.path.join(args.output_dir, "checkpoint.pth.tar"))

    args.writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('training and evaluation script')
    args = parser.parse_args()

    cudnn.benchmark = True
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.dataset = "coco"
    args.n_classes = 92  # number of different types of objects

    args.batch_size = 32
    args.num_workers = 4
    args.print_freq = 200  # print training status every __ batches (iterations)
    args.eval_freq = 4000  # print training status every __ batches (iterations)
    args.num_eval_images = 120  # evaluate __ b batches(iterations)
    args.eval = False

    args.backbone = "inception_v3"
    args.freeze_backbone = False

    args.optimizer = "adam"
    args.decay_lr_to = 0.5  # decay learning rate to this fraction of the existing learning rate
    args.momentum = 0.99  # momentum
    args.weight_decay = 3.9999998989515007e-05  # weight decay
    args.grad_norm_clip = 0.0  # clip if gradients are exploding, which may happen at larger batch sizes

    args.lr = 0.001  # learning rate
    args.lr_drop = None  # decay learning rate after _ iterations
    args.lr_scheduler = "step_lr"
    args.activation = "relu6"

    args.seed = 0
    args.iter_count = 0
    args.start_epoch = 0
    args.total_epochs = 50
    args.l1_l2_reg = False

    args.model_name_tag = "coco_test"

    args.coco_path = os.path.expanduser("~/coco")
    args.output_dir = os.path.expanduser(f"~/models/inception_v3/runs/{args.model_name_tag}")
    args.resume = os.path.expanduser(f"~/models/inception_v3/runs/{args.model_name_tag}/checkpoint.pth.tar")
    if not os.path.exists(args.resume):
        args.resume = None

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    args.img_height = 300
    args.img_width = 300
    args.img_size = 256
    args.top_k = 100

    args.cl_weight = 1.0
    args.loc_weight = 1.0

    args.writer = SummaryWriter(log_dir=args.output_dir)
    args.checkpoint_name = args.output_dir
    main(args)
