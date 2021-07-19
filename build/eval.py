import contextlib
import copy
import os
import resource
from pprint import PrettyPrinter

import numpy as np
from build.coco_eval import *
from build.coco_eval import evaluate as pycoco_evaluate
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
from utils import *

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

# Good formatting when printing the APs for each class and mAP
pp = PrettyPrinter()


def create_common_coco_eval(coco_eval, img_ids, eval_imgs):
    img_ids = list(img_ids)
    eval_imgs = list(eval_imgs.flatten())

    coco_eval.evalImgs = eval_imgs
    coco_eval.params.imgIds = img_ids
    coco_eval._paramsEval = copy.deepcopy(coco_eval.params)


def evaluate(test_loader, base_ds, model, criterion, args=None):
    """
    Evaluate.
    :param args:
    :param criterion:
    :param base_ds:
    :param test_loader: DataLoader for test data
    :param model: model
    """
    iou_types = ['bbox']
    self_img_ids = []
    self_coco_eval = {}
    self_eval_imgs = {k: [] for k in iou_types}
    top_k = args.top_k or 200

    # Make sure it's in eval mode
    model.eval()
    criterion.eval()

    # Lists to store detected and true boxes, labels, scores
    cus_det_boxes = list()
    cus_det_labels = list()
    cus_det_scores = list()
    cus_true_boxes = list()
    cus_true_labels = list()
    cus_true_difficulties = list()

    with torch.no_grad():
        # Batches
        for i, (images, boxes, labels, ids) in enumerate(tqdm(test_loader, desc='Evaluating')):
            if args and i == args.num_eval_images:
                break
            batch_size = images.size(0)
            predictions = {}

            images = images.to(device)
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]

            # Forward prop.
            predicted_locs, predicted_scores = model(images)
            loss, loss_dict = criterion(predicted_locs, predicted_scores, boxes, labels, args)

            # Detect objects in SSD output
            det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(predicted_locs,
                                                                                       predicted_scores,
                                                                                       min_score=0.01,
                                                                                       max_overlap=0.9900000095367432,
                                                                                       top_k=top_k)
            # Store this batch's results for mAP calculation

            batch_difficulties = []
            for i in range(len(labels)):
                difficulties = torch.zeros_like(labels[i]).to(device)
                batch_difficulties.append(difficulties)

            cus_det_boxes.extend(det_boxes_batch)
            cus_det_labels.extend(det_labels_batch)
            cus_det_scores.extend(det_scores_batch)
            cus_true_boxes.extend(boxes)
            cus_true_labels.extend(labels)
            cus_true_difficulties.extend(batch_difficulties)

            det_boxes_batch = [k * torch.tensor([args.img_width, args.img_height, args.img_width, args.img_height],
                                                dtype=torch.float32).to(device) for k in det_boxes_batch]

            boxes_batch = [l * torch.tensor([args.img_width, args.img_height, args.img_width, args.img_height],
                                            dtype=torch.float32).to(device) for l in boxes]

            for iou_type in iou_types:
                self_coco_eval[iou_type] = COCOeval(base_ds, iouType=iou_type)

            for j in range(batch_size):
                original_id = ids[j].numpy()[0]
                predictions[original_id] = {}
                predictions[original_id]["det_boxes"] = det_boxes_batch[j]
                predictions[original_id]["det_scores"] = det_scores_batch[j]
                predictions[original_id]["det_labels"] = det_labels_batch[j]

            img_ids = list(np.unique(list(predictions.keys())))
            self_img_ids.extend(img_ids)

            for iou_type in iou_types:
                coco_results = []
                for original_id, prediction in predictions.items():
                    if len(prediction) == 0:
                        continue

                    det_boxes = predictions[original_id]["det_boxes"].tolist()
                    det_scores = predictions[original_id]["det_scores"].tolist()
                    det_labels = predictions[original_id]["det_labels"].tolist()

                    coco_results.extend(
                        [
                            {
                                "image_id": original_id,
                                "category_id": det_labels[k],
                                "bbox": box,
                                "score": det_scores[k],
                            }
                            for k, box in enumerate(det_boxes)
                        ]
                    )

                # suppress pycocotools prints
                with open(os.devnull, 'w') as devnull:
                    with contextlib.redirect_stdout(devnull):
                        coco_dt = COCO.loadRes(base_ds, coco_results) if coco_results else COCO()

                coco_eval = self_coco_eval[iou_type]

                coco_eval.cocoDt = coco_dt
                coco_eval.params.imgIds = list(img_ids)

                img_ids, eval_imgs = pycoco_evaluate(coco_eval)
                self_eval_imgs[iou_type].append(eval_imgs)

        for iou_type in iou_types:
            self_eval_imgs[iou_type] = np.concatenate(self_eval_imgs[iou_type], 2)
            create_common_coco_eval(self_coco_eval[iou_type], self_img_ids, self_eval_imgs[iou_type])

        for coco_eval in self_coco_eval.values():
            coco_eval.accumulate()

        for iou_type, coco_eval in self_coco_eval.items():
            print("IoU metric: {}".format(iou_type))
            coco_eval.summarize()

        coco_results = coco_eval.stats
        args.writer.add_scalar("CocoEvaluation_Precision/mAP", coco_results[0], args.iter_count)
        args.writer.add_scalar("CocoEvaluation_Precision/mAP@.50IOU", coco_results[1], args.iter_count)
        args.writer.add_scalar("CocoEvaluation_Precision/mAP@.75IOU", coco_results[2], args.iter_count)
        args.writer.add_scalar("CocoEvaluation_Precision/mAP (small)", coco_results[3], args.iter_count)
        args.writer.add_scalar("CocoEvaluation_Precision/mAP (medium)", coco_results[4], args.iter_count)
        args.writer.add_scalar("CocoEvaluation_Precision/mAP (large)", coco_results[5], args.iter_count)

        args.writer.add_scalar("CocoEvaluation_Recall/AR@1", coco_results[6], args.iter_count)
        args.writer.add_scalar("CocoEvaluation_Recall/AR@10", coco_results[7], args.iter_count)
        args.writer.add_scalar("CocoEvaluation_Recall/AR@100", coco_results[8], args.iter_count)
        args.writer.add_scalar("CocoEvaluation_Recall/AR@100 (small)", coco_results[9], args.iter_count)
        args.writer.add_scalar("CocoEvaluation_Recall/AR@100 (medium)", coco_results[10], args.iter_count)
        args.writer.add_scalar("CocoEvaluation_Recall/AR@100 (large)", coco_results[11], args.iter_count)

        args.writer.add_scalar("Training_Loss/eval_cl_loss", loss_dict["cl_loss"], args.iter_count)
        args.writer.add_scalar("Training_Loss/eval_loc_loss", loss_dict["loc_loss"], args.iter_count)

        args.writer.flush()

        # Calculate mAP
        APs, mAP = calculate_mAP(det_boxes=cus_det_boxes,
                                 det_labels=cus_det_labels,
                                 det_scores=cus_det_scores,
                                 true_boxes=cus_true_boxes,
                                 true_labels=cus_true_labels,
                                 true_difficulties=cus_true_difficulties)

    # Print AP for each class
    pp.pprint(APs)

    print('\nMean Average Precision (mAP): %.3f' % mAP)

    return APs, mAP, coco_eval
