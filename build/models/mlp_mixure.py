from math import sqrt

import torch
from build.utils import *
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn, einsum

DETECTION = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
activation = nn.GELU


class FeedForward(nn.Module):
    def __init__(self, dim, mlp_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Mixure_Layer(nn.Module):
    def __init__(self, dim, mlp_dim, num_patches, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

        self.channel_mix = FeedForward(num_patches, mlp_dim, dropout)
        self.token_mix = FeedForward(dim, mlp_dim, dropout)

    def forward(self, x):
        skip = x
        x = self.norm(x)
        x = rearrange(x, 'b s c -> b c s')
        x = self.channel_mix(x)
        x = rearrange(x, 'b c s -> b s c')
        x += skip
        x = self.norm(x)
        x = self.token_mix(x)
        return x


class MLP_MIX(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, mlp_dim, channels=3,
                 dropout=0.):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = int((image_size * image_size * channels) / (patch_size * patch_size * channels))
        patch_dim = channels * patch_size ** 2

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.pool = 'mean'
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(Mixure_Layer(dim, mlp_dim, num_patches, dropout))

        if not DETECTION:
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, num_classes)
            )

    def forward(self, x):
        x = self.to_patch_embedding(x)
        for mixure in self.layers:
            x = mixure(x)
        if not DETECTION:
            x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
            x = self.mlp_head(x)
        return x


n_heads = 8


class Predictor(nn.Module):
    def __init__(self, args):
        super(Predictor, self).__init__()

        self.n_classes = args.n_classes
        self.n_heads = n_heads
        self.input_channels = 1

        n_boxes = {'1': 4,
                   '2': 6,
                   '3': 6,
                   '4': 6,
                   '5': 4,
                   '6': 4}

        # Localization prediction convolutions (predict offsets w.r.t prior-boxes)
        self.pre_loc1 = nn.Conv2d(n_heads, 16, kernel_size=3, padding=1, stride=2)
        self.loc1 = nn.Conv2d(16, n_boxes['1'] * 4, kernel_size=1, padding=0)

        self.pre_loc2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2)
        self.loc2 = nn.Conv2d(32, n_boxes['2'] * 4, kernel_size=1, padding=0)

        self.pre_loc3 = nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=2)
        self.loc3 = nn.Conv2d(32, n_boxes['3'] * 4, kernel_size=1, padding=0)

        self.pre_loc4 = nn.Conv2d(32, 16, kernel_size=3, padding=1, stride=2)
        self.loc4 = nn.Conv2d(16, n_boxes['4'] * 4, kernel_size=1, padding=0)

        self.pre_loc5 = nn.Conv2d(16, 8, kernel_size=3, padding=1, stride=2)
        self.loc5 = nn.Conv2d(8, n_boxes['5'] * 4, kernel_size=1, padding=0)

        self.pre_loc6 = nn.Conv2d(8, 4, kernel_size=3, padding=1, stride=2)
        self.loc6 = nn.Conv2d(4, n_boxes['6'] * 4, kernel_size=1, padding=0)

        # Class prediction convolutions (predict classes in localization boxes)
        self.pre_cl1 = nn.Conv2d(n_heads, 16, kernel_size=3, padding=1, stride=2)
        self.cl1 = nn.Conv2d(16, n_boxes['1'] * self.n_classes, kernel_size=1, padding=0)

        self.pre_cl2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2)
        self.cl2 = nn.Conv2d(32, n_boxes['2'] * self.n_classes, kernel_size=1, padding=0)

        self.pre_cl3 = nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=2)
        self.cl3 = nn.Conv2d(32, n_boxes['3'] * self.n_classes, kernel_size=1, padding=0)

        self.pre_cl4 = nn.Conv2d(32, 16, kernel_size=3, padding=1, stride=2)
        self.cl4 = nn.Conv2d(16, n_boxes['4'] * self.n_classes, kernel_size=1, padding=0)

        self.pre_cl5 = nn.Conv2d(16, 8, kernel_size=3, padding=1, stride=2)
        self.cl5 = nn.Conv2d(8, n_boxes['5'] * self.n_classes, kernel_size=1, padding=0)

        self.pre_cl6 = nn.Conv2d(8, 4, kernel_size=3, padding=1, stride=2)
        self.cl6 = nn.Conv2d(4, n_boxes['6'] * self.n_classes, kernel_size=1, padding=0)

    def forward(self, x):
        b = x.size(0)   # 32
        d = x.size(-1)  # 64
        n = x.size(-2)  # 512
        h = int(x.size(-1) / x.size(-2))
        assert (d % (n * h) == 0), "dimention should be multiple of head and head dim"
        assert (h == n_heads)

        x = x.reshape([b, h, n, n])
        # # TODO: apply attention
        attn = (einsum('b h i d, b h j d -> b h i j', x, x) * (d ** -0.5)).softmax(dim=-1)
        x = einsum('b h i j, b h j d -> b h i d', attn, x)

        pre_loc1 = activation()(self.pre_loc1(x))
        pre_loc2 = activation()(self.pre_loc2(pre_loc1))
        pre_loc3 = activation()(self.pre_loc3(pre_loc2))
        pre_loc4 = activation()(self.pre_loc4(pre_loc3))
        pre_loc5 = activation()(self.pre_loc5(pre_loc4))
        pre_loc6 = activation()(self.pre_loc6(pre_loc5))

        ######
        loc1 = self.arrange_and_predict(b, self.loc1, pre_loc1, 4)
        loc2 = self.arrange_and_predict(b, self.loc2, pre_loc2, 4)
        loc3 = self.arrange_and_predict(b, self.loc3, pre_loc3, 4)
        loc4 = self.arrange_and_predict(b, self.loc4, pre_loc4, 4)
        loc5 = self.arrange_and_predict(b, self.loc5, pre_loc5, 4)
        loc6 = self.arrange_and_predict(b, self.loc6, pre_loc6, 4)

        #######

        pre_cl1 = activation()(self.pre_cl1(x))
        pre_cl2 = activation()(self.pre_cl2(pre_cl1))
        pre_cl3 = activation()(self.pre_cl3(pre_cl2))
        pre_cl4 = activation()(self.pre_cl4(pre_cl3))
        pre_cl5 = activation()(self.pre_cl5(pre_cl4))
        pre_cl6 = activation()(self.pre_cl6(pre_cl5))

        ######
        cl1 = self.arrange_and_predict(b, self.cl1, pre_cl1, self.n_classes)
        cl2 = self.arrange_and_predict(b, self.cl2, pre_cl2, self.n_classes)
        cl3 = self.arrange_and_predict(b, self.cl3, pre_cl3, self.n_classes)
        cl4 = self.arrange_and_predict(b, self.cl4, pre_cl4, self.n_classes)
        cl5 = self.arrange_and_predict(b, self.cl5, pre_cl5, self.n_classes)
        cl6 = self.arrange_and_predict(b, self.cl6, pre_cl6, self.n_classes)

        locs = torch.cat([loc1, loc2, loc3, loc4, loc5, loc6], dim=1)
        classes_scores = torch.cat([cl1, cl2, cl3, cl4, cl5, cl6], dim=1)

        return locs, classes_scores

    def arrange_and_predict(self, batch_size, block, head_pred, output_size):
        loc1 = block(head_pred)
        loc1 = loc1.permute(0, 2, 3, 1).contiguous()
        loc1 = loc1.view(batch_size, -1, output_size)
        return loc1


def init_mlp_mix(c):
    if isinstance(c, nn.Conv2d):
        nn.init.xavier_uniform_(c.weight)
        nn.init.constant_(c.bias, 0.)
    if isinstance(c, nn.Linear):
        nn.init.xavier_uniform_(c.weight)
        nn.init.constant_(c.bias, 0.)


class Model(nn.Module):
    def __init__(self, args):
        self.n_classes = args.n_classes
        self.args = args
        super(Model, self).__init__()
        assert (args.img_size % 32 == 0), "image size should be muliple of 64"
        from timm.models.mlp_mixer import MlpMixer, _init_weights
        from functools import partial
        self.backbone = MlpMixer(num_classes=args.n_classes,
                                 img_size=args.img_size,
                                 patch_size=32,
                                 num_blocks=8,
                                 hidden_dim=512,
                                 tokens_dim=256,
                                 channels_dim=2048,
                                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                 act_layer=nn.GELU,
                                 drop=0.,
                                 drop_path=0.,
                                 nlhb=False,
                                 )

        self.head = Predictor(args=args)

        for n, m in self.head.named_modules():
            _init_weights(m, n, head_bias=0.)

        self.priors_cxcy = self.create_prior_boxes(args)

    def forward(self, x):
        x = self.backbone(x)
        loc, cl = self.head(x)
        return loc, cl

    def create_prior_boxes(self, args=None):
        """
        Create the prior (default) boxes.
        :return: prior boxes in center-size coordinates, a tensor of dimensions (8732, 4)
        """
        # TODO: need to automate for any image shape
        fmap_dims = {'1': int(args.img_size / 8),
                     '2': int(args.img_size / 16),
                     '3': int(args.img_size / 32),
                     '4': int(args.img_size / 64),
                     '5': int(args.img_size / 128),
                     '6': int(args.img_size / 256)}

        obj_scales = {'1': 0.1,
                      '2': 0.2,
                      '3': 0.375,
                      '4': 0.55,
                      '5': 0.70,
                      '6': 0.95}

        aspect_ratios = {'1': [1., 2., 0.5],
                         '2': [1., 2., 3., 0.5, .333],
                         '3': [1., 2., 3., 0.5, .333],
                         '4': [1., 2., 3., 0.5, .333],
                         '5': [1., 2., 0.5],
                         '6': [1., 2., 0.5]}

        fmaps = list(fmap_dims.keys())

        prior_boxes = []

        for k, fmap in enumerate(fmaps):
            for i in range(fmap_dims[fmap]):
                for j in range(fmap_dims[fmap]):
                    cx = (j + 0.5) / fmap_dims[fmap]
                    cy = (i + 0.5) / fmap_dims[fmap]

                    for ratio in aspect_ratios[fmap]:
                        prior_boxes.append([cx, cy, obj_scales[fmap] * sqrt(ratio), obj_scales[fmap] / sqrt(ratio)])

                        # For an aspect ratio of 1, use an additional prior whose scale is the geometric mean of the
                        # scale of the current feature map and the scale of the next feature map
                        if ratio == 1.:
                            try:
                                additional_scale = sqrt(obj_scales[fmap] * obj_scales[fmaps[k + 1]])
                            # For the last feature map, there is no "next" feature map
                            except IndexError:
                                additional_scale = 1.
                            prior_boxes.append([cx, cy, additional_scale, additional_scale])

        prior_boxes = torch.FloatTensor(prior_boxes).to(device)  # (8732, 4)
        prior_boxes.clamp_(0, 1)

        return prior_boxes

    def detect_objects(self, predicted_locs, predicted_scores, min_score, max_overlap, top_k):
        """
        Decipher the 8732 locations and class scores (output of ths SSD300) to detect objects.
        For each class, perform Non-Maximum Suppression (NMS) on boxes that are above a minimum threshold.
        :param predicted_locs: predicted locations/boxes w.r.t the 8732 prior boxes, a tensor of dimensions (N, 8732, 4)
        :param predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 8732, n_classes)
        :param min_score: minimum threshold for a box to be considered a match for a certain class
        :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via NMS
        :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
        :return: detections (boxes, labels, and scores), lists of length batch_size
        """
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        import torch.nn.functional as F
        predicted_scores = F.softmax(predicted_scores, dim=2)  # (N, 8732, n_classes)

        # Lists to store final predicted boxes, labels, and scores for all images
        all_images_boxes = list()
        all_images_labels = list()
        all_images_scores = list()

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        for i in range(batch_size):
            # Decode object coordinates from the form we regressed predicted boxes to
            if self.args.dataset == "coco":
                # decoded_locs = gcxgcy_to_cxcy(predicted_locs[i], self.priors_cxcy)
                decoded_locs = cxcy_to_xy(
                    gcxgcy_to_cxcy(predicted_locs[i], self.priors_cxcy))
            else:
                decoded_locs = cxcy_to_xy(
                    gcxgcy_to_cxcy(predicted_locs[i], self.priors_cxcy))

            decoded_locs = torch.clip(decoded_locs, min=0, max=1)

            # Lists to store boxes and scores for this image
            image_boxes = list()
            image_labels = list()
            image_scores = list()

            max_scores, best_label = predicted_scores[i].max(dim=1)

            # Check for each class
            for c in range(1, self.n_classes):
                # Keep only predicted boxes and scores where scores for this class are above the minimum score
                class_scores = predicted_scores[i][:, c]
                score_above_min_score = class_scores > min_score  # torch.uint8 (byte) tensor, for indexing
                n_above_min_score = score_above_min_score.sum().item()
                if n_above_min_score == 0:
                    continue
                class_scores = class_scores[score_above_min_score]  # (n_qualified)
                class_decoded_locs = decoded_locs[score_above_min_score]  # (n_qualified, 4)

                # Sort predicted boxes and scores by scores
                class_scores, sort_ind = class_scores.sort(dim=0, descending=True)  # (n_qualified), (n_min_score)
                class_decoded_locs = class_decoded_locs[sort_ind]  # (n_min_score, 4)

                # Find the overlap between predicted boxes
                overlap = find_jaccard_overlap(class_decoded_locs, class_decoded_locs)  # (n_qualified, n_min_score)

                # Non-Maximum Suppression (NMS)

                # A torch.uint8 (byte) tensor to keep track of which predicted boxes to suppress
                # 1 implies suppress, 0 implies don't suppress
                suppress = torch.zeros((n_above_min_score), dtype=torch.uint8).to(device)  # (n_qualified)

                # Consider each box in order of decreasing scores
                for box in range(class_decoded_locs.size(0)):
                    # If this box is already marked for suppression
                    if suppress[box] == 1:
                        continue

                    # Suppress boxes whose overlaps (with this box) are greater than maximum overlap
                    # Find such boxes and update suppress indices
                    suppress = torch.max(suppress, overlap[box] > max_overlap)
                    # The max operation retains previously suppressed boxes, like an 'OR' operation

                    # Don't suppress this box, even though it has an overlap of 1 with itself
                    suppress[box] = 0

                # Store only unsuppressed boxes for this class
                image_boxes.append(class_decoded_locs[1 - suppress])
                image_labels.append(torch.LongTensor((1 - suppress).sum().item() * [c]).to(device))
                image_scores.append(class_scores[1 - suppress])

            # If no object in any class is found, store a placeholder for 'background'
            if len(image_boxes) == 0:
                image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(device))
                image_labels.append(torch.LongTensor([0]).to(device))
                image_scores.append(torch.FloatTensor([0.]).to(device))

            # Concatenate into single tensors
            image_boxes = torch.cat(image_boxes, dim=0)  # (n_objects, 4)

            image_labels = torch.cat(image_labels, dim=0)  # (n_objects)
            image_scores = torch.cat(image_scores, dim=0)  # (n_objects)
            n_objects = image_scores.size(0)

            # Keep only the top k objects
            if n_objects > top_k:
                image_scores, sort_ind = image_scores.sort(dim=0, descending=True)
                image_scores = image_scores[:top_k]  # (top_k)
                image_boxes = image_boxes[sort_ind][:top_k]  # (top_k, 4)
                image_labels = image_labels[sort_ind][:top_k]  # (top_k)

            # Append to lists that store predicted boxes and scores for all images
            all_images_boxes.append(image_boxes)
            all_images_labels.append(image_labels)
            all_images_scores.append(image_scores)

        return all_images_boxes, all_images_labels, all_images_scores  # lists of length batch_size


