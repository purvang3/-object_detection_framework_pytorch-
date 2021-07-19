import warnings
from collections import namedtuple
from math import sqrt
from typing import Callable, Any, Optional, Tuple, List

import torch
import torch.nn.functional as F
from build.utils import *
from torch import nn, Tensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

__all__ = ['Inception3', 'inception_v3', 'InceptionOutputs', '_InceptionOutputs']

model_urls = {
    # Inception v3 ported from TensorFlow
    'inception_v3_google': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
}

InceptionOutputs = namedtuple('InceptionOutputs', ['logits', 'aux_logits'])
InceptionOutputs.__annotations__ = {'logits': Tensor, 'aux_logits': Optional[Tensor]}

# Script annotations failed with _GoogleNetOutputs = namedtuple ...
# _InceptionOutputs set here for backwards compat
_InceptionOutputs = InceptionOutputs

activation = torch.nn.ReLU()


class Inception3(nn.Module):

    def __init__(
            self,
            num_classes: int = 1000,
            aux_logits: bool = True,
            transform_input: bool = False,
            inception_blocks: Optional[List[Callable[..., nn.Module]]] = None,
            init_weights: Optional[bool] = None
    ) -> None:
        super(Inception3, self).__init__()
        if inception_blocks is None:
            inception_blocks = [
                BasicConv2d, InceptionA, InceptionB, InceptionC,
                InceptionD, InceptionE, InceptionAux
            ]
        if init_weights is None:
            warnings.warn('The default weight initialization of inception_v3 will be changed in future releases of '
                          'torchvision. If you wish to keep the old behavior (which leads to long initialization times'
                          ' due to scipy/scipy#11299), please set init_weights=True.', FutureWarning)
            init_weights = True
        assert len(inception_blocks) == 7
        conv_block = inception_blocks[0]
        inception_a = inception_blocks[1]
        inception_b = inception_blocks[2]
        inception_c = inception_blocks[3]
        inception_d = inception_blocks[4]
        inception_e = inception_blocks[5]
        inception_aux = inception_blocks[6]

        self.aux_logits = aux_logits
        self.transform_input = transform_input
        self.Conv2d_1a_3x3 = conv_block(3, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = conv_block(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = conv_block(32, 64, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.Conv2d_3b_1x1 = conv_block(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = conv_block(80, 192, kernel_size=3)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.Mixed_5b = inception_a(192, pool_features=32)
        self.Mixed_5c = inception_a(256, pool_features=64)
        self.Mixed_5d = inception_a(288, pool_features=64)
        self.Mixed_6a = inception_b(288)
        self.Mixed_6b = inception_c(768, channels_7x7=128)
        self.Mixed_6c = inception_c(768, channels_7x7=160)
        self.Mixed_6d = inception_c(768, channels_7x7=160)
        self.Mixed_6e = inception_c(768, channels_7x7=192)
        self.AuxLogits: Optional[nn.Module] = None
        if aux_logits:
            self.AuxLogits = inception_aux(768, num_classes)
        self.Mixed_7a = inception_d(768)
        self.Mixed_7b = inception_e(1280)
        self.Mixed_7c = inception_e(2048)

        self.detection = True
        if not self.detection:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.dropout = nn.Dropout()
            self.fc = nn.Linear(2048, num_classes)

        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    import scipy.stats as stats
                    stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                    X = stats.truncnorm(-2, 2, scale=stddev)
                    values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
                    values = values.view(m.weight.size())
                    with torch.no_grad():
                        m.weight.copy_(values)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        else:
            self.load_pretrained_layers()

    def _transform_input(self, x: Tensor) -> Tensor:
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x

    def _forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor]]:
        # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = self.maxpool1(x)
        # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = self.maxpool2(x)
        # N x 192 x 35 x 35
        x = self.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.Mixed_5c(x)
        # N x 288 x 35 x 35
        x_5d = self.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6d(x)
        # N x 768 x 17 x 17
        x_6e = self.Mixed_6e(x)
        # N x 768 x 17 x 17
        aux: Optional[Tensor] = None
        if self.AuxLogits is not None:
            if self.training:
                aux = self.AuxLogits(x)
        # N x 768 x 17 x 17
        x = self.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x_7c = self.Mixed_7c(x)
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        # x = self.avgpool(x)
        # # N x 2048 x 1 x 1
        # x = self.dropout(x)
        # # N x 2048 x 1 x 1
        # x = torch.flatten(x, 1)
        # # N x 2048
        # x = self.fc(x)
        # # N x 1000 (num_classes)
        return x_5d, x_6e, x_7c, aux

    @torch.jit.unused
    def eager_outputs(self, x: Tuple[Tensor, Tensor, Tensor], aux: Optional[Tensor]) -> InceptionOutputs:
        if self.training and self.aux_logits:
            return InceptionOutputs(x, aux)
        else:
            return x  # type: ignore[return-value]

    def forward(self, x: Tensor) -> InceptionOutputs:
        x = self._transform_input(x)
        x_5d, x_6e, x_7c, aux = self._forward(x)
        aux_defined = self.training and self.aux_logits
        if torch.jit.is_scripting():
            if not aux_defined:
                warnings.warn("Scripted Inception3 always returns Inception3 Tuple")
            return InceptionOutputs(x, aux)
        else:
            return self.eager_outputs((x_5d, x_6e, x_7c), aux)

    def load_pretrained_layers(self):
        # Current state of base
        state_dict = self.state_dict()
        param_names = list(state_dict.keys())

        try:
            from torch.hub import load_state_dict_from_url
        except ImportError:
            from torch.utils.model_zoo import load_url as load_state_dict_from_url

        pretrained_state_dict = load_state_dict_from_url(model_urls['inception_v3_google'],
                                                         progress=True)
        pretrained_param_names = list(pretrained_state_dict.keys())

        for i, param in enumerate(pretrained_param_names):  # excluding conv6 and conv7 parameters
            if "fc" or "AuxLogits" in param:
                continue
            state_dict[param] = pretrained_state_dict[pretrained_param_names[i]]

        self.load_state_dict(state_dict)


class InceptionA(nn.Module):

    def __init__(
            self,
            in_channels: int,
            pool_features: int,
            conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InceptionA, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = conv_block(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = conv_block(48, 64, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = conv_block(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv_block(96, 96, kernel_size=3, padding=1)

        self.branch_pool = conv_block(in_channels, pool_features, kernel_size=1)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionB(nn.Module):

    def __init__(
            self,
            in_channels: int,
            conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InceptionB, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch3x3 = conv_block(in_channels, 384, kernel_size=3, stride=2)

        self.branch3x3dbl_1 = conv_block(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv_block(96, 96, kernel_size=3, stride=2)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)

        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionC(nn.Module):

    def __init__(
            self,
            in_channels: int,
            channels_7x7: int,
            conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InceptionC, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 192, kernel_size=1)

        c7 = channels_7x7
        self.branch7x7_1 = conv_block(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = conv_block(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = conv_block(c7, 192, kernel_size=(7, 1), padding=(3, 0))

        self.branch7x7dbl_1 = conv_block(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = conv_block(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = conv_block(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = conv_block(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = conv_block(c7, 192, kernel_size=(1, 7), padding=(0, 3))

        self.branch_pool = conv_block(in_channels, 192, kernel_size=1)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionD(nn.Module):

    def __init__(
            self,
            in_channels: int,
            conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InceptionD, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch3x3_1 = conv_block(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = conv_block(192, 320, kernel_size=3, stride=2)

        self.branch7x7x3_1 = conv_block(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = conv_block(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = conv_block(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = conv_block(192, 192, kernel_size=3, stride=2)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionE(nn.Module):

    def __init__(
            self,
            in_channels: int,
            conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InceptionE, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 320, kernel_size=1)

        self.branch3x3_1 = conv_block(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = conv_block(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = conv_block(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3dbl_1 = conv_block(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = conv_block(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = conv_block(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = conv_block(in_channels, 192, kernel_size=1)

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):

    def __init__(
            self,
            in_channels: int,
            num_classes: int,
            conv_block: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InceptionAux, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.conv0 = conv_block(in_channels, 128, kernel_size=1)
        self.conv1 = conv_block(128, 768, kernel_size=5)
        self.conv1.stddev = 0.01  # type: ignore[assignment]
        self.fc = nn.Linear(768, num_classes)
        self.fc.stddev = 0.001  # type: ignore[assignment]

    def forward(self, x: Tensor) -> Tensor:
        # N x 768 x 17 x 17
        x = F.avg_pool2d(x, kernel_size=5, stride=3)
        # N x 768 x 5 x 5
        x = self.conv0(x)
        # N x 128 x 5 x 5
        x = self.conv1(x)
        # N x 768 x 1 x 1
        # Adaptive average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # N x 768 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 768
        x = self.fc(x)
        # N x 1000
        return x


class BasicConv2d(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            **kwargs: Any
    ) -> None:
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return activation(x)


class AuxiliaryConvolutions(nn.Module):
    """
    Additional convolutions to produce higher-level feature maps.
    """

    def __init__(self, input_channels=1024):
        super(AuxiliaryConvolutions, self).__init__()
        self.input_channels = input_channels
        self.layer_depth = [512, 256, 128]

        # Auxiliary/additional convolutions on top of the VGG base
        self.conv8_1 = nn.Conv2d(self.input_channels, 256, kernel_size=1, padding=0)  # stride = 1, by default

        self.conv8_1 = nn.Conv2d(2048, 512, kernel_size=1, padding=0, stride=1)
        self.conv8_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=2)

        self.conv9_1 = nn.Conv2d(512, 256, kernel_size=1, padding=0, stride=1)
        self.conv9_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2)

        self.conv10_1 = nn.Conv2d(256, 128, kernel_size=1, padding=0, stride=1)
        self.conv10_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2)

        # Initialize convolutions' parameters
        self.init_conv2d()

    def init_conv2d(self):
        """
        Initialize convolution parameters.
        """
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)

    def forward(self, Mixed_5d, Mixed_6e, Mixed_7c):
        """
        Forward propagation.
        :param conv7_feats: lower-level feature maps
        :return: higher-level feature maps
        """

        Conv_8_1 = activation(self.conv8_1(Mixed_7c))  # (N, 256, 19, 19)
        Conv_8_2 = activation(self.conv8_2(Conv_8_1))  # (N, 512, 10, 10)

        Conv_9_1 = activation(self.conv9_1(Conv_8_2))  # (N, 128, 10, 10)
        Conv_9_2 = activation(self.conv9_2(Conv_9_1))  # (N, 256, 5, 5)

        Conv_10_1 = activation(self.conv10_1(Conv_9_2))  # (N, 128, 5, 5)
        Conv_10_2 = activation(self.conv10_2(Conv_10_1))  # (N, 256, 3, 3)

        # Higher-level feature maps
        return Mixed_5d, Mixed_6e, Mixed_7c, Conv_8_2, Conv_9_2, Conv_10_2


class PredictionConvolutions(nn.Module):
    """
    Convolutions to predict class scores and bounding boxes using lower and higher-level feature maps.
    The bounding boxes (locations) are predicted as encoded offsets w.r.t each of the prior (default) boxes.
    See 'cxcy_to_gcxgcy' in utils.py for the encoding definition.
    The class scores represent the scores of each object class in each of the bounding boxes located.
    A high score for 'background' = no object.
    """

    def __init__(self, n_classes, input_channels=128):
        """
        :param n_classes: number of different types of objects
        """
        super(PredictionConvolutions, self).__init__()

        self.n_classes = n_classes
        self.input_channels = input_channels

        # Number of prior-boxes we are considering per position in each feature map
        n_boxes = {'Mixed_5d': 4,
                   'Mixed_6e': 6,
                   'Mixed_7c': 6,
                   'Conv_8_2': 6,
                   'Conv_9_2': 4,
                   'Conv_10_2': 4}

        # Localization prediction convolutions (predict offsets w.r.t prior-boxes)
        self.loc_Mixed_5d = nn.Conv2d(288, n_boxes['Mixed_5d'] * 4, kernel_size=3, padding=1)
        self.loc_Mixed_6e = nn.Conv2d(768, n_boxes['Mixed_6e'] * 4, kernel_size=3, padding=1)
        self.loc_Mixed_7c = nn.Conv2d(2048, n_boxes['Mixed_7c'] * 4, kernel_size=3, padding=1)
        self.loc_Conv_8_2 = nn.Conv2d(512, n_boxes['Conv_8_2'] * 4, kernel_size=3, padding=1)
        self.loc_Conv_9_2 = nn.Conv2d(256, n_boxes['Conv_9_2'] * 4, kernel_size=3, padding=1)
        self.loc_Conv_10_2 = nn.Conv2d(128, n_boxes['Conv_10_2'] * 4, kernel_size=3, padding=1)

        # Class prediction convolutions (predict classes in localization boxes)
        self.cl_Mixed_5d = nn.Conv2d(288, n_boxes['Mixed_5d'] * n_classes, kernel_size=3, padding=1)
        self.cl_Mixed_6e = nn.Conv2d(768, n_boxes['Mixed_6e'] * n_classes, kernel_size=3, padding=1)
        self.cl_Mixed_7c = nn.Conv2d(2048, n_boxes['Mixed_7c'] * n_classes, kernel_size=3, padding=1)
        self.cl_Conv_8_2 = nn.Conv2d(512, n_boxes['Conv_8_2'] * n_classes, kernel_size=3, padding=1)
        self.cl_Conv_9_2 = nn.Conv2d(256, n_boxes['Conv_9_2'] * n_classes, kernel_size=3, padding=1)
        self.cl_Conv_10_2 = nn.Conv2d(128, n_boxes['Conv_10_2'] * n_classes, kernel_size=3, padding=1)

        # Initialize convolutions' parameters
        self.init_conv2d()

    def init_conv2d(self):
        """
        Initialize convolution parameters.
        """
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)

    def forward(self, Mixed_5d, Mixed_6e, Mixed_7c, Conv_8_2, Conv_9_2,
                Conv_10_2):  # , conv9_2_feats, conv10_2_feats, conv11_2_feats):
        """
        Forward propagation.
        :param Mixed_5d: conv4_3 feature map, a tensor
        :param Mixed_6e: conv7 feature map, a tensor
        :param Mixed_7c: conv8_2 feature map, a tensor
        :param Conv_8_2: conv9_2 feature map, a tensor
        :param Conv_9_2: conv10_2 feature map, a tensor
        :param Conv_10_2: conv11_2 feature map, a tensor

        :return: bbox locations and class scores (i.e. w.r.t each prior box) for each image
        """
        batch_size = Mixed_5d.size(0)

        # Predict localization boxes' bounds (as offsets w.r.t prior-boxes)
        Mixed_5d_LOC = self.loc_Mixed_5d(Mixed_5d)
        Mixed_5d_LOC = Mixed_5d_LOC.permute(0, 2, 3, 1).contiguous()
        Mixed_5d_LOC = Mixed_5d_LOC.view(batch_size, -1, 4)

        Mixed_6e_LOC = self.loc_Mixed_6e(Mixed_6e)
        Mixed_6e_LOC = Mixed_6e_LOC.permute(0, 2, 3, 1).contiguous()
        Mixed_6e_LOC = Mixed_6e_LOC.view(batch_size, -1,
                                         4)

        Mixed_7c_LOC = self.loc_Mixed_7c(Mixed_7c)
        Mixed_7c_LOC = Mixed_7c_LOC.permute(0, 2, 3, 1).contiguous()
        Mixed_7c_LOC = Mixed_7c_LOC.view(batch_size, -1, 4)

        Conv_8_2_LOC = self.loc_Conv_8_2(Conv_8_2)
        Conv_8_2_LOC = Conv_8_2_LOC.permute(0, 2, 3, 1).contiguous()
        Conv_8_2_LOC = Conv_8_2_LOC.view(batch_size, -1, 4)

        Conv_9_2_LOC = self.loc_Conv_9_2(Conv_9_2)
        Conv_9_2_LOC = Conv_9_2_LOC.permute(0, 2, 3, 1).contiguous()
        Conv_9_2_LOC = Conv_9_2_LOC.view(batch_size, -1, 4)

        Conv_10_2_LOC = self.loc_Conv_10_2(Conv_10_2)
        Conv_10_2_LOC = Conv_10_2_LOC.permute(0, 2, 3, 1).contiguous()
        Conv_10_2_LOC = Conv_10_2_LOC.view(batch_size, -1, 4)

        # Predict classes in localization boxes
        Mixed_5d_CL = self.cl_Mixed_5d(Mixed_5d)
        Mixed_5d_CL = Mixed_5d_CL.permute(0, 2, 3, 1).contiguous()
        Mixed_5d_CL = Mixed_5d_CL.view(batch_size, -1,
                                       self.n_classes)

        Mixed_6e_CL = self.cl_Mixed_6e(Mixed_6e)
        Mixed_6e_CL = Mixed_6e_CL.permute(0, 2, 3, 1).contiguous()
        Mixed_6e_CL = Mixed_6e_CL.view(batch_size, -1, self.n_classes)

        Mixed_7c_CL = self.cl_Mixed_7c(Mixed_7c)
        Mixed_7c_CL = Mixed_7c_CL.permute(0, 2, 3, 1).contiguous()
        Mixed_7c_CL = Mixed_7c_CL.view(batch_size, -1, self.n_classes)

        Conv_8_2_CL = self.cl_Conv_8_2(Conv_8_2)
        Conv_8_2_CL = Conv_8_2_CL.permute(0, 2, 3, 1).contiguous()
        Conv_8_2_CL = Conv_8_2_CL.view(batch_size, -1, self.n_classes)

        Conv_9_2_CL = self.cl_Conv_9_2(Conv_9_2)
        Conv_9_2_CL = Conv_9_2_CL.permute(0, 2, 3, 1).contiguous()
        Conv_9_2_CL = Conv_9_2_CL.view(batch_size, -1, self.n_classes)

        Conv_10_2_CL = self.cl_Conv_10_2(Conv_10_2)
        Conv_10_2_CL = Conv_10_2_CL.permute(0, 2, 3, 1).contiguous()
        Conv_10_2_CL = Conv_10_2_CL.view(batch_size, -1, self.n_classes)

        # Concatenate in this specific order (i.e. must match the order of the prior-boxes)
        locs = torch.cat([Mixed_5d_LOC, Mixed_6e_LOC, Mixed_7c_LOC, Conv_8_2_LOC, Conv_9_2_LOC, Conv_10_2_LOC],
                         dim=1)
        classes_scores = torch.cat([Mixed_5d_CL, Mixed_6e_CL, Mixed_7c_CL, Conv_8_2_CL, Conv_9_2_CL, Conv_10_2_CL],
                                   dim=1)

        return locs, classes_scores


class Inception_V3(nn.Module):
    def __init__(self, args, aux_logits=False, init_weights=False):
        super(Inception_V3, self).__init__()
        global activation
        self.n_classes = args.n_classes
        self.aux_logits = aux_logits
        self.init_weights = init_weights
        activation = get_activation_function(name=args.activation)

        self.base = Inception3(aux_logits=self.aux_logits, init_weights=self.init_weights)

        if args.freeze_backbone:
            for param in self.base.parameters():
                param.requires_grad = False

        self.aux_convs = AuxiliaryConvolutions(input_channels=2048)
        self.pred_convs = PredictionConvolutions(self.n_classes, input_channels=128)

        # Prior boxes
        self.priors_cxcy = self.create_prior_boxes()

    def forward(self, image):
        """
        Forward propagation.
        :param image: images, a tensor of dimensions (N, 3, 300, 300)
        :return:  locations and class scores (i.e. w.r.t each prior box) for each image
        """
        x = self.base(image)
        Mixed_5d, Mixed_6e, Mixed_7c = x
        Mixed_5d, Mixed_6e, Mixed_7c, Conv_8_2, Conv_9_2, Conv_10_2 = self.aux_convs(Mixed_5d, Mixed_6e, Mixed_7c)
        locs, classes_scores = self.pred_convs(Mixed_5d, Mixed_6e, Mixed_7c, Conv_8_2, Conv_9_2, Conv_10_2)

        return locs, classes_scores

    def create_prior_boxes(self):
        """
        Create the prior (default) boxes for the SSD300, as defined in the paper.
        :return: prior boxes in center-size coordinates
        """
        # TODO: need to automate for any image shape
        fmap_dims = {'Mixed_5d': 35,
                     'Mixed_6e': 17,
                     'Mixed_7c': 8,
                     'Conv_8_2': 4,
                     'Conv_9_2': 2,
                     'Conv_10_2': 1}

        obj_scales = {'Mixed_5d': 0.1,
                      'Mixed_6e': 0.2,
                      'Mixed_7c': 0.375,
                      'Conv_8_2': 0.55,
                      'Conv_9_2': 0.70,
                      'Conv_10_2': 0.95}

        aspect_ratios = {'Mixed_5d': [1., 2., 0.5],
                         'Mixed_6e': [1., 2., 3., 0.5, .333],
                         'Mixed_7c': [1., 2., 3., 0.5, .333],
                         'Conv_8_2': [1., 2., 3., 0.5, .333],
                         'Conv_9_2': [1., 2., 0.5],
                         'Conv_10_2': [1., 2., 0.5]}

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

        prior_boxes = torch.FloatTensor(prior_boxes).to(device)
        prior_boxes.clamp_(0, 1)

        return prior_boxes

    def detect_objects(self, predicted_locs, predicted_scores, min_score, max_overlap, top_k):
        """
        Decipher the bbox locations and class scores to detect objects.
        For each class, perform Non-Maximum Suppression (NMS) on boxes that are above a minimum threshold.
        :param predicted_locs: predicted locations/boxes w.r.t the prior boxes, a tensor of dimensions
        :param predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions
        :param min_score: minimum threshold for a box to be considered a match for a certain class
        :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via NMS
        :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
        :return: detections (boxes, labels, and scores), lists of length batch_size
        """
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        predicted_scores = F.softmax(predicted_scores, dim=2)

        # Lists to store final predicted boxes, labels, and scores for all images
        all_images_boxes = list()
        all_images_labels = list()
        all_images_scores = list()

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        for i in range(batch_size):
            # Decode object coordinates from the form we regressed predicted boxes to
            if self.args.dataset == "coco":
                decoded_locs = gcxgcy_to_cxcy(predicted_locs[i], self.priors_cxcy)
            else:
                decoded_locs = cxcy_to_xy(gcxgcy_to_cxcy(predicted_locs[i], self.priors_cxcy))

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
                class_scores = class_scores[score_above_min_score]
                class_decoded_locs = decoded_locs[score_above_min_score]

                # Sort predicted boxes and scores by scores
                class_scores, sort_ind = class_scores.sort(dim=0, descending=True)
                class_decoded_locs = class_decoded_locs[sort_ind]

                # Find the overlap between predicted boxes
                overlap = find_jaccard_overlap(class_decoded_locs, class_decoded_locs)

                # Non-Maximum Suppression (NMS)

                # A torch.uint8 (byte) tensor to keep track of which predicted boxes to suppress
                # 1 implies suppress, 0 implies don't suppress
                suppress = torch.zeros((n_above_min_score), dtype=torch.uint8).to(device)

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
            if self.args.dataset == "coco":
                image_boxes = image_boxes * torch.tensor([self.args.img_width, self.args.img_height, self.args.img_width,
                                                          self.args.img_height], dtype=torch.float32).to(device)
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


