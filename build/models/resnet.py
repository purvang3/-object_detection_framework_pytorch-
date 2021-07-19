from math import sqrt

import torch
import torch.nn as nn
from build.utils import *
from torch import Tensor

activation = torch.nn.ReLU()

from torch.hub import load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            layers: List[int],
            num_classes: int = 1000,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        # TODO: if not pretrained then uncomment
        self.detection = True
        if not self.detection:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        # TODO: if not pretrained then only initialize
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # TODO: if not pretrained then only initialize
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
        arch: str,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        pretrained: bool,
        progress: bool,
        **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict, strict=False)
    return model


def resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


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
        :param conv7_feats: lower-level conv7 feature map , 19)
        :return: higher-level feature maps conv8_2, conv9_2, conv10_2, and conv11_2
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
        :param Mixed_5d: conv4_3 feature map  38)
        :param Mixed_6e: conv7 feature map , 19)
        :param Mixed_7c: conv8_2 feature map  10)
        :param Conv_8_2: conv9_2 feature map 5)
        :param Conv_9_2: conv10_2 feature map 3)
        :param Conv_10_2: conv11_2 feature map 1)

        :return: locations and class scores (i.e. w.r.t each prior box) for each image
        """
        batch_size = Mixed_5d.size(0)

        # Predict localization boxes' bounds (as offsets w.r.t prior-boxes)
        Mixed_5d_LOC = self.loc_Mixed_5d(Mixed_5d)
        Mixed_5d_LOC = Mixed_5d_LOC.permute(0, 2, 3,
                                            1).contiguous()
        Mixed_5d_LOC = Mixed_5d_LOC.view(batch_size, -1,
                                         4)

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
        Mixed_5d_CL = Mixed_5d_CL.permute(0, 2, 3,
                                          1).contiguous()
        Mixed_5d_CL = Mixed_5d_CL.view(batch_size, -1,
                                       self.n_classes)
        Mixed_6e_CL = self.cl_Mixed_6e(Mixed_6e)
        Mixed_6e_CL = Mixed_6e_CL.permute(0, 2, 3, 1).contiguous()
        Mixed_6e_CL = Mixed_6e_CL.view(batch_size, -1,
                                       self.n_classes)
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


class ResNetModel(nn.Module):
    def __init__(self, args, aux_logits=False, init_weights=False):
        super(ResNetModel, self).__init__()
        global activation
        self.n_classes = args.n_classes
        self.aux_logits = aux_logits
        self.init_weights = init_weights
        activation = get_activation_function(name=args.activation)

        self.base = resnet50(pretrained=True, progress=True)

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
        :param image: images 300)
        :return: locations and class scores (i.e. w.r.t each prior box) for each image
        """
        x = self.base(image)
        Mixed_5d, Mixed_6e, Mixed_7c = x
        Mixed_5d, Mixed_6e, Mixed_7c, Conv_8_2, Conv_9_2, Conv_10_2 = self.aux_convs(Mixed_5d, Mixed_6e, Mixed_7c)
        locs, classes_scores = self.pred_convs(Mixed_5d, Mixed_6e, Mixed_7c, Conv_8_2, Conv_9_2, Conv_10_2)

        return locs, classes_scores

    def create_prior_boxes(self):
        """
        Create the  prior (default) boxes.
        :return: prior boxes in center-size coordinates       """
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


if __name__ == "__main__":
    resnet50(pretrained=True, progress=True)
