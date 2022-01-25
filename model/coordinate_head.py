from typing import Optional
import torch

from detectron2.layers import cat, Conv2d
from detectron2.utils.registry import Registry
from detectron2.config import CfgNode

from .layers import conv_bn_relu, GroupedDilatedConv, GroupedDilatedConvV2, DilatedResNextBlock

ROI_COORDINATE_HEAD_REGISTRY = Registry('ROI_COORDINATE_HEAD_REGISTRY')

def coordinate_loss(pred_coordinate3d, instances):
    coordinate3d = cat([x.gt_coordinate for x in instances])
    diff = pred_coordinate3d - coordinate3d
    return {"coordinate_loss": torch.square(diff).mean()}

@ROI_COORDINATE_HEAD_REGISTRY.register()
class coordinateHead(torch.nn.Module):
    """
    A Fully convolutional coordinate head except applying average pooling at last to get 3d coordinate
    """
    def __init__(self, cfg: CfgNode, input_channels: int):
        super().__init__()
        kernel_size          = cfg.MODEL.ROI_COORDINATE_HEAD.CONV_HEAD_KERNEL_SIZE
        # hidden_dim           = cfg.MODEL.ROI_COORDINATE_HEAD.CONV_HEAD_DIM  
        self.n_stacked_convs = cfg.MODEL.ROI_COORDINATE_HEAD.NUM_STACKED_CONVS
        dilation             = cfg.MODEL.ROI_COORDINATE_HEAD.DILATION
        padding = (kernel_size-1) * dilation // 2
        for i in range(self.n_stacked_convs):
            # use custom layer instead of detectron2 wrapper layer due to compatible reason 
            # layer = conv_bn_relu(input_channels * (2**i), input_channels * (2**(i+1)), kernel_size, stride=2, padding=padding, dilation=dilation)
            # layer = GroupedDilatedConv(input_channels*(2**i), input_channels * (2**(i+1)), kernel_size, dilations=[2,3],stride=2)
            dilations = [1,2,3,4]
            # dilations = [2,2,2,2,2,2,2]
            layer = GroupedDilatedConvV2(input_channels*(2**i), input_channels * (2**(i+1)), kernel_size, stride=2, dilations=dilations)
            layer_name = self._name_layers(i)
            self.add_module(layer_name, layer)
        # self.conv_bn_relu_last = conv_bn_relu(input_channels, 3, kernel_size=1)
        final_num_channels = input_channels * (2**(self.n_stacked_convs))
        cardinality = 32
        self.dilated_resnext_block = DilatedResNextBlock(final_num_channels, bottleneck_width=final_num_channels//cardinality, cardinality=cardinality, expansion=2)
        self.avg_pooling_layer = torch.nn.AdaptiveAvgPool2d((1,1))
        self.linear = torch.nn.Linear(2*final_num_channels, 3)
                   
    def _name_layers(self, i:int):
        return "conv_bn_relu{}".format(i+1)

    def forward(self, features: torch.Tensor):
        x = features
        for i in range(self.n_stacked_convs):
            layer_name = self._name_layers(i)
            x = getattr(self, layer_name)(x)
        # x = self.conv_bn_relu_last(x)
        x = self.dilated_resnext_block(x)
        x = self.avg_pooling_layer(x)
        x = torch.squeeze(x)
        x = self.linear(x)
        return x


def build_coordinate_head(cfg:CfgNode, input_channels:int):
    """
    Build 3D coordinate head based on configs
    """
    name = cfg.MODEL.ROI_COORDINATE_HEAD.NAME
    return ROI_COORDINATE_HEAD_REGISTRY.get(name)(cfg, input_channels)