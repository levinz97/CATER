from typing import Optional
import torch

from detectron2.layers import cat, Conv2d
from detectron2.utils.registry import Registry
from detectron2.config import CfgNode

from .layers import conv_bn_relu

ROI_COORDINATE_HEAD_REGISTRY = Registry('ROI_COORDINATE_HEAD_REGISTRY')

def coordinate_loss(pred_coordinate3d, instances):
    diff = pred_coordinate3d - cat([x.gt_coordinate for x in instances])
    return {"coordinate_loss": torch.square(diff).sum()}

@ROI_COORDINATE_HEAD_REGISTRY.register()
class coordinateHead(torch.Module):
    """
    A Fully convolutional coordinate head, apply average pooling at last to get 3d coordinate
    """
    def __init__(self, cfg: CfgNode, input_channels: int):
        super().__init__()
        kernel_size          = cfg.MODEL.ROI_COORDINATE_HEAD.CONV_HEAD_KERNEL_SIZE
        # hidden_dim           = cfg.MODEL.ROI_COORDINATE_HEAD.CONV_HEAD_DIM  
        self.n_stacked_convs = cfg.MODEL.ROI_COORDINATE_HEAD.NUM_STACKED_CONVS
        padding = kernel_size // 2
        for i in range(self.n_stacked_convs):
            # use custom layer instead of detectron2 wrapper layer due to compatible reason 
            layer = conv_bn_relu(input_channels, input_channels, kernel_size, stride=1, padding = padding )
            layer_name = self._name_layers(i)
            self.add_module(layer_name, layer)
        self.conv_bn_relu_last = conv_bn_relu(input_channels, 3, kernel_size=1)
        self.avg_pooling_layer = torch.nn.AdaptiveAvgPool2d((1,1))
                   
    def _name_layers(self, i:int):
        return "conv_bn_relu{}".format(i+1)

    def forward(self, features: torch.Tensor):
        x = features
        for i in range(self.n_stacked_convs):
            layer_name = self._name_layers(i)
            x = getattr(self, layer_name)(x)
        x = self.conv_bn_relu_last(x)
        x = self.avg_pooling_layer(x)
        return x


def build_coordinate_head(cfg:CfgNode, input_channels:int):
    """
    Build 3D coordinate head based on configs
    """
    name = cfg.MODEL.ROI_COORDINATE_HEAD.NAME
    return ROI_COORDINATE_HEAD_REGISTRY.get(name)(cfg, input_channels)