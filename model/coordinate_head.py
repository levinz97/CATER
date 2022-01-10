from typing import Optional
import torch

from detectron2.layers import cat, Conv2d
from detectron2.utils.registry import Registry
from detectron2.config import CfgNode

ROI_COORDINATE_HEAD_REGISTRY = Registry('ROI_COORDINATE_HEAD_REGISTRY')

def coordinate_loss(pred_coordinate3d, instances):
    diff = pred_coordinate3d - cat([x.gt_coordinate for x in instances])
    return {"coordinate_loss": torch.square(diff).sum()}




def build_coordinate_head(cfg:CfgNode, input_shape):
    """
    Build 3D coordinate head based on configs
    """
    name = cfg.MODEL.ROI_COORDINATE_HEAD.NAME
    return ROI_COORDINATE_HEAD_REGISTRY.get(name)(cfg, input_shape)