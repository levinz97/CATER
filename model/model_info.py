import os
from detectron2.config import get_cfg
import sys
sys.path.append('..')
from config.cater_config import add_cater_config

import torchvision.models as models
import torch
from ptflops import get_model_complexity_info
from model.coordinate_head import coordinateHead
from model.layers import GroupedDilatedConv, GroupedDilatedConvV2, Decoder

with torch.cuda.device(0):
#   net = models.densenet161()
    # net = models.resnet18()
    cfg = get_cfg()
    cfg.merge_from_file(os.path.join('..', 'detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'))
    add_cater_config(cfg)
    cfg.merge_from_file(os.path.join('..', 'config','Cater.yaml'))
    net = coordinateHead(cfg, 14)
    in_channels = 122
    # net = GroupedDilatedConv(in_channels, 2*in_channels, 3, 2)
    # net = Decoder(256,5)

    macs, params = get_model_complexity_info(net, (14, 128, 128), as_strings=True,
                                             print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))