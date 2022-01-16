from typing import Dict, List, Optional
from detectron2.layers import ShapeSpec
from detectron2.layers.wrappers import cat
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads import select_foreground_proposals

from detectron2.modeling import StandardROIHeads, ROI_HEADS_REGISTRY

from detectron2.structures import ImageList, Instances
import torch
from torch import nn

from .coordinate_head import (
    build_coordinate_head,
    coordinate_loss
    )

from utils import dispImg

@ROI_HEADS_REGISTRY.register()
class CaterROIHeads(StandardROIHeads):
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        self.separate_attrpred_on = cfg.MODEL.SEPARATE_ATTR_ON
        if self.separate_attrpred_on:
            self._init_colormaterial_head(cfg, input_shape)
            self._init_shape_head(cfg, input_shape)
            self._init_size_head(cfg, input_shape)
        
        self._init_coordinate_head(cfg, input_shape)
        self.pixel_mean = cfg.MODEL.PIXEL_MEAN

        
    def _init_colormaterial_head(self, cfg, input_shape):
        pass
    
    def _init_shape_head(self, cfg, input_shape):
        pass

    def _init_size_head(self, cfg, input_shape):
        pass

    def _init_coordinate_head(self, cfg, input_shape):
        self.use_backbone_features       = cfg.MODEL.ROI_COORDINATE_HEAD.USE_BACKBONE_FEATURES
        coordinate_in_features           = cfg.MODEL.ROI_COORDINATE_HEAD.IN_FEATURES if self.use_backbone_features else None
        self.img_size                    = cfg.MODEL.ROI_COORDINATE_HEAD.IMG_SIZE
        if not self.use_backbone_features:
            in_channels = 6 # raw image 3 + cropped image within bbox 3
        else:
            coordinate_pooler_resolution     = cfg.MODEL.ROI_COORDINATE_HEAD.POOLER_RESOLUTION
            coordinate_pooler_sampling_ratio = cfg.MODEL.ROI_COORDINATE_HEAD.POOLER_SAMPLING_RATIO
            coordinate_pooler_type           = cfg.MODEL.ROI_COORDINATE_HEAD.POOLER_TYPE
            coordinate_pooler_scale  = tuple(1.0 / input_shape[k].stride for k in coordinate_in_features)
            self.coordinate_pooler = ROIPooler(
                output_size=coordinate_pooler_resolution,
                scales=coordinate_pooler_scale,
                sampling_ratio=coordinate_pooler_sampling_ratio,
                pooler_type=coordinate_pooler_type
            )
            in_channels = [input_shape[f].channels for f in coordinate_in_features][0]
            raise NotImplementedError
        self.coordinate_head = build_coordinate_head(cfg, in_channels)

    
    def _forward_color_material(self, features: Dict[str, torch.Tensor], instances: List[Instances]):
        if not self.separate_attrpred_on:
            return {} if self.training else instances
    

    def _forward_coordinate(self, features: Dict[str, torch.Tensor], instances: List[Instances], images: ImageList):
        """
        Forward the 3d coordinate here

        Args:
            features (dict[str, Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            instances (list[Instances]): length `N` list of `Instances`. The i-th
                `Instances` contains instances for the i-th input image,
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.
            images (ImageList): input image with shape [batch_size, W, H]

        Returns:
            In training, a dict of losses.\\
            In inference, update `instances` with new fields "coordinate3d" and return it.

        """
        features_list = [features[f] for f in self.in_features]
        if self.training:
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            pred_boxes = [i.proposal_boxes for i in proposals]
            # resized_img = nn.functional.interpolate(images,size=self.img_size,mode='bilinear')
            resized_img = images.__getitem__(0)

            _resized_img = resized_img.to("cpu")
            from torchvision.transforms import transforms as ttf
            import numpy as np
            toImg = ttf.ToPILImage()
            # _resized_img = toImg(_resized_img)
            img = _resized_img.permute(1,2,0).numpy()[:,:, ::-1]
            # img += np.asarray(self.pixel_mean)
            dispImg("resized_img", img)

                        
            pred_coordinates = self.coordinate_head(self.features_coord)

            if self.use_backbone_features:
                if len(proposals) > 0:
                    proposal_boxes = [x.proposal_boxes for x in proposals]
                    features_coord = self.coordinate_pooler(features_list, proposal_boxes)

                    features_coord = cat([])

            
            return {}
            # return loss
            pass
        else:
            # return instances
            raise NotImplementedError("Not implemented")

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None
    ):
        losses = {}
        if not self.separate_attrpred_on:
            instances, losses = super().forward(images, features, proposals, targets)

        if self.training:
            losses.update(self._forward_coordinate(features, instances, images))

        del targets, images
