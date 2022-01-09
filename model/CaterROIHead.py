from typing import Dict, List, Optional
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads import StandardROIHeads, ROI_HEADS_REGISTRY

from detectron2.structures import ImageList, Instances
import torch

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

        
    def _init_colormaterial_head(self, cfg, input_shape):
        pass
    
    def _init_shape_head(self, cfg, input_shape):
        pass

    def _init_size_head(self, cfg, input_shape):
        pass

    def _init_coordinate_head(self, cfg, input_shape):
        coordinate_pooler_resolution     = cfg.MODEL.ROI_COORDINATE_HEAD.POOLER_RESOLUTION
        coordinate_pooler_sampling_ratio = cfg.MODEL.ROI_COORDINATE_HEAD.POOLER_SAMPLING_RATIO
        coordinate_pooler_type           = cfg.MODEL.ROI_COORDINATE_HEAD.POOLER_TYPE
        in_features = cfg.ROI_HEADS.INFEATURES
        coordinate_pooler_scale  = tuple(1.0 / input_shape[k].stride for k in in_features)
        self.coordinate_pooler = ROIPooler(
            output_size=coordinate_pooler_resolution,
            scales=coordinate_pooler_scale,
            sampling_ratio=coordinate_pooler_sampling_ratio,
            pooler_type=coordinate_pooler_type
        )

    
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

        if self.training:
            
            # return loss
            pass
        else:
            # return instances
            pass

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None
    ):
        if not self.separate_attrpred_on:
            instances, losses = super().forward(images, features, proposals, targets)


        if self.training:
            losses.update(self._forward_coordinate(features, instances, images))

        del targets, images
