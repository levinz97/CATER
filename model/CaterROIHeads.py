from math import floor
from typing import Dict, List, Optional
from detectron2.layers import ShapeSpec
from detectron2.layers.wrappers import cat
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads import select_foreground_proposals

from detectron2.modeling import StandardROIHeads, ROI_HEADS_REGISTRY

from detectron2.structures import ImageList, Instances
import torch
from torch import nn

from.layers import Decoder
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
        self.img_per_batch         = cfg.SOLVER.IMS_PER_BATCH
        num_total_boxes            = self.batch_size_per_image
        self.num_fg_boxes          = floor(num_total_boxes * self.positive_fraction)
        self.use_backbone_features = cfg.MODEL.ROI_COORDINATE_HEAD.USE_BACKBONE_FEATURES
        coordinate_in_features     = cfg.MODEL.ROI_COORDINATE_HEAD.IN_FEATURES if self.use_backbone_features else None
        self.img_size              = cfg.MODEL.ROI_COORDINATE_HEAD.IMG_SIZE
        self.hide_img_size         = cfg.MODEL.ROI_COORDINATE_HEAD.HIDE_IMG_SIZE
        in_channels = 6 # raw image 3 + cropped image within bbox 3
        img_coordinate_pooler_resolution     = cfg.MODEL.ROI_COORDINATE_HEAD.HIDE_IMG_SIZE
        img_coordinate_pooler_type           = cfg.MODEL.ROI_COORDINATE_HEAD.POOLER_TYPE
        img_coordinate_pooler_sampling_ratio = 0
        img_coordinate_pooler_scale          = [1] # on raw image
        
        if self.use_backbone_features:
            bb_coordinate_pooler_resolution     = cfg.MODEL.ROI_COORDINATE_HEAD.POOLER_RESOLUTION
            bb_coordinate_pooler_sampling_ratio = cfg.MODEL.ROI_COORDINATE_HEAD.POOLER_SAMPLING_RATIO
            bb_coordinate_pooler_type           = cfg.MODEL.ROI_COORDINATE_HEAD.POOLER_TYPE
            bb_coordinate_pooler_scale = [1.0 / input_shape[k].stride for k in coordinate_in_features]
            decoder_in_channels = [input_shape[f].channels for f in coordinate_in_features][0]
            channel_decrease_ratio = 6 # 2^ratio
            decoder_out_channels = decoder_in_channels // (2**channel_decrease_ratio)
            in_channels += decoder_out_channels
            # decoder to decrease the number of channels from backbones' features, and upsample it 
            self.bb_decoder = Decoder(decoder_in_channels, channel_decrease_ratio, use_upsample=True)
            self.bb_coordinate_pooler = ROIPooler(
                output_size=bb_coordinate_pooler_resolution,
                scales=bb_coordinate_pooler_scale,
                sampling_ratio=bb_coordinate_pooler_sampling_ratio,
                pooler_type=bb_coordinate_pooler_type
            )

        self.img_coordinate_pooler = ROIPooler(
                output_size=img_coordinate_pooler_resolution,
                scales=img_coordinate_pooler_scale,
                sampling_ratio=img_coordinate_pooler_sampling_ratio,
                pooler_type=img_coordinate_pooler_type
            )
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
        need_visualization = False
        image_list = [images.tensor]
        if self.use_backbone_features:
            features_list = [features[f] for f in self.in_features]
        else:
            # dont use backbone features
            del features

        if self.training:
            fg_proposals, _ = select_foreground_proposals(instances, self.num_classes)
            pred_boxes = [i.proposal_boxes for i in fg_proposals]
            coordinate_features = self.img_coordinate_pooler(image_list, pred_boxes)
            resized_img = nn.functional.interpolate(images.tensor, size=self.hide_img_size, mode='bilinear', align_corners=False)
            if need_visualization:
                features_to_vis = coordinate_features[0, :, :, :]
                img_to_vis = resized_img[0, :, :, :]
            # use additional features from backbone
            if self.use_backbone_features:
                    bb_features_coord = self.bb_coordinate_pooler(features_list, pred_boxes)
                    bb_features_coord = self.bb_decoder(bb_features_coord)
                    coordinate_features = torch.cat((coordinate_features, bb_features_coord), dim=1)
            assert self.num_fg_boxes * self.img_per_batch == coordinate_features.shape[0], coordinate_features.shape
            # first reshape coordinate_features to [img_per_batch, num_fg_boxes, *HIDE_IMG_SIZE]
            coordinate_features = coordinate_features.unsqueeze_(0).view(self.img_per_batch, self.num_fg_boxes, -1, *self.hide_img_size)
            # then expand the resized_img to [img_per_batch, num_fg_boxes, *HIDE_IMG_SIZE]
            resized_img = resized_img.unsqueeze_(1).expand(-1, self.num_fg_boxes, -1,*self.hide_img_size)
            coordinate_features = torch.cat((coordinate_features, resized_img), dim=2)
            coordinate_features = coordinate_features.view(self.num_fg_boxes*self.img_per_batch, -1, *self.hide_img_size)
            
            def vis_tensor(tensor_to_vis: torch.Tensor, kill_window):
                tensor_to_vis = tensor_to_vis.to("cpu")
                from torchvision.transforms import transforms as ttf
                import numpy as np
                toImg = ttf.ToPILImage()
                # _resized_img = toImg(_resized_img)
                img = tensor_to_vis.permute(1,2,0).numpy()[:,:, ::-1]
                img += np.asarray(self.pixel_mean)
                img = np.array(img, dtype=np.int32)
                dispImg("resized_img", img, kill_window)
            if need_visualization:
                # vis_tensor(images.__getitem__(0), False)
                vis_tensor(img_to_vis, False)
                vis_tensor(features_to_vis, True)
            

            pred_coordinates = self.coordinate_head(coordinate_features)
            loss_coord = coordinate_loss(pred_coordinates, fg_proposals)
            
            return loss_coord
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
        del targets

        if self.training:
            losses.update(self._forward_coordinate(features, instances, images))

        del images

        return instances, losses
