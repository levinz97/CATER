import copy
from detectron2.data import DatasetMapper
import detectron2.data.detection_utils as detection_utils
import numpy as np
import torch

class CaterDatasetMapper(DatasetMapper):
    
    def _transform_annotations(self, dataset_dict, transforms, image_shape):
        """
        overwrite the _transform_annotations in DatasetMapper to store extra data for training
        """
        for anno in dataset_dict['annotations']:
            if not self.use_instance_mask:
                anno.pop("segmentation", None)
            if not self.use_keypoint:
                anno.pop("keypoint", None)
        
        annos = [
            detection_utils.transform_instance_annotations(
                obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
            )
            for obj in dataset_dict['annotations']
            if obj.get("iscrowd", 0) == 0
        ]
        instances = detection_utils.annotations_to_instances(
            annos, image_shape, mask_format=self.instance_mask_format
        )
        instances.gt_coordinate = torch.Tensor(
            [[x['attributes']['coordination_X'], x['attributes']['coordination_Y'],x['attributes']['coordination_Z']] for x in dataset_dict['annotations']]
        )
        # TODO: convert to Tensor for training
        # instances.gt_color    = np.array([x['attributes']['color'] for x in dataset_dict['annotations']])
        # instances.gt_material = np.array([x['attributes']['material'] for x in dataset_dict['annotations']])
        # instances.gt_size     = np.array([x['attributes']['size'] for x in dataset_dict['annotations']])
        # instances.gt_shape    = np.array([x['attributes']['shape'] for x in dataset_dict['annotations']])

        
        dataset_dict.pop('annotations')

        dataset_dict['instances'] = detection_utils.filter_empty_instances(instances)
        

        

