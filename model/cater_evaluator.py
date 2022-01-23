import copy
from detectron2.structures.boxes import BoxMode
from detectron2.evaluation.coco_evaluation import instances_to_coco_json
from detectron2.evaluation import COCOEvaluator
from model.coordinate_head import coordinate_loss
import numpy as np

import torch
from torch import cat
import torchvision.ops.boxes as bops

class CaterEvaluator(COCOEvaluator):
    def __init__(self,dataset_name, task=None):
        super().__init__(dataset_name, task)
        self.coordinate_errors = []

    def process(self, inputs, outputs):
        for input,output in zip(inputs, outputs):
            prediction = {"image_id": input['image_id']}
            if 'instances' in output:
                instances = output['instances'].to(self._cpu_device)
                prediction['instances'] = instances_to_coco_json(instances, input['image_id'])
                if 'pred_coordinates' in instances._fields:
                    ## get bbox from annotations
                    anno_ids = self._coco_api.getAnnIds(imgIds=input['image_id'])
                    coco_annos = self._coco_api.loadAnns(anno_ids)
                    gt_boxes = torch.as_tensor([
                        BoxMode.convert(obj["bbox"], BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
                        for obj in coco_annos
                        if obj["iscrowd"] == 0
                    ])
                    ## calculate IOU of gt_boxes and pred_boxes
                    pred_boxes = instances.pred_boxes.tensor
                    iou = bops.box_iou(pred_boxes, gt_boxes)
                    gt_idxs = torch.argmax(iou, dim=1)
                    ## get gt_coordinates for obj from obj with max IOU
                    get_coordinates3d = lambda item: torch.tensor([item['attributes'][f'coordination_{axis}'] for axis in ['X', 'Y', 'Z']])
                    gt_coordinates = [get_coordinates3d(coco_annos[int(id)]) for id in gt_idxs]
                    gt_coordinates = torch.stack(gt_coordinates)
                    pred_coordinate = instances.pred_coordinates
                    diff = pred_coordinate - gt_coordinates
                    self.coordinate_errors.append(np.square(diff))
            if "proposals" in output:
                prediction["proposals"] = output["proposals"].to(self._cpu_device)
            if len(prediction) > 1:
                self._predictions.append(prediction)


    def evaluate(self, img_ids=None):
        super().evaluate(img_ids)
        if len(self.coordinate_errors) > 0:
            self._results['coordinate_error'] = np.mean(cat(self.coordinate_errors).numpy())
        return copy.deepcopy(self._results)

