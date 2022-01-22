import copy
from detectron2.evaluation.coco_evaluation import instances_to_coco_json
from detectron2.evaluation import COCOEvaluator
from model.coordinate_head import coordinate_loss
import numpy as np

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
                    gt_coordinate = 0
                    pred_coordinate = instances.pred_coordinates
                    diff = pred_coordinate
                    self.coordinate_errors.append(np.square(diff))
            if "proposals" in output:
                prediction["proposals"] = output["proposals"].to(self._cpu_device)
            if len(prediction) > 1:
                self._predictions.append(prediction)


    def evaluate(self, img_ids=None):
        super().evaluate(img_ids)
        if len(self.coordinate_errors) > 0:
            self._results['coordinate_error'] = np.mean(self.coordinate_errors)
        return copy.deepcopy(self._results)

