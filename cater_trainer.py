from detectron2.data.build import build_detection_test_loader, build_detection_train_loader
from detectron2.engine import DefaultTrainer
from detectron2.config import CfgNode

from data.cater_dataset_loader import CaterDatasetMapper

class CaterTrainer(DefaultTrainer):

    @classmethod
    def build_test_loader(cls, cfg:CfgNode):
        return build_detection_test_loader(cfg, mapper=CaterDatasetMapper(cfg, is_train=False))
    
    @classmethod
    def build_train_loader(cls, cfg:CfgNode):
        return build_detection_train_loader(cfg, mapper=CaterDatasetMapper(cfg, is_train=True))