from detectron2.config import get_cfg, CfgNode
from detectron2.data.catalog import MetadataCatalog
from detectron2.data.build import build_detection_test_loader
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.evaluation.evaluator import inference_on_dataset
from detectron2.evaluation import COCOEvaluator

import torch
import cv2
import warnings
import os
from datetime import datetime
from model import CaterROIHeads
from config.cater_config import add_cater_config
from data import register_cater_dataset
from cater_trainer import CaterTrainer

def train_on_server(cfg:CfgNode):
    cfg.DATALOADER.NUM_WORKERS = 16
    cfg.SOLVER.IMS_PER_BATCH = 7
    cfg.SOLVER.BASE_LR = 0.01

USE_FPN_BACKBONE = True
if __name__ == "__main__":
    setup_logger()
    # register dataset
    annotation_location = os.path.join('.', 'dataset', 'annotations','31train.json')
    img_folder = os.path.join('.', 'dataset', 'images','image')
    register_cater_dataset.register_dataset(dataset_name='cater', annotations_location= annotation_location, image_folder= img_folder)
    test_annot_location = os.path.join('.', 'dataset', 'annotations','31test.json')
    test_img_folder = os.path.join('.', 'dataset', 'images','test_image')
    register_cater_dataset.register_dataset(dataset_name='cater_test', annotations_location=test_annot_location, image_folder=test_img_folder)
    # set configuration file
    cfg = get_cfg()
    if USE_FPN_BACKBONE:
        # use R50 FPN 
        cfg.merge_from_file("detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
    else:
        # use R50 C4 instead 
        cfg.merge_from_file("detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml")
    add_cater_config(cfg)
    cfg.merge_from_file("config/Cater.yaml")

    # TODO: add evaluator for cater 
    cfg.DATASETS.TEST = ()
    # cfg.TEST.EVAL_PERIOD = 80 # need to define hooks for evaluation
    cfg.DATALOADER.NUM_WORKERS = 6

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 193
    # Total number of RoIs per training minibatch =
    #   ROI_HEADS.BATCH_SIZE_PER_IMAGE * SOLVER.IMS_PER_BATCH
    # E.g., a common configuration is: 512 * 16 = 8192
    # number of ROI per image
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 50

    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.WARMUP_ITERS = 500
    cfg.SOLVER.BASE_LR = 0.005

    on_server  = False
    if os.path.expanduser('~').split('/')[-1] == 'group1':
        print("use cfg for server")
        on_server = True
        train_on_server(cfg)

    # if input("print configurations? ") == 'y':
    #     print(cfg.dump())
    
    output_dir = datetime.today().strftime('%d-%m_%H_%M')
    cfg.OUTPUT_DIR = "output/{}".format(output_dir)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = CaterTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()