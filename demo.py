import warnings
from utils import dispImg

from detectron2.config import get_cfg
from detectron2.data.catalog import MetadataCatalog
from detectron2.data.build import build_detection_test_loader
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.evaluation.evaluator import inference_on_dataset
from detectron2.evaluation import COCOEvaluator

import cv2
import register_cater_dataset
import os


if __name__ == "__main__":
    setup_logger()
    # register dataset
    annotation_location = os.path.join('.', 'dataset', 'annotations','5200-5214.json')
    img_folder = os.path.join('.', 'dataset', 'images','image')
    register_cater_dataset.register_dataset(dataset_name='cater', annotations_location= annotation_location, image_folder= img_folder)
    register_cater_dataset.register_dataset(dataset_name='cater_test')
    # set configuration file
    cfg = get_cfg()
    cfg.merge_from_file("detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

    cfg.DATASETS.TRAIN = ("cater",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 4

    cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"

    cfg.SOLVER.IMS_PER_BATCH = 3
    cfg.SOLVER.MAX_ITER = (1000)
    cfg.SOLVER.BASE_LR = 0.01
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 270
    # number of ROI per image
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 20

    cfg.OUTPUT_DIR = "output"
    if input("print configurations? ") == 'y':
        print(cfg.dump())
    trainer = DefaultTrainer(cfg)
    if input("continue to train? ") == 'y':
        trainer.resume_or_load(resume=False)
        trainer.train()

    if input("continue to evaluate? ") == 'y':
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.DATASETS.TEST = ("cater_test",)
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
        cfg.TEST.EVAL_PERIOD = 50
        model = trainer.build_model(cfg)
        res = trainer.test(cfg, model, evaluators=COCOEvaluator("cater",distributed=False))
        ## equivalent way for evaluation
        # evaluator = COCOEvaluator("cater", cfg, False)
        # val_loader = build_detection_test_loader(cfg, "cater")
        # inference_on_dataset(trainer.model, val_loader, evaluator)

    if input("continue to predict? ") == 'y':
        warnings.filterwarnings('ignore')
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
        predictor = DefaultPredictor(cfg)
        test_img = cv2.imread("test.png")
        test_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR)
        cater_metadata = MetadataCatalog.get("cater")
        cater_metadata.set(evaluator_type="coco")
        test_img_pred = predictor(test_img)
        pred = test_img_pred["instances"].to("cpu")
        print(f"there are totally {len(pred)} instances detected")
        for item in range(len(pred)):
            vis = Visualizer(test_img[:, :, ::-1],
                        metadata=cater_metadata,
                        instance_mode=ColorMode.IMAGE_BW)
            v = vis.draw_instance_predictions(pred[item])
            img = v.get_image()[:, :, ::-1]
            dispImg("prediction", img, kill_window=True)
            img = []