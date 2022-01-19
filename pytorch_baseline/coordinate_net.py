import os
from torch import nn
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
import torch
from .dataloader import CaterDataloader

class CoordinateNet():
    def __init__(self):
        cfg = get_cfg()
        cfg.merge_from_file("detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        cfg.OUTPUT_DIR = os.path.join("output", "best")
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
        assert os.path.isfile(cfg.MODEL.WEIGHTS), f'{cfg.MODEL.WEIGHTS} is not a file'
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 270
        self.detectron = DefaultPredictor(cfg)

        root = os.path.join('..', 'dataset') 
        if not os.path.exists(root):
            root = os.path.join('.', 'dataset')

        image_dir = os.path.join('images', 'image')
        annotations = os.path.join('annotations', '5200-5214.json')
        dataloader = CaterDataloader(root, image_dir, annotations)


    def train_net(self, model:nn.Model, device, dataloader, optimizer):
        num_epochs = 150
        model.to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0001)

        for epoch in range(num_epochs):
            model.train()
            i = 0
            for imgs, annotations in dataloader:
                i+=1
                imgs = list(img.to(device) for img in imgs)
                annotations = list({k: v.to(device)} for k, v in annotations if k =='coordinates')
                pred_bbox_d2 = self.detectron(imgs)
                loss = model(imgs, annotations, pred_bbox_d2)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print(f'Iteration: {i}/{len(dataloader)}, Loss:{loss}')



if __name__ == '__main__':
    pass
