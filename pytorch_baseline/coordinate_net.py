import os

import torchvision
from torch import nn
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from dataloader import CaterDataloader

class CoordinateNet():
    def __init__(self):
        cfg = get_cfg()
        cfg.merge_from_file("../detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        cfg.OUTPUT_DIR = os.path.join("../output", "best")
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
        assert os.path.isfile(cfg.MODEL.WEIGHTS), f'{cfg.MODEL.WEIGHTS} is not a file'
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 270
        self.detectron = DefaultPredictor(cfg)

        root = os.path.join('..', 'dataset') 
        if not os.path.exists(root):
            root = os.path.join('.', 'dataset')

        image_dir = os.path.join('images', 'image')
        annotations = os.path.join('annotations', '5301-5305.json')
        self.dataloader = CaterDataloader(root, image_dir, annotations)
        self.writer = SummaryWriter("coordinate_loss")

    def train_net(self, model, device):
        num_epochs = 50
        model.to(device)
        loss_mse = nn.MSELoss()
        loss_mse.to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0001)

        for epoch in range(num_epochs):
            model.train()
            i = 0
            for imgs, annotations in self.dataloader:
                loss_image = 0
                i += 1
                imgs = imgs.to(device)
                coordinates = annotations['coordinates'].to(device)
                output = model(imgs)
                loss = loss_mse(output, coordinates)
                # pred_bbox_d2 = self.detectron(imgs)
                # loss = model(imgs, annotations, pred_bbox_d2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print(f'Iteration: {i}/{len(self.dataloader)}, Loss:{loss}')


class Coordinate_model(nn.Module):
    def __init__(self):
        super(Coordinate_model, self).__init__()
        self.resnet_model = torchvision.models.resnet50(pretrained=True)
        num_ftrs = self.resnet_model.fc.in_features
        self.resnet_model.conv1 = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet_model.fc = nn.Linear(num_ftrs, 3)
        # print(self.resnet_model)

    def forward(self, img):
        output = self.resnet_model(img)
        # print(output.shape)
        return output

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    coordinatenet = CoordinateNet()
    coordinate_model = Coordinate_model()
    coordinatenet.train_net(coordinate_model, device)