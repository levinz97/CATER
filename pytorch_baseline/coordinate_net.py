import os

import torchvision
from torch import nn
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
import torch
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
        annotations = os.path.join('annotations', '5200-5214.json')
        self.dataloader = CaterDataloader(root, image_dir, annotations)


    def train_net(self, model, device):
        num_epochs = 150
        model.to(device)
        loss_mse = nn.MSELoss()
        loss_mse.to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0001)

        for epoch in range(num_epochs):
            model.train()
            i = 0
            for imgs, annotations in self.dataloader:
                print(imgs.shape)
                i+=1
                imgs = list(img.to(device) for img in imgs)
                bboxes = list({k: v.to(device)} for k, v in annotations if k =='boxes')
                coordinates = list({k: v.to(device)} for k, v in annotations if k =='coordinates')
                for j in range(len(bboxes)):
                    output = model(imgs, bboxes[j])
                    loss = loss_mse(output, coordinates[j])

                    # pred_bbox_d2 = self.detectron(imgs)
                    # loss = model(imgs, annotations, pred_bbox_d2)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            print(f'Iteration: {i}/{len(self.dataloader)}, Loss:{loss}')


class Coordinate_model(nn.Module):
    def __init__(self):
        super(Coordinate_model, self).__init__()
        self.data_transforms = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def forward(self, img, pred_bbox_d2): #
        idx = torch.tensor([0])
        pred_bbox_d2 = torch.cat((idx, pred_bbox_d2), dim=0)
        pred_bbox_d2 = torch.reshape(pred_bbox_d2, (-1, 5))
        roi = torchvision.ops.roi_align(img, pred_bbox_d2, output_size=(240, 320), spatial_scale=1.0, sampling_ratio=-1)
        new_img = torch.cat((img, roi), dim=1)
        output = self.data_transforms(new_img)
        output = resnet_model(output)
        return output


if __name__ == '__main__':
    resnet_model = torchvision.models.resnet50(pretrained=True)
    num_ftrs = resnet_model.fc.in_features
    resnet_model.fc = nn.Linear(num_ftrs, 3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    coordinatenet = CoordinateNet()
    coordinatenet.train_net(resnet_model, device)