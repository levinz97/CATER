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
        annotations = os.path.join('annotations', '5301-5305.json')
        self.dataloader = CaterDataloader(root, image_dir, annotations)


    def train_net(self, model, device):
        num_epochs = 2
        model.to(device)
        loss_mse = nn.MSELoss()
        loss_mse.to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0001)

        for epoch in range(num_epochs):
            model.train()
            i = 0
            for imgs, annotations in self.dataloader:
                loss_image = 0
                # print("-------------------")
                # print(imgs)
                # print(imgs.shape)
                # print(type(imgs))
                # transform1 = torchvision.transforms.ToPILImage()
                # a = transform1(imgs)
                # a.show()
                # print(annotations)
                # break
                i += 1
                imgs = imgs.to(device)
                bboxes = annotations['boxes'].to(device)
                coordinates = annotations['coordinates'].to(device)
                for j in range(len(bboxes)):
                    output = model(imgs, bboxes[j])
                    loss = loss_mse(output, coordinates[j])
                    loss_image += loss

                    # pred_bbox_d2 = self.detectron(imgs)
                    # loss = model(imgs, annotations, pred_bbox_d2)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                loss_image = loss_image / len(bboxes)
                print(f'Iteration: {i}/{len(self.dataloader)}, Loss:{loss_image}')


class Coordinate_model(nn.Module):
    def __init__(self):
        super(Coordinate_model, self).__init__()
        self.data_transforms = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                # transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406, 0.485, 0.456, 0.406], [0.229, 0.224, 0.225, 0.229, 0.224, 0.225])
            ])
        self.resnet_model = torchvision.models.resnet50(pretrained=True)
        num_ftrs = self.resnet_model.fc.in_features
        self.resnet_model.conv1 = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet_model.fc = nn.Linear(num_ftrs, 3)
        # print(self.resnet_model)

    def forward(self, img, pred_bbox_d2):
        reshape_img = torch.reshape(img, (-1, 3, 240, 320))
        idx = torch.tensor([0])
        idx = idx.to('cuda')
        pred_bbox_d2 = torch.cat((idx, pred_bbox_d2), dim=0)
        pred_bbox_d2 = torch.reshape(pred_bbox_d2, (-1, 5))
        roi = torchvision.ops.roi_align(reshape_img, pred_bbox_d2, output_size=(240, 320), spatial_scale=1.0, sampling_ratio=-1)
        new_img = torch.cat((reshape_img, roi), dim=1)
        output = self.data_transforms(new_img)
        output = self.resnet_model(output)
        return output


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    coordinatenet = CoordinateNet()
    coordinate_model = Coordinate_model()
    coordinatenet.train_net(coordinate_model, device)