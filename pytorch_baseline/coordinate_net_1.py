import os

import torchvision
from torch import nn
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
import torch
from torch.utils.data import DataLoader
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
        train_annotations = os.path.join('annotations', 'train_dataset.json')
        test_annotations = os.path.join('annotations', 'test_dataset.json')
        train_batch_size = 6
        test_batch_size = 1
        self.train_dataset = CaterDataloader(root, image_dir, train_annotations)
        self.train_dataloader = DataLoader(dataset=self.train_dataset, batch_size=train_batch_size, collate_fn=collate)
        self.test_dataset = CaterDataloader(root, image_dir, test_annotations)
        self.test_dataloader = DataLoader(dataset=self.test_dataset, batch_size=test_batch_size, collate_fn=collate)
        self.writer = SummaryWriter("coordinate_loss_{}".format(train_batch_size))

    def train_net(self, model, device):
        num_epochs = 20
        model.to(device)
        loss_mse = nn.MSELoss()
        loss_mse.to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0001)

        total_train_step = 0
        for epoch in range(num_epochs):
            print("------------------{} epoch start -------------------".format(epoch+1))
            model.train()
            for data in self.train_dataloader:
                imgs, coordinates = data
                # print(imgs.shape)
                # print(coordinates.shape)
                imgs = imgs.to(device)
                coordinates = coordinates.to(device)
                # print(coordinates.shape)
                output = model(imgs)
                loss = loss_mse(output, coordinates)
                # pred_bbox_d2 = self.detectron(imgs)
                # loss = model(imgs, annotations, pred_bbox_d2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_train_step += 1
                if total_train_step % 50 == 0:
                    print("number of training：{}, Loss:{}".format(total_train_step, loss.item()))
                    self.writer.add_scalar("train_loss", loss.item(), total_train_step)
            torch.save(model, "./model_output/coord_model_{}.pth".format(epoch))

    def test(self, model, device):
        model.to(device)
        loss_mse = nn.MSELoss()
        loss_mse.to(device)
        model.eval()
        total_test_loss = 0
        step = 0
        with torch.no_grad():
            for data in self.test_dataloader:
                imgs, coordinates = data
                # print(imgs.shape)
                # print(coordinates.shape)
                imgs = imgs.to(device)
                coordinates = coordinates.to(device)
                # print(coordinates.shape)
                output = model(imgs)
                loss = loss_mse(output, coordinates)

                # loss = model(imgs, annotations, pred_bbox_d2)
                total_test_loss += loss
                step += 1
                if step % 5 == 0:
                    print(step)
            print(total_test_loss.item())
            average_loss = total_test_loss / step
            print(average_loss.item())

    def predict(self, imgs):
        pred_bbox_d2 = self.detectron(imgs)
        

def collate(batch):
    list1 = []
    list2 = []
    for data in batch:
        list1.append(data[0])
        list2.append(data[1])
        imgs = torch.cat(list1, dim=0)
        labels = torch.cat(list2, dim=0)
    return imgs, labels

class Coordinate_model(nn.Module):
    def __init__(self):
        super(Coordinate_model, self).__init__()
        self.resnet_model = torchvision.models.resnet34(pretrained=True)
        self.resnet_model.conv1 = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_ftrs = self.resnet_model.fc.in_features
        self.resnet_model.fc = nn.Linear(num_ftrs, 3)
        # print(self.resnet_model)

    def forward(self, img):
        output = self.resnet_model(img)
        # print(output.shape)
        return output


if __name__ == '__main__':
    train = False # if train or not
    test = False # if test or not
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    coordinatenet = CoordinateNet()
    coordinate_model = Coordinate_model()
    print(coordinate_model)
    if train:
        coordinatenet.train_net(coordinate_model, device)
    if test:
        test_model = torch.load("./trained_model/coord_model_bz7_19.pth")  # load the trained model
        coordinatenet.test(test_model, device)