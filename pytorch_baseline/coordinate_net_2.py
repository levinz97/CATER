import os

import torchvision
from torch import nn
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from dataloader_2 import CaterDataloader

class CoordinateNet():
    def __init__(self):
        root = os.path.join('..', 'dataset') 
        if not os.path.exists(root):
            root = os.path.join('.', 'dataset')

        image_dir = os.path.join('images', 'image')
        train_annotations = os.path.join('annotations', 'train_dataset.json')
        test_annotations = os.path.join('annotations', 'test_dataset.json')
        self.train_batch_size = 2
        self.test_batch_size = 1
        self.train_dataset = CaterDataloader(root, image_dir, train_annotations, train=False)
        self.train_dataloader = DataLoader(dataset=self.train_dataset, batch_size=self.train_batch_size, shuffle=True ,collate_fn=collate)
        self.test_dataset = CaterDataloader(root, image_dir, test_annotations, train=False)
        self.test_dataloader = DataLoader(dataset=self.test_dataset, batch_size=self.test_batch_size, collate_fn=collate)
        self.writer = SummaryWriter("resnet_p_{}".format(self.train_batch_size))

    def train_net(self, model, device):
        num_epochs = 20
        model.to(device)
        loss_mse = nn.MSELoss()
        loss_mse.to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0001)
        # scheduler = StepLR(optimizer, step_size=5, gamma=0.2)

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
                if total_train_step % 10 == 0:
                    print("number of training：{}, Loss:{}".format(total_train_step, loss.item()))
                    self.writer.add_scalar("train_loss", loss.item(), total_train_step)
            torch.save(model, "./model_output/resnext_drop_{}.pth".format(epoch))
            # scheduler.step()

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
                # if step % 5 == 0:
                #     print(step)
            print(total_test_loss.item())
            average_loss = total_test_loss / step
            print(average_loss.item())

    # def predict(self, imgs):
    #     pred_bbox_d2 = self.detectron(imgs)

def collate(batch):
    list1 = []
    list2 = []
    for data in batch:
        list1.append(data[0])
        list2.append(data[1])
        imgs = torch.cat(list1, dim=0)
        labels = torch.cat(list2, dim=0)
    return imgs, labels

# for epoch in range(60):
#     lr = 30e-5
#     if epoch > 25:
#         lr = 15e-5
#     if epoch > 30:
#         lr = 7.5e-5
#     if epoch > 35:
#         lr = 3e-5
#     if epoch > 40:
#         lr = 1e-5
#     adjust_learning_rate(optimizer, lr)

class Coordinate_model(nn.Module):
    def __init__(self):
        super(Coordinate_model, self).__init__()
        self.model = torchvision.models.resnet50(pretrained=True)
        self.model.conv1 = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # self.backbone = torch.nn.Sequential(*list(model.children())[:-3])
        # self.avgpool = model.avgpool
        # self.flatten = nn.Flatten()
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_features=num_ftrs, out_features=256, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(in_features=256, out_features=3, bias=True),
        )

    def forward(self, img):
        # output = self.backbone(img)
        # output = self.avgpool(output)
        # output = self.flatten(output)
        # output = self.fc(output)
        output = self.model(img)
        return output


if __name__ == '__main__':
    train = False# if train or not
    test = True # if test or not
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    coordinatenet = CoordinateNet()
    coordinate_model = Coordinate_model()
    # print(coordinate_model)
    if train:
        coordinatenet.train_net(coordinate_model, device)
    if test:
        test_model = torch.load("./trained_model/resnet_drop_bz8_18.pth")  # load the trained model
        coordinatenet.test(test_model, device)