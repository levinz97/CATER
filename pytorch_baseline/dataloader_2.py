import cv2
import detectron2
import torch.utils.data as data
import torch
import torchvision
from PIL import Image
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from torchvision import transforms
from pycocotools.coco import COCO

import os


class CaterDataloader(data.Dataset):
    def __init__(self, root, image, annotations, transforms=None, train=True):
        cfg = get_cfg()
        cfg.merge_from_file("../detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        cfg.OUTPUT_DIR = os.path.join("../output", "best")
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
        assert os.path.isfile(cfg.MODEL.WEIGHTS), f'{cfg.MODEL.WEIGHTS} is not a file'
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 270
        self.detectron = DefaultPredictor(cfg)

        image = os.path.join(root, image)
        annotations = os.path.join(root,annotations)
        assert os.path.isdir(image), image
        assert os.path.isfile(annotations), annotations
        self.image_dir = image
        self.coco = COCO(annotations)
        self.transforms = transforms
        self.train = train
        self.ids = list(sorted(self.coco.imgs.keys()))  # 数据集中所有样本的id号


    def __getitem__(self, index):  # access through index  s = CaterDaraloader s[1] ...
        img_id = self.ids[index]  # get img_id
        anno_ids = self.coco.getAnnIds(imgIds=img_id) # get annonation id
        coco_annos = self.coco.loadAnns(anno_ids) # loading annotation
        img_path = self.coco.loadImgs(img_id)[0]['file_name'] #loading img
        img = Image.open(os.path.join(self.image_dir, img_path))
        img = transforms.ToTensor()(img)

        # self.coco.showAnns(coco_annos,draw_bbox=True)
        num_objs = len(coco_annos)
        # Bounding boxes for objects
        boxes = []
        # Tensorise img_id
        img_id = torch.tensor([img_id])
        # Size of bbox (Rectangular)
        areas = []
        coordinates = []
        categories = []
        for item in coco_annos:
            # In coco format, bbox = [xmin, ymin, width, height]
            # In pytorch, the input should be [xmin, ymin, xmax, ymax]
            xmin = item['bbox'][0]
            ymin = item['bbox'][1]
            xmax = xmin + item['bbox'][2]
            ymax = ymin + item['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])

            areas.append(item['area'])
            coordinate3d = []
            for axis in ['X', 'Y', 'Z']:
                coordinate3d.append(item['attributes'][f'coordination_{axis}'])
            coordinates.append(coordinate3d) 
            categories.append(item['category_id'])

        categories = torch.as_tensor(categories, dtype=torch.int32) # change the variable type to torch.tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        coordinates = torch.as_tensor(coordinates, dtype=torch.float32)
        iscrowd = torch.zeros((num_objs,), dtype=torch.int32)

        # Annotation is in dictionary format
        cater_annotation = {}
        cater_annotation["boxes"] = boxes
        cater_annotation["category_ids"] = categories
        cater_annotation["image_id"] = img_id
        cater_annotation["area"] = areas
        cater_annotation["iscrowd"] = iscrowd
        cater_annotation["coordinates"] = coordinates
        # print(boxes)

        if self.train:
            #use the boxes from mask rcnn prediction
            true_box = detectron2.structures.Boxes(cater_annotation["boxes"]) # shape
            # print(true_box)
            img_cv =cv2.imread(os.path.join(self.image_dir, img_path))
            output = self.detectron(img_cv)
            pred_bbox_d2 = output['instances'].pred_boxes.to('cpu')
            # print(pred_bbox_d2)
            iou = detectron2.structures.pairwise_iou(pred_bbox_d2, true_box)
            # print(iou)
            order = torch.argmax(iou, dim=1)
            size = order.size(dim=0)
            coordinates_list = []
            for i in range(size):
                number = order[i].item()
                coordinates_list.append(cater_annotation["coordinates"][number])
            coordinates_list = torch.stack(coordinates_list)
            cater_annotation["coordinates"] = coordinates_list
            cater_annotation["boxes"] = output['instances'].pred_boxes.tensor.to('cpu').round()
            # print(cater_annotation["boxes"])
            # print(cater_annotation["coordinates"])

        if self.transforms is not None:
            img = self.transforms(img)

        data_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        reshape_img = torch.reshape(img, (-1, 3, 240, 320))
        idx = torch.tensor([0])
        step = 0
        for bbox in cater_annotation["boxes"]:
            bbox = torch.cat((idx, bbox), dim=0)
            bbox = torch.reshape(bbox, (-1, 5))
            roi = torchvision.ops.roi_align(reshape_img, bbox, output_size=(224, 224), spatial_scale=1.0,
                                            sampling_ratio=-1)
            roi = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(roi)
            transform_img = data_transforms(reshape_img)
            new_img = torch.cat((transform_img, roi), dim=1)
            if step == 0:
                cat_img = new_img
            else:
                cat_img = torch.cat((cat_img, new_img), dim=0)
            step += 1
        img = cat_img

        return img, cater_annotation["coordinates"]

    def __len__(self):
        return len(self.ids)


if __name__ == '__main__':
    root = os.path.join('..', 'dataset') 
    if not os.path.exists(root):
        root = os.path.join('.', 'dataset')
    
    image_dir = os.path.join('images', 'image')
    annotations = os.path.join('annotations', 'train_dataset.json')
    cdl = CaterDataloader(root, image_dir, annotations, train=True)
    cdl.__getitem__(0)

