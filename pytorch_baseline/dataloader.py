import numpy as np
import torch.utils.data as data
import torch
from PIL import Image
from torchvision import transforms
from pycocotools.coco import COCO

import os

class CaterDataloader(data.Dataset):
    def __init__(self, root, image, annotations, transforms=None):
        image = os.path.join(root, image)
        annotations = os.path.join(root,annotations)
        assert os.path.isdir(image), image
        assert os.path.isfile(annotations), annotations
        self.image_dir = image
        self.root = root
        self.coco = COCO(annotations)
        self.transforms = transforms
        self.ids = list(sorted(self.coco.imgs.keys()))


    def __getitem__(self, index):
        img_id = self.ids[index]
        anno_ids = self.coco.getAnnIds(imgIds=img_id)
        coco_annos = self.coco.loadAnns(anno_ids)
        path = self.coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, self.image_dir, path))
        # self.coco.showAnns(coco_annos,draw_bbox=True)
        num_objs = len(coco_annos)
        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        for i in range(num_objs):
            xmin = coco_annos[i]['bbox'][0]
            ymin = coco_annos[i]['bbox'][1]
            xmax = xmin + coco_annos[i]['bbox'][2]
            ymax = ymin + coco_annos[i]['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        # Tensorise img_id
        img_id = torch.tensor([img_id])
        # Size of bbox (Rectangular)
        areas = []
        coordinates = []
        for i in range(num_objs):
            areas.append(coco_annos[i]['area'])
            coordinate3d = []
            for axis in zip('X', 'Y', 'Z'):
                coordinate3d.append(coco_annos[i]['attributes'][f'coordination_{axis}'])
            coordinates.append(coordinate3d) 
        areas = torch.as_tensor(areas, dtype=torch.float32)
        coordinates = torch.as_tensor(coordinates, dtype=torch.float32)
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Annotation is in dictionary format
        cater_annotation = {}
        cater_annotation["boxes"] = boxes
        cater_annotation["labels"] = labels
        cater_annotation["image_id"] = img_id
        cater_annotation["area"] = areas
        cater_annotation["iscrowd"] = iscrowd
        cater_annotation["coordinates"] = coordinates


        if self.transforms is not None:
            img = self.transforms(img)


        return img, cater_annotation

    def __len__(self):
        return len(self.ids)


if __name__ == '__main__':
    root = os.path.join('..', 'dataset') 
    if not os.path.exists(root):
        root = os.path.join('.', 'dataset')
    
    image_dir = os.path.join('images', 'image')
    annotations = os.path.join('annotations', '5200-5214.json')
    cdl = CaterDataloader(root, image_dir, annotations)
    cdl.__getitem__(0)

