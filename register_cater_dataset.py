from shutil import Error
from detectron2.data.datasets import register_coco_instances
from detectron2.data.datasets.coco import load_coco_json
import os


def register_dataset(dataset_name: str = 'cater', annotations_location: str = "./raw_data/cocotest/annotations/instances_default.json", image_folder: str = "./raw_data/cocotest/images"):
    if os.path.isfile(annotations_location) and os.path.isdir(image_folder):
        register_coco_instances(dataset_name, {}, annotations_location, image_folder)
    else:
        raise TypeError(f"{image_folder} is not a folder" if not os.path.isdir(image_folder) else f"{annotations_location} is not a file")


def load_json():
    return load_coco_json(json_file="./raw_data/cocotest/annotations/instances_default.json",
                          image_root="./raw_data/cocotest/images", dataset_name="cater", extra_annotation_keys=["attributes"])


if __name__ == "__main__":
    print(load_json())
