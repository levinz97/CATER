from detectron2.data.datasets import register_coco_instances
from detectron2.data.datasets.coco import load_coco_json

def register_dataset():
    register_coco_instances("cater",{}, "./raw_data/cocotest/annotations/instances_default.json", "./raw_data/cocotest/images")

def load_json():
    return load_coco_json(json_file="./raw_data/cocotest/annotations/instances_default.json",
                        image_root="./raw_data/cocotest/images", dataset_name="cater", extra_annotation_keys=["attributes"])


if __name__ == "__main__":
    print(load_json())
