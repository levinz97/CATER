import os
from detectron2.data.datasets.coco import load_coco_json
from detectron2.data.catalog import DatasetCatalog, MetadataCatalog


def register_dataset(dataset_name: str = 'cater', annotations_location: str = "./raw_data/cocotest/annotations/instances_default.json", image_folder: str = "./raw_data/cocotest/images", extra_annotation_keys=["attributes"], metadata: dict = {}):
    """
    Register a dataset in COCO's json annotation format for
    instance detection, instance segmentation and keypoint detection.
    (i.e., Type 1 and 2 in http://cocodataset.org/#format-data.
    `instances*.json` and `person_keypoints*.json` in the dataset).

    Args:
        dataset_name (str): the name that identifies a dataset, e.g. "coco_2014_train".
        annotaions_location (str): path to the json instance annotation file.
        image_folder (str or path-like): directory which contains all the images.
        metadata (dict): extra metadata associated with this dataset.  You can leave it as an empty dict.
    """
    assert isinstance(dataset_name, str), dataset_name
    if os.path.isfile(annotations_location) and os.path.isdir(image_folder):
        # register_coco_instances(dataset_name, metadata, annotations_location, image_folder)
        DatasetCatalog.register(dataset_name, lambda: load_coco_json(
            annotations_location, image_folder, dataset_name, extra_annotation_keys))
        MetadataCatalog.get(dataset_name).set(
            json_file=annotations_location, image_root=image_folder, evaluator_type='coco', **metadata
        )
    else:
        raise TypeError(f"{image_folder} is not a folder" if not os.path.isdir(
            image_folder) else f"{annotations_location} is not a file")


def load_json():
    return load_coco_json(json_file="./raw_data/cocotest/annotations/instances_default.json",
                          image_root="./raw_data/cocotest/images", dataset_name="cater", extra_annotation_keys=["attributes"])


if __name__ == "__main__":
    print(load_json())
