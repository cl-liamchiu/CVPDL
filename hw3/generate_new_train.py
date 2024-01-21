import json
import numpy as np
import argparse
import os

def get_args_parser():
    parser = argparse.ArgumentParser('Generate_new_annotation', add_help=False)
    parser.add_argument('--dataset_dir', default='',
                    help='train.json path where to load')
    return parser


def bbox2coco(bbox, width, height):
    x_min = int(bbox[0]*width)
    y_min = int(bbox[1]*height)
    bbox_width = int((bbox[2]-bbox[0])*width)
    bbox_height = int((bbox[3]-bbox[1])*height)
    return [x_min, y_min, bbox_width, bbox_height]

def main(args):
    with open('images_with_caption.json') as f:
        images_with_caption_json = json.load(f)

    with open(os.path.join(args.dataset_dir, 'annotations', 'train.json')) as f:
        train_json = json.load(f)

    categroy_name2id = {}
    for category in train_json["categories"]:
        categroy_name2id[category["name"]] = category["id"]

    last_image_id = train_json["images"][-1]["id"]
    last_annotation_id = train_json["annotations"][-1]["id"]
    for index, example in enumerate(images_with_caption_json):
        file_name = f"{example['label']}_{example['image']}"
        image_id = last_image_id + index + 1
        image ={
            "id": image_id,
            "license": 1,
            "file_name": file_name,
            "height": 512,
            "width": 512,
            "date_captured": "2023-11-28T19:53:47+00:00"
        }
        train_json["images"].append(image)

        for annotation_index, bbox in enumerate(example["bboxes"]):
            annotation__id = last_annotation_id + annotation_index + 1
            
            bbox_coco = bbox2coco(bbox, 512, 512)
            annotation = {
                "id": annotation__id,
                "image_id": image_id,
                "category_id": categroy_name2id[example["label"]],
                "bbox": bbox_coco,
                "area": bbox_coco[2]*bbox_coco[3],
                "segementation": [],
                "iscrowd": 0
            }
            train_json["annotations"].append(annotation)

    with open(os.path.join(args.dataset_dir, 'annotations', 'train.json'), 'w') as f:
        json.dump(train_json, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Generate_new_annotation', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)