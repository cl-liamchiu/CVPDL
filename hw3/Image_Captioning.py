from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import shutil
import argparse
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-flan-t5-xl", device_map={"": 0}, torch_dtype=torch.float16
)

def get_args_parser():
    parser = argparse.ArgumentParser('Image_caption', add_help=False)
    parser.add_argument('--dataset_dir', default='',
                    help='train.json path where to load')
    return parser

def generate_caption(img_path):
    image = Image.open(img_path)

    inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)

    generated_ids = model.generate(**inputs)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return generated_text

def do_sample_caption():
    sample_folder_path = "image_caption_samples"

    for img in os.listdir(sample_folder_path):
        img_path = os.path.join(sample_folder_path, img)
        generated_text = generate_caption(img_path)

        print("Image:", img)
        print(generated_text)

def bbox_normalize(bbox, width, height):
    x_min = np.round(bbox[0]/width, 2)
    y_min = np.round(bbox[1]/height, 2)
    x_max = np.round((bbox[0]+bbox[2])/width, 2)
    y_max = np.round((bbox[1]+bbox[3])/height, 2)

    return [x_min, y_min, x_max, y_max]

def main(args):
    with open(os.path.join(args.dataset_dir, 'annotations', 'train.json'), 'r') as f:
        data = json.load(f)

    category_name2id = {}
    category_id2name = {}

    for i in range(1,len(data['categories'])):
        category_name2id[data['categories'][i]['name']] = data['categories'][i]['id']

    for i in range(1,len(data['categories'])):
        category_id2name[data['categories'][i]['id']] = data['categories'][i]['name']

    annotations_df = pd.DataFrame(data['annotations'])

    for i in range(len(data['images'])):
        condition = (annotations_df['image_id']==i)
        # delete images with more than one category
        if len(annotations_df[condition]["category_id"].unique()) > 1:
            annotations_df = annotations_df.drop(annotations_df[condition].index)

        # delete images with more than 6 objects
        elif annotations_df[condition]["category_id"].count() > 6:
            # don't delete jellyfish, because there are fewer than 20 jellyfish images if we delete them.
            if ((annotations_df[condition]["category_id"] == category_name2id['jellyfish']).all()
                & (annotations_df[condition]["category_id"].count() < 20)):
                continue
            annotations_df = annotations_df.drop(annotations_df[condition].index)

    np.random.seed(42)

    image_ids_to_keep = np.array([])

    for name, id in category_name2id.items():
        imgage_ids = np.random.choice(annotations_df[annotations_df['category_id'] == id]["image_id"].unique(), size=20, replace=False)
        image_ids_to_keep = np.concatenate((image_ids_to_keep, imgage_ids))

    keep_df = annotations_df[annotations_df['image_id'].isin(image_ids_to_keep)]

    images_for_captioning = []
    image_folder_path = os.path.join(args.dataset_dir, 'train')
    Path('selected_images').mkdir(parents=True, exist_ok=True)
    for image_id in tqdm(keep_df['image_id'].unique(), desc="Generating captions"):
        image = {}
        image['image'] = data['images'][image_id]['file_name']
        image['label'] = category_id2name[keep_df[keep_df['image_id']==image_id]['category_id'].unique()[0]]
        image['height'] = data['images'][image_id]['height']
        image['width'] = data['images'][image_id]['width']
        bboxes = []
        for box in keep_df[keep_df['image_id']==image_id]['bbox']:
            bboxes.append(bbox_normalize(box, image['width'], image['height']))
        image['bboxes'] = bboxes
        img_path = os.path.join(image_folder_path, image['image'])

        target_file_path = f"selected_images/{image['image']}"

        shutil.copy(img_path, target_file_path)

        generated_text = generate_caption(img_path)
        image['generated_text'] = generated_text
        image['prompt1'] = f"{generated_text}, underwater background, real word, high quality, 8K Ultra HD, high detailed, Composition: shot with a Canon EOS-1D X Mark III, 50mm lens"
        image['prompt2'] = f"{generated_text}, {image['label']}, height: {image['height']}, width: {image['width']}, ocean, undersea background, HD quality, high detailed"

        images_for_captioning.append(image)

    with open('images_with_caption.json', 'w', encoding='utf-8') as f:
        json.dump(images_for_captioning, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Image_caption', parents=[get_args_parser()])
    args = parser.parse_args()
    # do_sample_caption()
    main(args)