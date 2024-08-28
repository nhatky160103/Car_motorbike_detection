from dataset import CustomDataset, transform
from torch.utils.data import DataLoader, Dataset
import yaml
import os
import json
from collections import defaultdict

# This function aim to load annotation file from difference dataset each dataset have difference num of class(> 3), so
# we need to label them into 2 class
def load_annotations_from_json(folder_path, phase):
    json_file = os.path.join(folder_path, phase, '_annotations.coco.json')
    with open(json_file, 'r') as f:
        data = json.load(f)

    id_to_image_path = {image['id']: os.path.join(folder_path, phase, image['file_name']) for image in data['images']}
    id_to_category_name = {category['id']: category['name'] for category in data['categories']}

    image_annotations_map = defaultdict(lambda: {'bboxes': [], 'labels': []})

    for annotation in data['annotations']:
        image_id = annotation['image_id']
        category_id = annotation['category_id']
        if category_id == 5 or category_id == 0:
            continue

        bbox = annotation['bbox']
        x1, y1, width, height = bbox
        x2 = x1 + width
        y2 = y1 + height

        # 0: car
        # 1: motorbike

        if len(data['categories']) == 3:  # 1:car 2:motorbike
            label = 1 if category_id == 1 else 2
        elif len(data['categories']) == 7:  # 1:bicycle 2:bus 3:car 4:motorbike 5:person 6: truck
            label = 1 if category_id in [2, 3, 6] else 2

        elif len(data['categories']) == 5:  # 1:car 2:motobike 3:truck 4:bus
            label = 1 if category_id in [1, 3, 4] else 2

        image_annotations_map[image_id]['bboxes'].append([x1, y1, x2, y2])
        image_annotations_map[image_id]['labels'].append(label)

    image_paths = []
    annotations = []

    for image_id, image_path in id_to_image_path.items():
        image_annotations = image_annotations_map[image_id]
        if image_annotations['bboxes']:
            image_paths.append(image_path)
            annotations.append(image_annotations)

    return image_paths, annotations


def load_data(list_folder_path):
    with open('config.yaml', 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)

    dataset_config = config['dataset_params']
    model_config = config['model_params']
    train_config = config['train_params']

    image_paths_train = []
    annotations_train = []
    image_paths_val = []
    annotations_val = []

    for folder_path in list_folder_path:
        im_train, an_train = load_annotations_from_json(folder_path, "train")
        im_val, an_val = load_annotations_from_json(folder_path, "valid")
        image_paths_train += im_train
        annotations_train += an_train
        image_paths_val += im_val
        annotations_val += an_val
    train_dataset = CustomDataset(image_paths_train, annotations_train, transform)
    val_dataset = CustomDataset(image_paths_val, annotations_val, transform)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=train_config['batch_size'],
                                  shuffle=True,
                                  num_workers=4,
                                  )

    val_dataloader = DataLoader(val_dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=4)

    dataloader_dict = {"train": train_dataloader, 'val': val_dataloader}
    return dataloader_dict


if __name__ == "__main__":

    list_folder = [os.path.join('dataset',folder) for folder in os.listdir('dataset')]
    dataloader_dict = load_data(list_folder)
    print(len(dataloader_dict['train']))
    print(len(dataloader_dict['val']))
