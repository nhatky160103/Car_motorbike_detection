import os
import torch
import cv2
import random
import yaml
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt
from load_data import load_annotations_from_json
from dataset import CustomDataset, transform
from rpn_layer import FasterRCNN
from PIL import Image
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model_and_dataset(config, list_folder, model_path, phase):
    dataset_config = config['dataset_params']
    model_config = config['model_params']

    image_paths_test=[]
    annotations_test = []

    for folder_path in list_folder:
        im_test, an_test = load_annotations_from_json(folder_path, phase)
        image_paths_test += im_test
        annotations_test += an_test

    test_dataset = CustomDataset(image_paths_test, annotations_test, transform)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    faster_rcnn_model = FasterRCNN(model_config, num_classes=dataset_config['num_classes'])
    faster_rcnn_model.eval()
    faster_rcnn_model.to(device)
    faster_rcnn_model.load_state_dict(torch.load(model_path, map_location=device))
    return faster_rcnn_model, test_dataset, test_dataloader


def display_images(gt_im_list, im_pred_list):
    seperate = torch.zeros(640, 40, 3).numpy().astype(np.uint8)

    for idx, (gt_image, pred_image) in enumerate(zip(gt_im_list, im_pred_list)):

        concat_image = np.concatenate((gt_image, seperate, pred_image), axis=1)

        plt.figure(figsize=(10, 5))
        plt.imshow(concat_image)
        plt.axis('off')
        plt.title(f"Concatenated Image {idx + 1}")
        plt.show()


def save_images(gt_im_list, im_pred_list, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    seperate=torch.zeros(640, 40, 3).numpy()
    for idx, (gt_image, pred_image) in enumerate(zip(gt_im_list, im_pred_list)):

        concat_image = np.concatenate((gt_image,seperate, pred_image), axis=1)

        file_name = os.path.join(save_dir, f"concat_{idx + 1}.png")
        cv2.imwrite(file_name, concat_image)
        print(f"Saved {file_name}")



def infer(config, list_folder, model_path, num_sample):
    faster_rcnn_model, test_dataset, test_dataloader = load_model_and_dataset(config, list_folder, model_path, "test")

    faster_rcnn_model.roi_head.low_score_threshold = 0.7

    gt_im_list = []
    im_pred_list = []
    for sample_count in tqdm(range(num_sample)):
        random_idx = random.randint(0, len(test_dataset) - 1)
        im = test_dataset[random_idx]['image']
        fname = test_dataset[random_idx]['image_path']
        target = test_dataset[random_idx]['target']

        im = im.unsqueeze(0).float().to(device)

        gt_im = cv2.imread(fname)
        gt_im_copy = gt_im.copy()

        for idx, box in enumerate(target['bboxes']):
            x1, y1, x2, y2 = box.detach().cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            cv2.rectangle(gt_im, (x1, y1), (x2, y2), thickness=1, color=[0, 255, 0])
            cv2.rectangle(gt_im_copy, (x1, y1), (x2, y2), thickness=1, color=[0, 255, 0])
            text = config['idx2label'][target['labels'][idx].detach().cpu().item()]
            cv2.putText(gt_im, text, (x1 + 5, y1 + 15), cv2.FONT_ITALIC, 1, [255, 255, 255], 1)
            cv2.putText(gt_im_copy, text, (x1 + 5, y1 + 15), cv2.FONT_ITALIC, 1, [255, 255, 255], 1)

        with torch.no_grad():
            _, frcnn_output = faster_rcnn_model(im)

        boxes = frcnn_output['boxes']
        labels = frcnn_output['labels']
        scores = frcnn_output['scores']

        im_pred = cv2.imread(fname)
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box.detach().cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(im_pred, (x1, y1), (x2, y2), thickness=1, color=[0, 0, 255])
            text = f'{config["idx2label"][labels[idx].item()]}: {scores[idx]:.2f}'
            cv2.putText(im_pred, text, (x1 + 5, y1 + 15), cv2.FONT_ITALIC, 0.5, [255, 255, 255], 1)

        gt_im_list.append(gt_im)
        im_pred_list.append(im_pred)

    # save_images(gt_im_list, im_pred_list, save_dir="test_result")
    display_images(gt_im_list, im_pred_list)


def infer_single_image(folder_path, model_path, config):
    or_im_list = []
    pred_im_list = []

    for image_name in os.listdir(folder_path):
        if image_name == "_annotations.coco.json":
            continue
        image_path= os.path.join(folder_path, image_name)
        dataset_config = config['dataset_params']
        model_config = config['model_params']

        faster_rcnn_model = FasterRCNN(model_config, num_classes=dataset_config['num_classes'])
        faster_rcnn_model.eval()
        faster_rcnn_model.to(device)
        faster_rcnn_model.load_state_dict(torch.load(model_path, map_location=device))

        faster_rcnn_model.roi_head.low_score_threshold = 0.7
        faster_rcnn_model.eval()
        im = Image.open(image_path).convert("RGB")
        im = transform(im)
        im = im.unsqueeze(0).float().to(device)

        origin_image = cv2.imread(image_path)
        origin_image = cv2.resize(origin_image, (640, 640))

        with torch.no_grad():
            _, frcnn_output = faster_rcnn_model(im)

        boxes = frcnn_output['boxes']
        labels = frcnn_output['labels']
        scores = frcnn_output['scores']

        im_pred = cv2.imread(image_path)
        im_pred = cv2.resize(im_pred, (640, 640))
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box.detach().cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(im_pred, (x1, y1), (x2, y2), thickness=1, color=[0, 0, 255])
            text = f'{config["idx2label"][labels[idx].item()]}: {scores[idx]:.2f}'
            cv2.putText(im_pred, text, (x1 + 5, y1 + 15), cv2.FONT_HERSHEY_PLAIN, 0.5, [0, 0, 0], 1)

        or_im_list.append(origin_image)
        pred_im_list.append(im_pred)
    # save_images(or_im_list, pred_im_list, save_dir="test_result")
    display_images(or_im_list, pred_im_list)

if __name__== "__main__":

    with open('config.yaml', 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)


    model_path = config['model_params']['model_path']

    test_folder = "../dataset/cctv_car_bike_detection.v6i.coco/test"
    infer_single_image(test_folder, model_path, config)

    # list_folder = [os.path.join('../dataset',folder) for folder in os.listdir('../dataset')]
    # infer(config,list_folder, model_path, 10 )
