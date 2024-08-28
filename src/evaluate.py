import numpy as np
import torch
import matplotlib.pyplot as plt
from infer import load_model_and_dataset
import yaml
from tqdm import tqdm

def get_iou(det, gt):
    r"""
    Method to compute iou between two boxes.
    :param det: List[float] box1 coordinates [x1, y1, x2, y2]
    :param gt: List[float] box2 coordinates [x1, y1, x2, y2]
    :return iou: (float) Intersection over union between det and gt
    """
    det_x1, det_y1, det_x2, det_y2 = det
    gt_x1, gt_y1, gt_x2, gt_y2 = gt

    x_left = max(det_x1, gt_x1)
    y_top = max(det_y1, gt_y1)
    x_right = min(det_x2, gt_x2)
    y_bottom = min(det_y2, gt_y2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    area_intersection = (x_right - x_left) * (y_bottom - y_top)
    det_area = (det_x2 - det_x1) * (det_y2 - det_y1)
    gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
    area_union = float(det_area + gt_area - area_intersection + 1E-6)
    iou = area_intersection / area_union
    return iou


def compute_map(det_boxes, gt_boxes, iou_threshold, method='interp'):
    r"""
    Method to calculate Mean Average Precision between two sets of boxes.
    :param det_boxes: List[Dict[List[float]]] prediction boxes for ALL images
    :param gt_boxes: List[Dict[List[float]]] ground truth boxes for ALL images
    :param iou_threshold: (float) Threshold used for true positive.
    :param method: (str) One of area/interp. Default:interp
    :return: mean_ap, all_aps: Tuple(float, Dict[float])
    """
    gt_labels = {cls_key for im_gt in gt_boxes for cls_key in im_gt.keys()}
    all_aps = {}
    aps = []
    precisions = {}
    recalls = {}

    for idx, label in enumerate(gt_labels):
        cls_dets = [
            [im_idx, im_dets_label] for im_idx, im_dets in enumerate(det_boxes)
            if label in im_dets for im_dets_label in im_dets[label]
        ]

        cls_dets = sorted(cls_dets, key=lambda k: -k[1][-1])

        gt_matched = [[False for _ in im_gts[label]] for im_gts in gt_boxes]
        num_gts = sum([len(im_gts[label]) for im_gts in gt_boxes])
        tp = [0] * len(cls_dets)
        fp = [0] * len(cls_dets)

        for det_idx, (im_idx, det_pred) in enumerate(cls_dets):
            im_gts = gt_boxes[im_idx][label]
            max_iou_found = -1
            max_iou_gt_idx = -1

            for gt_box_idx, gt_box in enumerate(im_gts):
                gt_box_iou = get_iou(det_pred[:-1], gt_box)
                if gt_box_iou > max_iou_found:
                    max_iou_found = gt_box_iou
                    max_iou_gt_idx = gt_box_idx
            if max_iou_found < iou_threshold or gt_matched[im_idx][max_iou_gt_idx]:
                fp[det_idx] = 1
            else:
                tp[det_idx] = 1
                gt_matched[im_idx][max_iou_gt_idx] = True
        tp = np.cumsum(tp)
        fp = np.cumsum(fp)

        eps = np.finfo(np.float32).eps
        recalls[label] = tp / np.maximum(num_gts, eps)
        precisions[label] = tp / np.maximum((tp + fp), eps)

        if method == 'area':
            recalls[label] = np.concatenate(([0.0], recalls[label], [1.0]))
            precisions[label] = np.concatenate(([0.0], precisions[label], [0.0]))

            for i in range(precisions[label].size - 1, 0, -1):
                precisions[label][i - 1] = np.maximum(precisions[label][i - 1], precisions[label][i])
            i = np.where(recalls[label][1:] != recalls[label][:-1])[0]
            ap = np.sum((recalls[label][i + 1] - recalls[label][i]) * precisions[label][i + 1])
        elif method == 'interp':
            ap = 0.0
            for interp_pt in np.arange(0, 1 + 1E-3, 0.1):
                prec_interp_pt = precisions[label][recalls[label] >= interp_pt]
                prec_interp_pt = prec_interp_pt.max() if prec_interp_pt.size > 0.0 else 0.0
                ap += prec_interp_pt
            ap = ap / 11.0
        else:
            raise ValueError('Method can only be area or interp')
        if num_gts > 0:
            aps.append(ap)
            all_aps[label] = ap
        else:
            all_aps[label] = np.nan

        # Plot Precision-Recall Curve for this class
        # plt.figure(figsize=(8, 6))
        # plt.plot(recalls[label], precisions[label], linestyle='-', linewidth=0.5,
        #          label=f'Class {label}')
        # plt.xlabel('Recall')
        # plt.ylabel('Precision')
        # plt.title(f'Precision-Recall Curve for Class {label}')
        # plt.legend()
        # plt.grid(True)
        # plt.show()

    mean_ap = sum(aps) / (len(aps) + 1E-6)
    return mean_ap, all_aps


def nms(dets, nms_threshold=0.5):
    r"""
    Method to do non-maximum suppression.
    :param dets: List[List[float]] detections for this image [[x1, y1, x2, y2, score], ...]
    :param nms_threshold: iou used for rejecting boxes. Default:0.5
    :return: Filtered sets of detections List[List[float]]
    """
    sorted_dets = sorted(dets, key=lambda k: -k[-1])

    keep_dets = []
    while len(sorted_dets) > 0:
        keep_dets.append(sorted_dets[0])
        sorted_dets = [
            box for box in sorted_dets[1:]
            if get_iou(sorted_dets[0][:-1], box[:-1]) < nms_threshold
        ]
    return keep_dets


def evaluate_map(test_dataloader, faster_rcnn_model, device):
    """
    Evaluate Mean Average Precision (mAP) for a Faster R-CNN model.

    :param test_dataloader: DataLoader for the test dataset.
    :param faster_rcnn_model: Trained Faster R-CNN model.
    :param device: Device to use (e.g., 'cuda' or 'cpu').
    """
    faster_rcnn_model.eval()
    all_gts = []
    all_preds = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            im = batch['image'].float().to(device)
            target = batch['target']
            target['bboxes'] = target['bboxes'].float().to(device)
            target['labels'] = target['labels'].long().to(device)

            # Chạy mô hình để lấy dự đoán
            _, frcnn_output = faster_rcnn_model(im)
            boxes = frcnn_output['boxes'].detach().cpu().numpy()
            labels = frcnn_output['labels'].detach().cpu().numpy()
            scores = frcnn_output['scores'].detach().cpu().numpy()

            true_boxes = target['bboxes'].detach().cpu().numpy()[0]
            true_labels = target['labels'].detach().cpu().numpy()[0]

            pred_boxes_dict = {}
            gt_boxes_dict = {}
            for label in [1, 2]:
                label_str = str(label)
                pred_boxes_dict[label_str] = []
                gt_boxes_dict[label_str] = []

            for box, label, score in zip(boxes, labels, scores):
                label_str = str(label)
                pred_boxes_dict[label_str].append([*box, score])

            for box, label in zip(true_boxes, true_labels):
                label_str = str(label)
                gt_boxes_dict[label_str].append(box)

            all_preds.append(pred_boxes_dict)
            all_gts.append(gt_boxes_dict)

        map_per_threshold = []
        ap_per_threshold = []
        for i in np.arange(0.5, 1.0, 0.05):
            mean_ap, all_aps = compute_map(all_preds, all_gts, i, method='interp')
            map_per_threshold.append(mean_ap)
            ap_per_threshold.append(all_aps)
        print(map_per_threshold)
        print(ap_per_threshold)

        mAP = sum(map_per_threshold) / len(map_per_threshold)
        print("mAP :", mAP)


if __name__ == "__main__":
    with open('config.yaml', 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)

    dataset_config = config['dataset_params']
    model_config = config['model_params']
    train_config = config['train_params']


    model_path = model_config['model_path']
    list_folder = ["dataset/cctv_car_bike_detection.v6i.coco", "dataset/Vietnamese vehicle.v3-2023-02-01-5-31pm.coco"]

    faster_rcnn_model, test_dataset, test_dataloader = load_model_and_dataset(config, list_folder, model_path, "test")
    faster_rcnn_model.roi_head.low_score_threshold = 0.7

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    evaluate_map(test_dataloader, faster_rcnn_model, device)