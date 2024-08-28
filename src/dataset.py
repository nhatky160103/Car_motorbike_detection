import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import torch
from torchvision import transforms


class CustomDataset(Dataset):
    def __init__(self, image_paths, annotations, transform = None):
        self.image_paths = image_paths
        self.annotations = annotations
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        boxes = torch.tensor(self.annotations[idx]['bboxes'], dtype=torch.float32)
        labels = torch.tensor(self.annotations[idx]['labels'], dtype=torch.int64)

        target = {
            "bboxes": boxes,
            "labels": labels
        }

        return {'image_path': self.image_paths[idx], 'image': image, "target": target}


def collate_fn(batch):
    images = [item['image'] for item in batch]
    targets = [item['target'] for item in batch]
    image_paths = [item['image_path'] for item in batch]

    images = torch.stack(images, dim=0)

    max_box_per_image = max(target['bboxes'].shape[0] for target in targets)

    batch_bboxes = torch.zeros((len(batch), max_box_per_image, 4), dtype=torch.float32)
    batch_labels = torch.zeros((len(batch), max_box_per_image), dtype=torch.int64)

    for i, target in enumerate(targets):
        num_boxes = target['bboxes'].shape[0]
        batch_bboxes[i, :num_boxes] = target['bboxes']
        batch_labels[i, :num_boxes] = target['labels']

    # Create the final target dictionary
    batch_targets = {
        'bboxes': batch_bboxes,
        'labels': batch_labels
    }

    return {'image_path': image_paths, 'image': images, 'target': batch_targets}


transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
])



