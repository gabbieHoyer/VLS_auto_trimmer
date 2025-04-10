import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from collections import Counter
import numpy as np

import torch
import torchvision.transforms as transforms

import logging
logger = logging.getLogger(__name__)

class FaceDataset(Dataset):
    def __init__(self, img_dir, split='train'):
        self.img_dir = os.path.join(img_dir, split, 'face')
        self.img_paths = []
        self.bboxes = []
        self.original_sizes = []

        for fname in os.listdir(self.img_dir):
            if fname.endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(self.img_dir, fname)
                label_path = os.path.splitext(img_path)[0] + '.txt'
                if os.path.exists(label_path):
                    with open(label_path, 'r') as f:
                        boxes = [list(map(float, line.strip().split())) for line in f.readlines() if line.strip()]
                    if boxes:
                        img = Image.open(img_path).convert('RGB')
                        original_size = img.size
                        self.img_paths.append(img_path)
                        self.bboxes.append(boxes)
                        self.original_sizes.append(original_size)

        print(f"[Dataset Initialization - {split}] Loaded {len(self.img_paths)} images with faces")

        if split == 'train':
            self.transform = transforms.Compose([
                # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # lol this made it suck
                transforms.Resize((112, 112)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((112, 112)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image = Image.open(self.img_paths[idx]).convert('RGB')
        bboxes = torch.tensor(self.bboxes[idx], dtype=torch.float32)
        original_size = self.original_sizes[idx]

        # Apply bounding box jitter during training
        if self.transform.transforms[0].__class__.__name__ != 'Resize':  # Check if in train mode
            bboxes = self.apply_bbox_jitter(bboxes, original_size)

        image = self.transform(image)

        return image, bboxes, original_size

    def apply_bbox_jitter(self, bboxes, original_size, jitter_factor=0.1):
        """
        Apply random jitter to bounding boxes.
        jitter_factor: Maximum fraction of width/height to jitter by.
        """
        bboxes_jittered = bboxes.clone()
        width, height = original_size
        for i in range(len(bboxes)):
            x, y, w, h = bboxes[i]
            # Jitter x_min and y_min
            x_jitter = torch.randn(1).item() * w * jitter_factor
            y_jitter = torch.randn(1).item() * h * jitter_factor
            x_new = x + x_jitter
            y_new = y + y_jitter
            # Ensure new coordinates are within bounds
            x_new = max(0, min(x_new, width - w))
            y_new = max(0, min(y_new, height - h))
            bboxes_jittered[i, 0] = x_new
            bboxes_jittered[i, 1] = y_new
            # Jitter width and height
            w_jitter = torch.randn(1).item() * w * jitter_factor
            h_jitter = torch.randn(1).item() * h * jitter_factor
            w_new = max(10, w + w_jitter)  # Ensure minimum size
            h_new = max(10, h + h_jitter)
            # Ensure new dimensions don't exceed image bounds
            w_new = min(w_new, width - x_new)
            h_new = min(h_new, height - y_new)
            bboxes_jittered[i, 2] = w_new
            bboxes_jittered[i, 3] = h_new
        return bboxes_jittered
    

#-------------------------------------------------------

def custom_collate_fn(batch):
    logger.debug("Using custom_collate_fn to process batch")
    images = []
    bboxes = []
    original_sizes = []
    for image, bbox, original_size in batch:
        images.append(image)
        bboxes.append(bbox)
        original_sizes.append(original_size)
    return torch.stack(images), bboxes, original_sizes



