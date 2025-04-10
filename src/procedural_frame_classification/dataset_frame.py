
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

class FrameDataset(Dataset):
    def __init__(self, img_dir, split='train'):
        self.img_dir = os.path.join(img_dir, split)
        self.img_paths = []
        self.labels = []

        # Label 0: Nonprocedural, Label 1: Procedural
        classes = ['nonprocedural', 'procedural']
        class_counts = Counter()

        for label, cls in enumerate(classes):
            cls_path = os.path.join(self.img_dir, cls)
            for fname in os.listdir(cls_path):
                self.img_paths.append(os.path.join(cls_path, fname))
                self.labels.append(label)
                class_counts[cls] += 1

        print(f"[Dataset Initialization - {split}] Class counts:", class_counts)

        # Define transforms
        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:  # val or test
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image = Image.open(self.img_paths[idx]).convert('RGB')
        image = self.transform(image)
        label = self.labels[idx]
        return image, label

#-------------------------------------------------------------
def compute_class_distribution(dataset, class_names):
    """Compute the class distribution of a dataset."""
    labels = []
    for _, label in dataset:
        labels.append(label)
    labels = np.array(labels)
    class_counts = {class_names[i]: np.sum(labels == i) for i in range(len(class_names))}
    total = len(labels)
    class_ratios = {class_name: count / total for class_name, count in class_counts.items()}
    return class_counts, class_ratios
