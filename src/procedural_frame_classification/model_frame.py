# model_frame.py

import torch
import torch.nn as nn
from torchvision.models import resnet18

def get_model(pretrained=True, freeze_backbone=False):
    # model = resnet18(pretrained=pretrained)
    model = resnet18(weights=pretrained)
    
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace final layer for binary classification
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(model.fc.in_features, 2)
    )
    
    return model