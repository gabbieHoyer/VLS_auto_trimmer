
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

# YOLOTinyFaceNet: Extremely lightweight, fast inference, good for UI integration. May improve recall on intermittent faces due to a larger feature map (7x7 vs. 4x4).
# EfficientFaceNet: Balances efficiency and accuracy with a stronger backbone (EfficientNet-B0). The 14x14 feature map could capture finer details in surgical videos, potentially improving precision.

class BaseFaceDetector(nn.Module):
    def __init__(self):
        super(BaseFaceDetector, self).__init__()

    def forward(self, x):
        # Must return cls_preds [B, N, 2], loc_preds [B, N, 4]
        raise NotImplementedError

    def decode_boxes(self, loc_preds, device):
        # Must return decoded boxes [B, N, 4] (x_min, y_min, w, h)
        raise NotImplementedError
    
# -----------------------------------------

class MobileFaceNet(BaseFaceDetector):
    def __init__(self, weights=True, num_anchors=3):
        super(MobileFaceNet, self).__init__()
        # Existing initialization code...
        if weights:
            backbone = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        else:
            backbone = mobilenet_v2(weights=None)
        self.features = backbone.features
        self.conv = nn.Conv2d(1280, 512, 1)
        self.cls_head = nn.Conv2d(512, num_anchors * 2, 1)
        self.loc_head = nn.Conv2d(512, num_anchors * 4, 1)
        self.num_anchors = num_anchors
        self.feature_map_size = (4, 4)
        self.resized_dim = (112, 112)
        self.anchors = self._generate_anchors()

    def _generate_anchors(self):
        H, W = self.feature_map_size
        base_sizes = [23, 31, 49]
        aspect_ratios = [1.0, 2.0, 0.5]
        anchors = []
        stride_x = self.resized_dim[1] / W
        stride_y = self.resized_dim[0] / H
        for h in range(H):
            for w in range(W):
                center_x = (w + 0.5) * stride_x
                center_y = (h + 0.5) * stride_y
                for size in base_sizes:
                    for ar in aspect_ratios:
                        w_box = size
                        h_box = size * ar
                        w_box = max(stride_x / 2, min(w_box, stride_x * 2))
                        h_box = max(stride_y / 2, min(h_box, stride_y * 2))
                        anchors.append([center_x - w_box / 2, center_y - h_box / 2, w_box, h_box])
        return torch.tensor(anchors, dtype=torch.float32)

    def decode_boxes(self, loc_preds, device):
        """
        Decode loc_preds (offsets) into absolute bounding boxes using anchor boxes.
        loc_preds: [B, H*W*num_anchors, 4] (dx, dy, dw, dh)
        Returns: [B, H*W*num_anchors, 4] (x_min, y_min, width, height)
        """
        anchors = self.anchors.to(device)  # [H*W*num_anchors, 4]
        batch_size = loc_preds.size(0)
        num_preds = loc_preds.size(1)

        # Expand anchors to match batch size
        anchors = anchors.unsqueeze(0).expand(batch_size, -1, -1)  # [B, H*W*num_anchors, 4]

        # Decode offsets
        pred_boxes = torch.zeros_like(loc_preds)
        pred_boxes[:, :, 0] = loc_preds[:, :, 0] * anchors[:, :, 2] + anchors[:, :, 0]  # x_min = dx * anchor_w + anchor_x
        pred_boxes[:, :, 1] = loc_preds[:, :, 1] * anchors[:, :, 3] + anchors[:, :, 1]  # y_min = dy * anchor_h + anchor_y
        pred_boxes[:, :, 2] = torch.exp(loc_preds[:, :, 2]) * anchors[:, :, 2]          # width = exp(dw) * anchor_w
        pred_boxes[:, :, 3] = torch.exp(loc_preds[:, :, 3]) * anchors[:, :, 3]          # height = exp(dh) * anchor_h

        return pred_boxes
    
    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        cls = self.cls_head(x)
        loc = self.loc_head(x)
        B, _, H, W = cls.shape
        cls = cls.permute(0, 2, 3, 1).reshape(B, -1, 2)
        loc = loc.permute(0, 2, 3, 1).reshape(B, -1, 4)
        return cls, loc
    
# -------------

class YOLOTinyFaceNet(BaseFaceDetector):
    def __init__(self, num_anchors=3):
        super(YOLOTinyFaceNet, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 56x56
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28x28
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 7x7
        )
        self.conv = nn.Conv2d(128, 256, 3, padding=1)
        self.cls_head = nn.Conv2d(256, num_anchors * 2, 1)
        self.loc_head = nn.Conv2d(256, num_anchors * 4, 1)
        self.num_anchors = num_anchors
        self.feature_map_size = (7, 7)  # 112/16 ≈ 7
        self.resized_dim = (112, 112)
        self.anchors = self._generate_anchors()

    def _generate_anchors(self):
        H, W = self.feature_map_size
        base_sizes = [16, 31, 49]
        aspect_ratios = [1.0, 2.0, 0.5]
        anchors = []
        stride_x = self.resized_dim[1] / W
        stride_y = self.resized_dim[0] / H
        for h in range(H):
            for w in range(W):
                center_x = (w + 0.5) * stride_x
                center_y = (h + 0.5) * stride_y
                for size in base_sizes:
                    for ar in aspect_ratios:
                        w_box = size
                        h_box = size * ar
                        w_box = max(stride_x / 2, min(w_box, stride_x * 2))
                        h_box = max(stride_y / 2, min(h_box, stride_y * 2))
                        anchors.append([center_x - w_box / 2, center_y - h_box / 2, w_box, h_box])
        return torch.tensor(anchors, dtype=torch.float32)

    def decode_boxes(self, loc_preds, device):
        anchors = self.anchors.to(device)
        batch_size = loc_preds.size(0)
        anchors = anchors.unsqueeze(0).expand(batch_size, -1, -1)
        pred_boxes = torch.zeros_like(loc_preds)
        pred_boxes[:, :, 0] = loc_preds[:, :, 0] * anchors[:, :, 2] + anchors[:, :, 0]
        pred_boxes[:, :, 1] = loc_preds[:, :, 1] * anchors[:, :, 3] + anchors[:, :, 1]
        pred_boxes[:, :, 2] = torch.exp(loc_preds[:, :, 2]) * anchors[:, :, 2]
        pred_boxes[:, :, 3] = torch.exp(loc_preds[:, :, 3]) * anchors[:, :, 3]
        return pred_boxes

    def forward(self, x):
        x = self.backbone(x)
        x = self.conv(x)
        cls = self.cls_head(x)
        loc = self.loc_head(x)
        B, _, H, W = cls.shape
        cls = cls.permute(0, 2, 3, 1).reshape(B, -1, 2)
        loc = loc.permute(0, 2, 3, 1).reshape(B, -1, 4)
        return cls, loc
    
# -----------

class EfficientFaceNet(BaseFaceDetector):
    def __init__(self, num_anchors=9):  # Updated to 9 as per your setup
        super(EfficientFaceNet, self).__init__()
        from torchvision.models import efficientnet_b0
        backbone = efficientnet_b0(weights="IMAGENET1K_V1")
        self.features = nn.Sequential(*list(backbone.features.children())[:4])  # Up to 14x14
        self.conv = nn.Conv2d(40, 256, 1)  # Adjusted to 40 input channels
        self.cls_head = nn.Conv2d(256, num_anchors * 2, 1)
        self.loc_head = nn.Conv2d(256, num_anchors * 4, 1)
        self.num_anchors = num_anchors
        self.feature_map_size = (14, 14)  # 112/8 ≈ 14, now correct
        self.resized_dim = (112, 112)
        self.anchors = self._generate_anchors()

    def _generate_anchors(self):
        H, W = self.feature_map_size
        base_sizes = [8, 16, 31]  # Small, medium, large
        aspect_ratios = [1.0, 2.0, 0.5]
        anchors = []
        stride_x = self.resized_dim[1] / W
        stride_y = self.resized_dim[0] / H
        for h in range(H):
            for w in range(W):
                center_x = (w + 0.5) * stride_x
                center_y = (h + 0.5) * stride_y
                for size in base_sizes:
                    for ar in aspect_ratios:
                        w_box = size
                        h_box = size * ar
                        w_box = max(stride_x / 2, min(w_box, stride_x * 4))  # Relaxed cap
                        h_box = max(stride_y / 2, min(h_box, stride_y * 4))
                        anchors.append([center_x - w_box / 2, center_y - h_box / 2, w_box, h_box])
        return torch.tensor(anchors, dtype=torch.float32)

    def decode_boxes(self, loc_preds, device):
        anchors = self.anchors.to(device)
        batch_size = loc_preds.size(0)
        anchors = anchors.unsqueeze(0).expand(batch_size, -1, -1)
        pred_boxes = torch.zeros_like(loc_preds)
        pred_boxes[:, :, 0] = loc_preds[:, :, 0] * anchors[:, :, 2] + anchors[:, :, 0]
        pred_boxes[:, :, 1] = loc_preds[:, :, 1] * anchors[:, :, 3] + anchors[:, :, 1]
        pred_boxes[:, :, 2] = torch.exp(loc_preds[:, :, 2]) * anchors[:, :, 2]
        pred_boxes[:, :, 3] = torch.exp(loc_preds[:, :, 3]) * anchors[:, :, 3]
        return pred_boxes

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        cls = self.cls_head(x)
        loc = self.loc_head(x)
        B, _, H, W = cls.shape
        cls = cls.permute(0, 2, 3, 1).reshape(B, -1, 2)
        loc = loc.permute(0, 2, 3, 1).reshape(B, -1, 4)
        return cls, loc


# ------------------------------------------------


# def convert_to_yolo_format(bbox, img_width, img_height):
#     x, y, w, h = bbox
#     x_center = (x + w / 2) / img_width
#     y_center = (y + h / 2) / img_height
#     w_norm = w / img_width
#     h_norm = h / img_height
#     return [0, x_center, y_center, w_norm, h_norm]  # class_id=0 for faces


# #data.yaml:
# train: path/to/train/images
# val: path/to/val/images
# nc: 1  # Number of classes (1 for faces)
# names: ['face']


# from ultralytics import YOLO
# model = YOLO("yolov8n.pt")  # Load a pretrained model
# model.train(data="path/to/your_dataset.yaml", epochs=100, imgsz=112)  # Train on your dataset