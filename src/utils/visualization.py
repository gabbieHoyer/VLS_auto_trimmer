
import os
import cv2
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
import torch.nn as nn
from torchvision.ops import nms
from torchvision.transforms.functional import to_pil_image

logger = logging.getLogger(__name__)

def visualize_bboxes_yolov8(model, images, gt_bboxes, results, original_sizes, output_dir, epoch, prefix="val", score_threshold=0.9, nms_iou_threshold=0.5):
    """Visualize ground truth and predicted bounding boxes for YOLOv8."""
    model.eval()
    images = images.cpu()
    batch_size = images.shape[0]

    for i in range(min(batch_size, 8)):  # Visualize up to 8 images
        img = images[i].permute(1, 2, 0).numpy()  # [C, H, W] -> [H, W, C]
        img = (img * 255).astype(np.uint8)  # Denormalize if needed
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        orig_w, orig_h = original_sizes[i]

        # Scale image back to original size
        img = cv2.resize(img, (orig_w, orig_h))

        # Draw ground truth boxes (green)
        gt_bbox = gt_bboxes[i].numpy()
        for box in gt_bbox:
            x_min, y_min, w, h = box
            x_min, y_min, w, h = int(x_min), int(y_min), int(w), int(h)
            cv2.rectangle(img, (x_min, y_min), (x_min + w, y_min + h), (0, 255, 0), 2)

        # Draw predicted boxes (red)
        result = results[i]
        if result.boxes is not None:
            boxes = result.boxes.xywh.cpu().numpy()  # [N, 4] (x_center, y_center, w, h)
            scores = result.boxes.conf.cpu().numpy()
            for box, score in zip(boxes, scores):
                if score >= score_threshold:
                    x_center, y_center, w, h = box
                    x_min = int(x_center - w / 2)
                    y_min = int(y_center - h / 2)
                    w, h = int(w), int(h)
                    cv2.rectangle(img, (x_min, y_min), (x_min + w, y_min + h), (0, 0, 255), 2)
                    cv2.putText(img, f"{score:.2f}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Save the image
        output_path = os.path.join(output_dir, f"{prefix}_epoch_{epoch}_img_{i}.png")
        cv2.imwrite(output_path, img)
        logger.info(f"Saved visualization to {output_path}")


# Notes
# This updated visualization function handles YOLOv8’s output format (results objects) and draws ground truth (green) and predicted (red) bounding boxes on the original-sized images.
# Adjust score_threshold and nms_iou_threshold as needed for visualization.

# ---------------------------------------------------------------------------

def visualize_bboxes(model, images, bboxes_list, loc_preds, cls_preds, original_sizes, output_dir, epoch, prefix="train", score_threshold=0.9, nms_iou_threshold=0.5):
    resized_dim = (112, 112)
    cls_preds = torch.sigmoid(cls_preds)
    fig, axs = plt.subplots(1, min(8, images.size(0)), figsize=(20, 5))
    if images.size(0) == 1:
        axs = [axs]
    pred_boxes = model.decode_boxes(loc_preds, images.device)

    for i in range(min(8, images.size(0))):
        img = images[i].cpu().detach().permute(1, 2, 0).numpy() * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        axs[i].imshow(img)
        original_dim = original_sizes[i]
        scale_x = resized_dim[1] / original_dim[0]
        scale_y = resized_dim[0] / original_dim[1]
        gt_boxes = bboxes_list[i].cpu().numpy()
        for box in gt_boxes:
            x, y, w, h = box
            x *= scale_x
            y *= scale_y
            w *= scale_x
            h *= scale_y
            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='g', facecolor='none', label='GT')
            axs[i].add_patch(rect)

        # Apply NMS
        pred_boxes_i = pred_boxes[i]  # [N, 4] (x_min, y_min, w, h)
        pred_scores_i = cls_preds[i, :, 1]  # [N]
        # Convert to (x1, y1, x2, y2) format for NMS
        boxes_nms = torch.zeros_like(pred_boxes_i)
        boxes_nms[:, 0] = pred_boxes_i[:, 0]  # x1
        boxes_nms[:, 1] = pred_boxes_i[:, 1]  # y1
        boxes_nms[:, 2] = pred_boxes_i[:, 0] + pred_boxes_i[:, 2]  # x2
        boxes_nms[:, 3] = pred_boxes_i[:, 1] + pred_boxes_i[:, 3]  # y2
        keep = nms(boxes_nms, pred_scores_i, iou_threshold=nms_iou_threshold)
        pred_boxes_i = pred_boxes_i[keep].cpu().detach().numpy()
        pred_scores_i = pred_scores_i[keep].cpu().detach().numpy()

        for j, (box, score) in enumerate(zip(pred_boxes_i, pred_scores_i)):
            if score > score_threshold:
                x, y, w, h = box
                rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none', label='Pred')
                axs[i].add_patch(rect)
                score_y = min(resized_dim[0] - 10, max(0, y + h + 5))
                score_x = max(0, min(resized_dim[1] - 20, x))
                axs[i].text(score_x, score_y, f'{score:.2f}', color='r', fontsize=8, bbox=dict(facecolor='white', alpha=0.5))
        axs[i].axis('off')
        axs[i].set_title(f'GT: {len(gt_boxes)} faces')

    handles = [patches.Rectangle((0, 0), 1, 1, facecolor='none', edgecolor='g', label='Ground Truth'),
               patches.Rectangle((0, 0), 1, 1, facecolor='none', edgecolor='r', label='Prediction')]
    fig.legend(handles=handles, loc='upper center', ncol=2, bbox_to_anchor=(0.5, -0.05))
    output_path = os.path.join(output_dir, f'{prefix}_sample_predictions_epoch_{epoch}.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved {prefix} sample predictions to {output_path}")

# ----------------------------------------------------------------------------

def visualize_bboxes_initial(model, images, bboxes_list, loc_preds, cls_preds, original_sizes, output_dir, epoch, prefix="train"):
    resized_dim = (112, 112)
    cls_preds = torch.sigmoid(cls_preds)
    fig, axs = plt.subplots(1, min(8, images.size(0)), figsize=(20, 5))
    if images.size(0) == 1:
        axs = [axs]
    pred_boxes = model.decode_boxes(loc_preds, images.device)

    for i in range(min(8, images.size(0))):
        img = images[i].cpu().detach().permute(1, 2, 0).numpy() * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        axs[i].imshow(img)
        original_dim = original_sizes[i]
        scale_x = resized_dim[1] / original_dim[0]
        scale_y = resized_dim[0] / original_dim[1]
        gt_boxes = bboxes_list[i].cpu().numpy()
        for box in gt_boxes:
            x, y, w, h = box
            x *= scale_x
            y *= scale_y
            w *= scale_x
            h *= scale_y
            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='g', facecolor='none', label='GT')
            axs[i].add_patch(rect)
        pred_boxes_i = pred_boxes[i].cpu().detach().numpy()
        pred_scores = cls_preds[i, :, 1].cpu().detach().numpy()
        for j, (box, score) in enumerate(zip(pred_boxes_i, pred_scores)):
            if score > 0.5: #0.5:
                x, y, w, h = box
                rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none', label='Pred')
                axs[i].add_patch(rect)
                score_y = min(resized_dim[0] - 10, max(0, y + h + 5))
                score_x = max(0, min(resized_dim[1] - 20, x))
                axs[i].text(score_x, score_y, f'{score:.2f}', color='r', fontsize=8, bbox=dict(facecolor='white', alpha=0.5))
        axs[i].axis('off')
        axs[i].set_title(f'GT: {len(gt_boxes)} faces')

    handles = [patches.Rectangle((0, 0), 1, 1, facecolor='none', edgecolor='g', label='Ground Truth'),
               patches.Rectangle((0, 0), 1, 1, facecolor='none', edgecolor='r', label='Prediction')]
    fig.legend(handles=handles, loc='upper center', ncol=2, bbox_to_anchor=(0.5, -0.05))
    output_path = os.path.join(output_dir, f'{prefix}_sample_predictions_epoch_{epoch}.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved {prefix} sample predictions to {output_path}")