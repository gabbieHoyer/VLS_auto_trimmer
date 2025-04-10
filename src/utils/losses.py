
import torch
import torch.nn as nn
from torchvision.ops import box_iou

import os
import logging

logger = logging.getLogger(__name__)

def focal_loss(cls_preds, cls_targets, alpha=0.5, gamma=3.0, reduction='sum'):
    cls_preds = torch.sigmoid(cls_preds)
    pos_mask = cls_targets == 1
    neg_mask = cls_targets == 0
    pos_loss = -alpha * (1 - cls_preds[pos_mask]) ** gamma * torch.log(cls_preds[pos_mask] + 1e-8)
    neg_loss = -(1 - alpha) * cls_preds[neg_mask] ** gamma * torch.log(1 - cls_preds[neg_mask] + 1e-8)
    loss = pos_loss.sum() + neg_loss.sum()
    if reduction == 'sum':
        return loss
    elif reduction == 'mean':
        return loss / (pos_mask.sum() + neg_mask.sum() + 1e-8)
    else:
        return loss
def compute_loss(model, cls_preds, loc_preds, bboxes_list, original_sizes, device):
    batch_size = cls_preds.size(0)
    num_preds = cls_preds.size(1)
    cls_targets = torch.zeros(batch_size, num_preds, 2, device=device)
    loc_targets = torch.zeros(batch_size, num_preds, 4, device=device)
    pos_mask = torch.zeros(batch_size, num_preds, dtype=torch.bool, device=device)
    pred_boxes = model.decode_boxes(loc_preds, device)

    resized_dim = (112, 112)
    for b in range(batch_size):
        gt_boxes = bboxes_list[b].to(device)
        logger.debug(f"Batch {b}: gt_boxes_shape={gt_boxes.shape}, original_size={original_sizes[b]}")
        if len(gt_boxes) == 0:
            cls_targets[b, :, 0] = 1  # All predictions are negative if no GT boxes
            continue
        original_dim = original_sizes[b]
        scale_x = resized_dim[1] / original_dim[0]
        scale_y = resized_dim[0] / original_dim[1]
        gt_boxes_scaled = gt_boxes.clone()
        gt_boxes_scaled[:, 0] *= scale_x
        gt_boxes_scaled[:, 1] *= scale_y
        gt_boxes_scaled[:, 2] *= scale_x
        gt_boxes_scaled[:, 3] *= scale_y
        iou_matrix = box_iou(pred_boxes[b], gt_boxes_scaled)
        max_iou, max_idx = iou_matrix.max(dim=1)
        logger.debug(f"Batch {b}: iou_matrix_shape={iou_matrix.shape}, max_iou={max_iou}, max_idx={max_idx}")
        for p in range(num_preds):
            if max_iou[p] >= 0.3:
                gt_idx = max_idx[p].item()
                if gt_idx < len(gt_boxes_scaled):  # Safety check
                    loc_targets[b, p] = gt_boxes_scaled[gt_idx]
                    cls_targets[b, p, 1] = 1
                    pos_mask[b, p] = 1
                else:
                    logger.warning(f"Batch {b}, pred {p}: max_idx[{p}]={gt_idx} exceeds gt_boxes_scaled size {len(gt_boxes_scaled)}")
            else:
                cls_targets[b, p, 0] = 1

    # Hard negative mining
    neg_mask = (cls_targets[:, :, 0] == 1) & (~pos_mask)
    cls_preds_sigmoid = torch.sigmoid(cls_preds)
    neg_scores = cls_preds_sigmoid[neg_mask][:, 1]  # Confidence for negative class
    num_neg = neg_mask.sum().item()
    num_pos = pos_mask.sum().item()
    num_hard_neg = min(3 * num_pos, num_neg)
    logger.debug(f"Positives: {num_pos}, Negatives: {num_neg}, Hard negatives: {num_hard_neg}")
    if num_hard_neg > 0 and neg_scores.numel() > 0:
        neg_scores, indices = neg_scores.topk(min(num_hard_neg, neg_scores.numel()))
        hard_neg_mask = torch.zeros_like(neg_mask, dtype=torch.bool, device=device)
        hard_neg_mask[neg_mask] = False
        hard_neg_mask.view(-1)[indices] = True
        neg_mask = hard_neg_mask
    else:
        neg_mask = torch.zeros_like(neg_mask, dtype=torch.bool, device=device)

    cls_loss_mask = pos_mask | neg_mask
    cls_preds_masked = cls_preds[cls_loss_mask]
    cls_targets_masked = cls_targets[cls_loss_mask]
    if cls_preds_masked.numel() > 0:
        cls_loss = focal_loss(cls_preds_masked, cls_targets_masked, alpha=0.5, gamma=3.0, reduction='mean')
    else:
        cls_loss = torch.tensor(0.0, device=device)
        logger.warning("No samples for cls_loss computation")

    loc_loss = nn.SmoothL1Loss(reduction='mean')(pred_boxes[pos_mask], loc_targets[pos_mask]) if pos_mask.sum() > 0 else torch.tensor(0.0, device=device)
    total_loss = cls_loss + loc_loss
    # total_loss = cls_loss + 2.0 * loc_loss  # Emphasize localization loss
    logger.debug(f"cls_loss={cls_loss.item()}, loc_loss={loc_loss.item()}, total_loss={total_loss.item()}")
    return total_loss
