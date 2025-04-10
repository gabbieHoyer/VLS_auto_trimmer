
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim
from torchvision.ops import box_iou

from src.utils import gpu_setup as GPUSetup

def compute_metrics(model, cls_preds, loc_preds, bboxes_list, original_sizes, device, iou_threshold=0.3):
    cls_preds = torch.sigmoid(cls_preds)
    batch_size = cls_preds.size(0)
    num_preds = cls_preds.size(1)
    total_tp = torch.tensor(0, dtype=torch.float, device=device)
    total_fp = torch.tensor(0, dtype=torch.float, device=device)
    total_fn = torch.tensor(0, dtype=torch.float, device=device)
    total_iou = torch.tensor(0.0, dtype=torch.float, device=device)
    resized_dim = (112, 112)
    pred_boxes = model.decode_boxes(loc_preds, device)

    for b in range(batch_size):
        gt_boxes = bboxes_list[b].to(device)
        if len(gt_boxes) == 0:
            continue
        original_dim = original_sizes[b]
        scale_x = resized_dim[1] / original_dim[0]
        scale_y = resized_dim[0] / original_dim[1]
        gt_boxes_scaled = gt_boxes.clone()
        gt_boxes_scaled[:, 0] *= scale_x
        gt_boxes_scaled[:, 1] *= scale_y
        gt_boxes_scaled[:, 2] *= scale_x
        gt_boxes_scaled[:, 3] *= scale_y
        cls_prob = cls_preds[b, :, 1]
        iou_matrix = box_iou(pred_boxes[b], gt_boxes_scaled)
        max_iou, max_idx = iou_matrix.max(dim=1)
        matched_gt = set()
        for p in range(num_preds):
            if cls_prob[p] > 0.5 and max_iou[p] >= iou_threshold:
                gt_idx = max_idx[p].item()
                if gt_idx not in matched_gt:
                    total_tp += 1
                    total_iou += max_iou[p]
                    matched_gt.add(gt_idx)
            elif cls_prob[p] > 0.5:
                total_fp += 1
        total_fn += max(0, len(gt_boxes) - len(matched_gt))

    if GPUSetup.is_distributed():
        dist.all_reduce(total_tp)
        dist.all_reduce(total_fp)
        dist.all_reduce(total_fn)
        dist.all_reduce(total_iou)

    total_tp = total_tp.item()
    total_fp = total_fp.item()
    total_fn = total_fn.item()
    total_iou = total_iou.item()

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    mean_iou = total_iou / total_tp if total_tp > 0 else 0.0
    return precision, recall, mean_iou