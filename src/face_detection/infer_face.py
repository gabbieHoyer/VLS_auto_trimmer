import os
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
from torch.utils.data import DataLoader
from torchvision.ops import box_iou, nms
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from src.utils import gpu_setup as GPUSetup
from src.utils import load_config, compute_metrics
from src.face_detection import (
    FaceDataset,
    custom_collate_fn,
    MobileFaceNet,
    EfficientFaceNet,
    YOLOTinyFaceNet,
)

# Retrieve a logger for the module
logger = logging.getLogger(__name__)

def main_process_only(func):
    """Decorator to ensure function runs only on the main process."""
    def wrapper(*args, **kwargs):
        if GPUSetup.is_main_process():
            return func(*args, **kwargs)
    return wrapper

@main_process_only
def log_info(message):
    """Log info messages only on the main process."""
    logger.info(message)

def save_metrics_to_file(output_dir, avg_precision, avg_recall, avg_iou, test_mAP, test_mAP_50):
    """Save inference metrics to a text file in the output directory."""
    metrics_str = (
        f"Test Metrics:\n"
        f"Precision: {avg_precision:.4f}\n"
        f"Recall: {avg_recall:.4f}\n"
        f"IoU: {avg_iou:.4f}\n"
        f"mAP: {test_mAP:.4f}\n"
        f"mAP@0.5: {test_mAP_50:.4f}\n"
    )
    output_path = os.path.join(output_dir, 'test_metrics.txt')
    with open(output_path, 'w') as f:
        f.write(metrics_str)
    return output_path

def plot_sample_predictions(model, images, gt_bboxes, loc_preds, cls_preds, original_sizes, output_dir, resized_dim=(112, 112), score_threshold=0.55, nms_iou_threshold=0.5):
    """Plot and save sample predictions with bounding boxes, aligned with training visualization."""
    cls_preds = torch.sigmoid(cls_preds)
    pred_boxes = model.decode_boxes(loc_preds, images.device)  # Decode raw predictions

    fig, axs = plt.subplots(1, min(8, images.size(0)), figsize=(20, 5))
    if images.size(0) == 1:
        axs = [axs]

    for i in range(min(8, images.size(0))):
        img = images[i].cpu().permute(1, 2, 0).numpy() * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        axs[i].imshow(img)

        original_dim = original_sizes[i]
        scale_x = resized_dim[1] / original_dim[0]
        scale_y = resized_dim[0] / original_dim[1]

        # Ground truth boxes (green)
        gt_boxes = gt_bboxes[i].cpu().numpy()
        for box in gt_boxes:
            x, y, w, h = box
            x, y, w, h = x * scale_x, y * scale_y, w * scale_x, h * scale_y
            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='g', facecolor='none', label='GT')
            axs[i].add_patch(rect)

        # Predicted boxes with NMS (red, score > threshold)
        pred_boxes_i = pred_boxes[i]  # [N, 4] (x_min, y_min, w, h)
        pred_scores_i = cls_preds[i, :, 1]  # [N]
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

    output_path = os.path.join(output_dir, 'test_sample_predictions.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    return output_path

def get_checkpoint_path(config, args_weights_path):
    """Determine the checkpoint path based on config, args, or default latest run."""
    inference_config = config.get("inference", {})
    weights_path = inference_config.get("weights_path") or args_weights_path

    if weights_path:
        if not os.path.isabs(weights_path):
            weights_path = os.path.join(config["paths"]["output_base_dir"], weights_path.lstrip("../"))
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Specified weights path not found: {weights_path}")
        return weights_path

    output_base_dir = config["paths"]["output_base_dir"]
    latest_run_file = os.path.join(output_base_dir, "latest_face_det_run.txt")
    if not os.path.exists(latest_run_file):
        raise FileNotFoundError("No training run found in latest_face_det_run.txt. Please run training first or specify a weights path.")
    with open(latest_run_file, "r") as f:
        run_dir = f.read().strip()
    checkpoint_path = os.path.join(run_dir, "checkpoints/best_model.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Best checkpoint not found at {checkpoint_path}")
    return checkpoint_path

def main(config, args_weights_path):
    # --------------- SET UP ENVIRONMENT --------------- #
    rank = GPUSetup.get_rank()
    local_rank = int(os.environ.get('LOCAL_RANK', 0)) if torch.cuda.is_available() else 0
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    log_info(f"Local Rank {local_rank}: Starting inference script logic")

    # ------------- LOAD MODEL AND CHECKPOINT ------------- #
    model_type = config["model"]["type"]
    num_anchors = config["model"]["num_anchors"]
    log_info(f"Loading {model_type} with {num_anchors} anchors")

    if model_type == "mobilefacenet":
        model = MobileFaceNet(weights=False, num_anchors=num_anchors).to(device)
    elif model_type == "yolotinynet":
        model = YOLOTinyFaceNet(num_anchors=num_anchors).to(device)
    elif model_type == "efficientfacenet":
        model = EfficientFaceNet(num_anchors=num_anchors).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    checkpoint_path = get_checkpoint_path(config, args_weights_path)
    state_dict = torch.load(checkpoint_path, map_location=device)
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()
    log_info(f"Loaded checkpoint: {checkpoint_path}")

    # ------------- PREPARE DATASET AND DATALOADER ------------- #
    dataset_dir = config["paths"]["dataset_dir"]
    test_dataset = FaceDataset(img_dir=dataset_dir, split='test')
    if len(test_dataset) == 0:
        raise ValueError("Test dataset is empty.")

    inference_config = config.get("inference", {})
    batch_size = inference_config.get("batch_size", config["training"]["batch_size"])
    num_workers = inference_config.get("num_workers", config["training"].get("num_workers", 4))
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,  # Set to True as in your latest version
        num_workers=num_workers,
        collate_fn=custom_collate_fn
    )
    log_info(f"Test dataset size: {len(test_dataset)} samples")

    # ------------- RUN INFERENCE ------------- #
    total_precision = 0.0
    total_recall = 0.0
    total_iou = 0.0
    test_samples_processed = 0
    metric = MeanAveragePrecision(box_format='xywh', iou_type='bbox')

    with torch.no_grad():
        for batch_idx, (images, gt_bboxes, original_sizes) in enumerate(test_loader):
            images = images.to(device)
            gt_bboxes = [gt_bbox.to(device) for gt_bbox in gt_bboxes]
            cls_preds, loc_preds = model(images)

            precision, recall, mean_iou = compute_metrics(model, cls_preds, loc_preds, gt_bboxes, original_sizes, device)
            total_precision += precision * images.size(0)
            total_recall += recall * images.size(0)
            total_iou += mean_iou * images.size(0)
            test_samples_processed += images.size(0)

            cls_preds_sigmoid = torch.sigmoid(cls_preds)
            preds = []
            targets = []
            resized_dim = (112, 112)
            for b in range(images.size(0)):
                pred_boxes = model.decode_boxes(loc_preds[b:b+1], device)[0]  # Decode for single image
                pred_scores = cls_preds_sigmoid[b, :, 1]
                gt_boxes = gt_bboxes[b]

                original_dim = original_sizes[b]
                scale_x = resized_dim[1] / original_dim[0]
                scale_y = resized_dim[0] / original_dim[1]

                pred_boxes_scaled = pred_boxes.clone()
                pred_boxes_scaled[:, 0] *= scale_x
                pred_boxes_scaled[:, 1] *= scale_y
                pred_boxes_scaled[:, 2] *= scale_x
                pred_boxes_scaled[:, 3] *= scale_y

                gt_boxes_scaled = gt_boxes.clone()
                gt_boxes_scaled[:, 0] *= scale_x
                gt_boxes_scaled[:, 1] *= scale_y
                gt_boxes_scaled[:, 2] *= scale_x
                gt_boxes_scaled[:, 3] *= scale_y

                mask = pred_scores > 0.5
                pred_boxes_scaled = pred_boxes_scaled[mask]
                pred_scores = pred_scores[mask]

                preds.append({
                    'boxes': pred_boxes_scaled,
                    'scores': pred_scores,
                    'labels': torch.ones(pred_scores.size(0), dtype=torch.int64, device=device)
                })
                targets.append({
                    'boxes': gt_boxes_scaled,
                    'labels': torch.ones(gt_boxes.size(0), dtype=torch.int64, device=device)
                })

            metric.update(preds, targets)

    # ------------- EVALUATE AND LOG RESULTS ------------- #
    if GPUSetup.is_main_process():
        avg_precision = total_precision / len(test_dataset)
        avg_recall = total_recall / len(test_dataset)
        avg_iou = total_iou / len(test_dataset)
        mAP_result = metric.compute()
        test_mAP = mAP_result['map'].item()
        test_mAP_50 = mAP_result['map_50'].item()

        metrics_str = f"Test Metrics - Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, IoU: {avg_iou:.4f}, mAP: {test_mAP:.4f}, mAP@0.5: {test_mAP_50:.4f}"
        log_info(metrics_str)
        print(metrics_str)

        inference_config = config.get("inference", {})
        output_dir = inference_config.get("inference_output_dir", config["paths"]["output_base_dir"])
        if not os.path.isabs(output_dir):
            config_dir = os.path.dirname(config_path)
            output_dir = os.path.join(config_dir, output_dir)
        os.makedirs(output_dir, exist_ok=True)

        # Save metrics to file
        metrics_path = save_metrics_to_file(output_dir, avg_precision, avg_recall, avg_iou, test_mAP, test_mAP_50)
        log_info(f"Saved metrics to {metrics_path}")

        # Generate and save sample predictions
        images, gt_bboxes, original_sizes = next(iter(test_loader))
        images = images.to(device)
        cls_preds, loc_preds = model(images)
        sample_path = plot_sample_predictions(model, images, gt_bboxes, loc_preds, cls_preds, original_sizes, output_dir)
        log_info(f"Saved sample predictions to {sample_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face detection inference script.")
    parser.add_argument(
        "--config_file",
        default="face_config.yaml",
        help="Name of the config YAML file in the ../config/ folder."
    )
    parser.add_argument(
        "--weights_path",
        default=None,
        help="Path to the model weights file (overrides config and latest_face_det_run.txt if provided)."
    )
    args = parser.parse_args()
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),  # Up three levels
        "config",
        args.config_file
    )

    try:
        config = load_config(config_path)
        logger = GPUSetup.setup_logging(
            config_level='INFO',
            logger=logging.getLogger(__name__),
            log_file=config["pipeline"]["log_file"]
        )
        GPUSetup.setup(distributed=False, seed=42)
        main(config, args.weights_path)
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise
    finally:
        if 'GPUSetup' in locals():
            GPUSetup.cleanup()
            log_info("Cleanup completed.")


# Default (Latest Run):
# python -m src.face_detection.infer_face --config_file face_config.yaml

# Custom Weights:
# python -m src.face_detection.infer_face --config_file face_config.yaml --weights_path "/path/to/weights.pt"