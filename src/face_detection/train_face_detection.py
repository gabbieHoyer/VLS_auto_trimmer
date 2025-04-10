
import os
import wandb
import argparse
import logging
from functools import wraps
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

from src.utils import gpu_setup as GPUSetup
from src.utils import load_config, compute_loss, compute_metrics, visualize_bboxes

# Import face-related dataset/models from face_detection
from src.face_detection import (
    FaceDataset,
    custom_collate_fn,
    MobileFaceNet,
    EfficientFaceNet,
    YOLOTinyFaceNet,
)

logger = logging.getLogger(__name__)

def main_process_only(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if GPUSetup.is_main_process():
            return func(*args, **kwargs)
    return wrapper

@main_process_only
def log_info(message):
    logger.info(message)


def main(config):
    # --------------- SET UP ENVIRONMENT --------------- #
    rank = GPUSetup.get_rank()
    ngpus_per_node = torch.cuda.device_count()

    if torch.cuda.is_available():
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cpu')

    logger.info(f"Local Rank {local_rank}: Starting main script logic")

    if GPUSetup.is_distributed():
        if rank % ngpus_per_node == 0:
            print("Before DDP initialization:", flush=True)
            os.system("nvidia-smi")

    # ------------- SET UP EXPERIMENT RUN  ------------- #
    output_base_dir = config["paths"]["output_base_dir"]
    if GPUSetup.is_main_process():
        run_dir = os.path.join(output_base_dir, f"face_det_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(output_base_dir, "latest_face_det_run.txt"), "w") as f:
            f.write(run_dir)
    else:
        run_dir = None

    if GPUSetup.is_distributed():
        run_dir_tensor = torch.tensor([0], dtype=torch.int).cuda(device) if run_dir is None else torch.tensor([1], dtype=torch.int).cuda(device)
        dist.broadcast(run_dir_tensor, src=0)
        if run_dir_tensor.item() == 1:
            with open(os.path.join(output_base_dir, "latest_face_det_run.txt"), "r") as f:
                run_dir = f.read().strip()
        dist.barrier()  # Ensure all processes have run_dir before proceeding

    if GPUSetup.is_distributed() and GPUSetup.is_main_process():
        wandb.init(project="face-detection", config={
            "epochs": config["training"]["epochs"],
            "batch_size": config["training"]["batch_size"],
            "learning_rate": config["training"]["learning_rate"],
            "model_type": config["model"]["type"],
            "num_anchors": config["model"]["num_anchors"],
        })

    # ------------- LOAD DATASETS  ------------- #
    log_info("Preparing face detection datasets")
    dataset_dir = config["paths"]["dataset_dir"]
    train_dataset = FaceDataset(img_dir=dataset_dir, split='train')
    val_dataset = FaceDataset(img_dir=dataset_dir, split='val')

    if len(train_dataset) == 0:
        logger.error("Training dataset is empty. Cannot proceed with training.")
        raise ValueError("Training dataset is empty.")
    if len(val_dataset) == 0:
        logger.warning("Validation dataset is empty. Training will proceed without validation.")

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if GPUSetup.is_distributed() else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=config["training"].get("num_workers", 4),
        collate_fn=custom_collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"].get("num_workers", 4),
        collate_fn=custom_collate_fn
    )

    if GPUSetup.is_main_process():
        logger.info(f"Training dataset size: {len(train_dataset)} samples")
        logger.info(f"Validation dataset size: {len(val_dataset)} samples")

    # ------------- INITIALIZE MODEL  ------------- #

    # Initialize model based on config
    model_type = config["model"]["type"]
    num_anchors = config["model"]["num_anchors"]
    log_info(f"Preparing {model_type} on {device}")
    if model_type == "mobilefacenet":
        model = MobileFaceNet(weights=True, num_anchors=num_anchors).to(device)
    elif model_type == "yolotinynet":
        model = YOLOTinyFaceNet(num_anchors=num_anchors).to(device)
    elif model_type == "efficientfacenet":
        model = EfficientFaceNet(num_anchors=num_anchors).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if GPUSetup.is_distributed():
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[local_rank], broadcast_buffers=True, find_unused_parameters=True)
        torch.backends.cudnn.benchmark = True

    if GPUSetup.is_distributed() and rank % ngpus_per_node == 0:
        print("After DDP initialization:", flush=True)
        os.system("nvidia-smi")

    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, min_lr=1e-6)

    # Early stopping
    early_stopping_config = config["training"].get("early_stopping", {})
    patience = early_stopping_config.get("patience", 5)
    min_delta = float(early_stopping_config.get("min_delta", 0.001))
    best_loss = float('inf')
    epochs_no_improve = 0
    early_stop = False

    # ------------- TRAINING LOOP  ------------- #
    logger.info(f"Local Rank {local_rank}: Starting training phase...")
    max_epochs = config["training"]["epochs"]

    for epoch in range(max_epochs):
        model.train()
        if GPUSetup.is_distributed():
            train_sampler.set_epoch(epoch)

        total_loss = 0.0
        total_precision = 0.0
        total_recall = 0.0
        total_iou = 0.0
        train_samples_processed = 0
        visualized_train = False

        for batch_idx, (images, gt_bboxes, original_sizes) in enumerate(train_loader): 
        # for images, bboxes_list, original_sizes in train_loader:
            images = images.to(device)
            gt_bboxes = [gt_bbox.to(device) for gt_bbox in gt_bboxes]
            optimizer.zero_grad()
            cls_preds, loc_preds = model(images)
            #loss = compute_loss(model.module if GPUSetup.is_distributed() else model, cls_preds, loc_preds, bboxes_list, original_sizes, device)
            loss = compute_loss(model.module if GPUSetup.is_distributed() else model, cls_preds, loc_preds, gt_bboxes, original_sizes, device)
            # loss.backward()
            if loss != 0:  # Only call backward if loss is non-zero
                loss.backward()
                optimizer.step()
            else:
                logger.warning(f"Skipping backward step: loss is 0 for batch {batch_idx}")
                    
            optimizer.step()

            #precision, recall, mean_iou = compute_metrics(model.module if GPUSetup.is_distributed() else model, cls_preds, loc_preds, bboxes_list, original_sizes, device)
            precision, recall, mean_iou = compute_metrics(model.module if GPUSetup.is_distributed() else model, cls_preds, loc_preds, gt_bboxes, original_sizes, device)

            total_loss += loss.item() * images.size(0)
            total_precision += precision * images.size(0)
            total_recall += recall * images.size(0)
            total_iou += mean_iou * images.size(0)
            train_samples_processed += images.size(0)

            if epoch < 5 and not visualized_train and GPUSetup.is_main_process():
                output_dir = os.path.join(run_dir, "figures")
                os.makedirs(output_dir, exist_ok=True)
                visualize_bboxes(model.module if GPUSetup.is_distributed() else model, images, gt_bboxes, loc_preds, cls_preds, original_sizes, output_dir, epoch, prefix="train")
                visualized_train = True

        avg_loss = total_loss / len(train_dataset) if len(train_dataset) > 0 else 0.0
        avg_precision = total_precision / len(train_dataset) if len(train_dataset) > 0 else 0.0
        avg_recall = total_recall / len(train_dataset) if len(train_dataset) > 0 else 0.0
        avg_iou = total_iou / len(train_dataset) if len(train_dataset) > 0 else 0.0

        if GPUSetup.is_main_process():
            logger.info(f"Epoch {epoch+1}: Loss: {avg_loss:.4f}, Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, IoU: {avg_iou:.4f}")
            print(f"Epoch {epoch+1}: Loss: {avg_loss:.4f}, Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, IoU: {avg_iou:.4f}")
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': avg_loss,
                'train_precision': avg_precision,
                'train_recall': avg_recall,
                'train_iou': avg_iou,
                'learning_rate': optimizer.param_groups[0]['lr'],
            })

        # Validation loop
        if GPUSetup.is_main_process():
            model.eval()
            val_loss = 0.0
            val_precision = 0.0
            val_recall = 0.0
            val_iou = 0.0
            val_samples_processed = 0
            visualized_val = False

            with torch.no_grad():
                for images, bboxes_list, original_sizes in val_loader:
                    images = images.to(device)
                    cls_preds, loc_preds = model(images)
                    loss = compute_loss(model.module if GPUSetup.is_distributed() else model, cls_preds, loc_preds, bboxes_list, original_sizes, device)
                    precision, recall, mean_iou = compute_metrics(model.module if GPUSetup.is_distributed() else model, cls_preds, loc_preds, bboxes_list, original_sizes, device)
                    val_loss += loss.item() * images.size(0)
                    val_precision += precision * images.size(0)
                    val_recall += recall * images.size(0)
                    val_iou += mean_iou * images.size(0)
                    val_samples_processed += images.size(0)

                    if epoch < 5 and not visualized_val:
                        output_dir = os.path.join(run_dir, "figures")
                        os.makedirs(output_dir, exist_ok=True)
                        visualize_bboxes(model.module if GPUSetup.is_distributed() else model, images, bboxes_list, loc_preds, cls_preds, original_sizes, output_dir, epoch, prefix="val")
                        visualized_val = True

            avg_val_loss = val_loss / len(val_dataset) if len(val_dataset) > 0 else 0.0
            avg_val_precision = val_precision / len(val_dataset) if len(val_dataset) > 0 else 0.0
            avg_val_recall = val_recall / len(val_dataset) if len(val_dataset) > 0 else 0.0
            avg_val_iou = val_iou / len(val_dataset) if len(val_dataset) > 0 else 0.0

            logger.info(f"Epoch {epoch+1}: Val Loss: {avg_val_loss:.4f}, Val Precision: {avg_val_precision:.4f}, Val Recall: {avg_val_recall:.4f}, Val IoU: {avg_val_iou:.4f}")
            print(f"Epoch {epoch+1}: Val Loss: {avg_val_loss:.4f}, Val Precision: {avg_val_precision:.4f}, Val Recall: {avg_val_recall:.4f}, Val IoU: {avg_val_iou:.4f}")
            wandb.log({
                'val_loss': avg_val_loss,
                'val_precision': avg_val_precision,
                'val_recall': avg_val_recall,
                'val_iou': avg_val_iou,
            })

            # Save checkpoints
            if (epoch + 1) % 5 == 0:
                checkpoint_path = os.path.join(run_dir, f"checkpoints/model_epoch_{epoch+1}.pt")
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                torch.save(model.module.state_dict() if GPUSetup.is_distributed() else model.state_dict(), checkpoint_path)
                logger.info(f"Saved checkpoint: {checkpoint_path}")
                if GPUSetup.is_distributed():
                    dist.barrier()  # Ensure all processes wait until the checkpoint is saved

            if avg_val_loss < best_loss - min_delta:
                best_loss = avg_val_loss
                epochs_no_improve = 0
                best_checkpoint_path = os.path.join(run_dir, "checkpoints/best_model.pt")
                os.makedirs(os.path.dirname(best_checkpoint_path), exist_ok=True)
                torch.save(model.module.state_dict() if GPUSetup.is_distributed() else model.state_dict(), best_checkpoint_path)
                logger.info(f"Saved best checkpoint: {best_checkpoint_path}")
                if GPUSetup.is_distributed():
                    dist.barrier()  # Ensure all processes wait until the checkpoint is saved
            else:
                epochs_no_improve += 1
                logger.info(f"No improvement in val_loss for {epochs_no_improve} epochs")
                
            scheduler.step(avg_val_loss)

            if epochs_no_improve >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                early_stop = True

        if GPUSetup.is_distributed():
            early_stop_tensor = torch.tensor([1 if early_stop else 0], dtype=torch.int).to(device)
            dist.broadcast(early_stop_tensor, src=0)
            early_stop = early_stop_tensor.item() == 1

        if early_stop:
            break

    if GPUSetup.is_main_process():
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face/person detection training script.")
    parser.add_argument(
        "--config_file",
        default="face_config.yaml",
        help="Name of the config YAML file in the ../config/ folder."
    )
    args = parser.parse_args()
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),  # up one additional level
        "config",
        args.config_file
    )

    try:
        config = load_config(config_path)
        logger = GPUSetup.setup_logging(
            config_level='DEBUG',
            logger=logging.getLogger(__name__),
            log_file=config["pipeline"]["log_file"]
        )
        GPUSetup.setup(distributed=True, seed=42)
        main(config)
    except Exception as e:
        import traceback
        if logger is not None:
            logger.error(f"An error occurred: {e}\nStack trace:\n{traceback.format_exc()}")
        else:
            print(f"An error occurred before logger was set up: {e}\nStack trace:\n{traceback.format_exc()}")
        raise e
    finally:
        if 'GPUSetup' in locals():
            GPUSetup.cleanup()
            log_info("Cleanup completed.")


# python -m src.face_detection.train_face_detection
# python -m src.face_detection.train_face_detection --config_file face_config.yaml