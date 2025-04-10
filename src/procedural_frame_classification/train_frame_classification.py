
import os
import wandb
import argparse
import logging
import numpy as np
from pathlib import Path
from functools import wraps
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.metrics import confusion_matrix, f1_score

from src.procedural_frame_classification import (
    FrameDataset,
    compute_class_distribution,
    get_model,
)
from src.utils import gpu_setup as GPUSetup
from src.utils import load_config

# Retrieve a logger for the module
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

    # Detect if we have a GPU available and choose device accordingly
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
    # Create a new run folder (only on main process)
    output_base_dir = config["paths"]["output_base_dir"]
    if GPUSetup.is_main_process():
        run_dir = os.path.join(output_base_dir, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        # Save the run directory path to a file for inference to use
        with open(os.path.join(output_base_dir, "latest_run.txt"), "w") as f:
            f.write(run_dir)
    else:
        run_dir = None

    # Synchronize run_dir across all processes
    if GPUSetup.is_distributed():
        run_dir_tensor = torch.tensor([0], dtype=torch.int).cuda(device) if run_dir is None else torch.tensor([1], dtype=torch.int).cuda(device)
        dist.broadcast(run_dir_tensor, src=0)
        if run_dir_tensor.item() == 1:
            with open(os.path.join(output_base_dir, "latest_run.txt"), "r") as f:
                run_dir = f.read().strip()

    # Initialize WandB
    if GPUSetup.is_distributed():
        if GPUSetup.is_main_process(): 
            wandb.init(project="frame-classification", config={
                "epochs": config["training"]["epochs"],
                "batch_size": 32,
                "learning_rate": config["training"]["learning_rate"],
            })

    # Load datasets based on explicit splits
    log_info(f"Preparing datasets")
    dataset_dir = config["paths"]["dataset_dir"]
    train_dataset = FrameDataset(img_dir=dataset_dir, split='train')
    val_dataset = FrameDataset(img_dir=dataset_dir, split='val')

    # Distributed Sampler for DDP
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    num_workers = 4
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=train_sampler, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=num_workers)

    # Log dataset sizes and class distribution
    class_names = ['Non-Procedural', 'Procedural']
    if GPUSetup.is_main_process():
        logger.info(f"Training dataset size: {len(train_dataset)} samples")
        logger.info(f"Validation dataset size: {len(val_dataset)} samples")

        # Compute and log class distribution
        train_class_counts, train_class_ratios = compute_class_distribution(train_dataset, class_names)
        val_class_counts, val_class_ratios = compute_class_distribution(val_dataset, class_names)
        
        logger.info("Training class distribution (counts):")
        logger.info(str(train_class_counts))
        logger.info("Training class distribution (ratios):")
        logger.info(str(train_class_ratios))
        logger.info("Validation class distribution (counts):")
        logger.info(str(val_class_counts))
        logger.info("Validation class distribution (ratios):")
        logger.info(str(val_class_ratios))

    # Initialize model and move to GPU
    log_info(f"Preparing training base on {device}")
    model = get_model().cuda(device)

    if GPUSetup.is_distributed():
        # Convert all BatchNorm layers to SyncBatchNorm layers
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], 
                                                         broadcast_buffers=True, find_unused_parameters=True)
        torch.backends.cudnn.benchmark = True

    if GPUSetup.is_distributed():
        if rank % ngpus_per_node == 0:
            print("After DDP initialization:", flush=True)
            os.system("nvidia-smi")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

    # Learning rate scheduler
    scheduler_config = config["training"].get("lr_scheduler", {})
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',  # Maximize the F1-score
        factor=scheduler_config.get("factor", 0.1),
        patience=scheduler_config.get("patience", 3),
        min_lr=scheduler_config.get("min_lr", 1e-6),
        # verbose=True
    )
    # Early stopping parameters
    early_stopping_config = config["training"].get("early_stopping", {})
    patience = early_stopping_config.get("patience", 5)
    min_delta = float(early_stopping_config.get("min_delta", 0.001))  # Convert to float

    monitor = early_stopping_config.get("monitor", "val_f1")
    
    logger.info(f"Early stopping monitor: {monitor} (type: {type(monitor)})")
    if monitor not in ["val_f1", "train_loss"]:
        raise ValueError(f"Invalid monitor value: {monitor}. Must be 'val_f1' or 'train_loss'.")
    best_metric = float('-inf') if monitor == "val_f1" else float('inf')
    epochs_no_improve = 0
    early_stop = False

    logger.info(f"Local Rank {local_rank}: Starting training phase...")
    max_epochs = config["training"]["epochs"]
    for epoch in range(max_epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        total_loss = 0.0
        train_samples_processed = 0

        for images, labels in train_loader:
            images, labels = images.cuda(device), labels.cuda(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            train_samples_processed += images.size(0)

        avg_loss = total_loss / len(train_loader)

        if GPUSetup.is_main_process():
            logger.info(f"Epoch {epoch+1}: Processed {train_samples_processed} training samples (total dataset size: {len(train_dataset)})")
            print(f"Epoch {epoch+1}: Processed {train_samples_processed} training samples (total dataset size: {len(train_dataset)})")

        # Validation loop
        if GPUSetup.is_main_process(): 
            logger.info(f"Local Rank {local_rank}: Starting validation phase...")
            model.eval()

            all_preds = []
            all_labels = []
            val_samples_processed = 0

            with torch.no_grad():
                for batch_idx, (images, labels) in enumerate(val_loader):
                    images, labels = images.cuda(device), labels.cuda(device)
                    outputs = model(images)
                    _, preds = torch.max(outputs, 1)

                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    val_samples_processed += images.size(0)

                    # Debugging: print labels and predictions for the first batch only
                    if batch_idx == 0:
                        print("True labels:", labels.cpu().numpy())
                        print("Predictions:", preds.cpu().numpy())

            logger.info(f"Epoch {epoch+1}: Processed {val_samples_processed} validation samples (total dataset size: {len(val_dataset)})")
            print(f"Epoch {epoch+1}: Processed {val_samples_processed} validation samples (total dataset size: {len(val_dataset)})")

            # Compute accuracy
            correct = np.sum(np.array(all_preds) == np.array(all_labels))
            val_accuracy = correct / len(all_labels)

            # Compute macro-averaged F1-score
            val_f1 = f1_score(all_labels, all_preds, average='macro')

            # Compute confusion matrix
            cm = confusion_matrix(all_labels, all_preds)

            print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, Validation F1-Score: {val_f1:.4f}')
            print("Confusion Matrix:\n", cm)
            logger.info(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, Validation F1-Score: {val_f1:.4f}')
            logger.info(f"Confusion Matrix:\n{cm}")

            # Log current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"Epoch {epoch+1}: Current learning rate: {current_lr}")
            print(f"Epoch {epoch+1}: Current learning rate: {current_lr}")

            # WandB Logging
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': avg_loss,
                'validation_accuracy': val_accuracy,
                'validation_f1': val_f1,
                'learning_rate': current_lr,
                'confusion_matrix': wandb.plot.confusion_matrix(
                    preds=all_preds,
                    y_true=all_labels,
                    class_names=['Non-Procedural', 'Procedural']
                )
            })

            # Save checkpoint for this epoch
            checkpoint_path = os.path.join(run_dir, f"checkpoints/model_epoch_{epoch+1}.pt")
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")

            # Early stopping logic (using F1-score)
            if monitor == "val_f1":
                current_metric = val_f1
                logger.info(f"Early stopping: monitor={monitor}, current_metric={current_metric} (type: {type(current_metric)}), best_metric={best_metric} (type: {type(best_metric)}), min_delta={min_delta} (type: {type(min_delta)})")
                improved = current_metric > best_metric + min_delta
            elif monitor == "train_loss":
                current_metric = avg_loss
                logger.info(f"Early stopping: monitor={monitor}, current_metric={current_metric} (type: {type(current_metric)}), best_metric={best_metric} (type: {type(best_metric)}), min_delta={min_delta} (type: {type(min_delta)})")
                improved = current_metric < best_metric - min_delta
            else:
                raise ValueError(f"Unsupported monitor metric: {monitor}")

            # Update learning rate scheduler
            scheduler.step(current_metric)

            if improved:
                best_metric = current_metric
                epochs_no_improve = 0
                # Save the best checkpoint
                best_checkpoint_path = os.path.join(run_dir, "checkpoints/best_model.pt")
                torch.save(model.state_dict(), best_checkpoint_path)
                logger.info(f"Saved best checkpoint (based on {monitor}): {best_checkpoint_path}")
            else:
                epochs_no_improve += 1
                logger.info(f"No improvement in {monitor} for {epochs_no_improve} epochs (best: {best_metric:.4f}, current: {current_metric:.4f})")

            if epochs_no_improve >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs (no improvement in {monitor} for {patience} epochs)")
                print(f"Early stopping triggered after {epoch+1} epochs")
                early_stop = True

        # Synchronize early stopping decision across all processes
        if GPUSetup.is_distributed():
            early_stop_tensor = torch.tensor([1 if early_stop else 0], dtype=torch.int).cuda(device)
            dist.broadcast(early_stop_tensor, src=0)
            early_stop = early_stop_tensor.item() == 1

        if early_stop:
            break

    if GPUSetup.is_main_process(): 
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Frame classification training script.")
    parser.add_argument(
        "--config_file",
        default="config.yaml",
        help="Name of the config YAML file in the ../config/ folder."
    )
    args = parser.parse_args()
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),  # up one additional level
        "config",
        args.config_file
    )

    try:
        # Load config and set up logging
        config = load_config(config_path)
        logger = GPUSetup.setup_logging(
            config_level='DEBUG', #'INFO',
            logger=logging.getLogger(__name__),
            log_file=config["pipeline"]["log_file"])

        # Setup environment for distributed training
        GPUSetup.setup(distributed=True, seed=42)
        main(config)
            
    except Exception as e:
        if logger is not None:
            logger.error(f"An error occurred: {e}")
        else:
            print(f"An error occurred before logger was set up: {e}")
    finally:
        if 'GPUSetup' in locals():
            GPUSetup.cleanup()
            log_info("Cleanup completed.")


# python -m src.procedural_frame_classification.train_frame_classification
# python -m src.procedural_frame_classification.train_frame_classification --config_file config.yaml