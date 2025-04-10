
import os
import argparse
import logging
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.procedural_frame_classification import FrameDataset, get_model
from src.utils import gpu_setup as GPUSetup
from src.utils import load_config

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

def plot_confusion_matrix(cm, output_dir, class_names):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Test Set Confusion Matrix')
    output_path = os.path.join(output_dir, 'test_confusion_matrix.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    return output_path

def plot_sample_predictions(images, labels, preds, output_dir, class_names):
    """Plot and save sample predictions for a batch."""
    fig, axs = plt.subplots(1, 8, figsize=(20, 5))
    for i in range(min(8, len(images))):
        img = images[i].cpu().permute(1, 2, 0).numpy()
        axs[i].imshow(img)
        axs[i].set_title(f'True: {class_names[labels[i]]}\nPred: {class_names[preds[i]]}')
        axs[i].axis('off')
    output_path = os.path.join(output_dir, 'test_sample_predictions.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    return output_path

def get_checkpoint_path(config, args_weights_path):
    """Determine the checkpoint path based on config, args, or default latest run."""
    inference_config = config.get("inference", {})
    weights_path = inference_config.get("weights_path") or args_weights_path

    if weights_path:
        # If not an absolute path, make it relative to output_base_dir
        if not os.path.isabs(weights_path):
            weights_path = os.path.join(config["paths"]["output_base_dir"], weights_path.lstrip("../"))
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Specified weights path not found: {weights_path}")
        return weights_path, True  # Indicate a specific weights path was provided

    # Fallback to latest run if no weights path is provided
    output_base_dir = config["paths"]["output_base_dir"]
    latest_run_file = os.path.join(output_base_dir, "latest_run.txt")
    if not os.path.exists(latest_run_file):
        raise FileNotFoundError("No training run found in latest_run.txt. Please run training first or specify a weights path.")
    with open(latest_run_file, "r") as f:
        run_dir = f.read().strip()
    checkpoint_path = os.path.join(run_dir, "checkpoints/best_model.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Best checkpoint not found at {checkpoint_path}")
    return checkpoint_path, False  # Indicate default path was used

def main(config, args_weights_path, config_path):
    # --------------- SET UP ENVIRONMENT --------------- #
    rank = GPUSetup.get_rank()
    local_rank = int(os.environ.get('LOCAL_RANK', 0)) if torch.cuda.is_available() else 0
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    log_info(f"Local Rank {local_rank}: Starting inference script logic")

    # ------------- LOAD MODEL AND CHECKPOINT ------------- #
    checkpoint_path, is_specific_weights = get_checkpoint_path(config, args_weights_path)
    model = get_model().to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    # Handle potential DDP prefix from training
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()
    log_info(f"Loaded checkpoint: {checkpoint_path}")

    # ------------- PREPARE DATASET AND DATALOADER ------------- #
    transform_config = config.get("inference", {}).get("transforms", {})
    transform = transforms.Compose([
        transforms.Resize((transform_config.get("resize", 256), transform_config.get("resize", 256))),
        transforms.CenterCrop(transform_config.get("crop", 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset_dir = config["paths"]["dataset_dir"]
    test_dataset = FrameDataset(img_dir=dataset_dir, split='test')  #, transform=transform)
    batch_size = config.get("inference", {}).get("batch_size", 32)
    num_workers = config.get("inference", {}).get("num_workers", 4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    log_info(f"Test dataset size: {len(test_dataset)} samples")

    # ------------- RUN INFERENCE ------------- #
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # ------------- EVALUATE AND LOG RESULTS ------------- #
    if GPUSetup.is_main_process():
        class_names = ['Non-Procedural', 'Procedural']
        report = classification_report(all_labels, all_preds, target_names=class_names)
        print("Classification Report:")
        print(report)
        log_info("Classification Report:\n" + report)

        # # Determine output directory based on whether a specific weights path was provided
        # if is_specific_weights:
        #     # Use inference_output_dir from config
        #     inference_config = config.get("inference", {})
        #     output_dir = inference_config.get("inference_output_dir", config["paths"]["output_base_dir"])
        #     if not os.path.isabs(output_dir):
        #         # Resolve relative to config file's parent directory
        #         config_dir = os.path.dirname(config_path)
        #         output_dir = os.path.join(config_dir, output_dir)
        # else:
        #     # Use the run directory from latest_run.txt
        #     output_base_dir = config["paths"]["output_base_dir"]
        #     latest_run_file = os.path.join(output_base_dir, "latest_run.txt")
        #     with open(latest_run_file, "r") as f:
        #         run_dir = f.read().strip()
        #     output_dir = run_dir

        # Determine output directory based on whether a specific weights path was provided
        if is_specific_weights:
            inference_config = config.get("inference", {})
            output_dir = inference_config.get("inference_output_dir", config["paths"]["output_base_dir"])
        else:
            output_base_dir = config["paths"]["output_base_dir"]
            latest_run_file = os.path.join(output_base_dir, "latest_run.txt")
            with open(latest_run_file, "r") as f:
                run_dir = f.read().strip()
            output_dir = run_dir

        # Create 'figures' subfolder in the output directory
        figures_dir = os.path.join(output_dir, "figures")
        os.makedirs(figures_dir, exist_ok=True)

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        cm_path = plot_confusion_matrix(cm, figures_dir, class_names)
        log_info(f"Saved confusion matrix to {cm_path}")

        # Sample predictions visualization
        images, labels = next(iter(test_loader))
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        sample_path = plot_sample_predictions(images, labels, preds, figures_dir, class_names)
        log_info(f"Saved sample predictions to {sample_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Frame classification inference script.")
    parser.add_argument(
        "--config_file",
        default="config.yaml",
        help="Name of the config YAML file in the ../config/ folder."
    )
    parser.add_argument(
        "--weights_path",
        default=None,
        help="Path to the model weights file (overrides config and latest_run.txt if provided)."
    )
    args = parser.parse_args()
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),  # Up three levels
        "config",
        args.config_file
    )

    try:
        # Load config and set up logging
        config = load_config(config_path)
        logger = GPUSetup.setup_logging(
            config_level='INFO',
            logger=logging.getLogger(__name__),
            log_file=config["pipeline"]["log_file"]
        )

        # Setup environment (non-distributed by default, but compatible if needed)
        GPUSetup.setup(distributed=False, seed=42)
        main(config, args.weights_path, config_path)

    except Exception as e:
        if logger is not None:
            logger.error(f"An error occurred: {e}")
        else:
            print(f"An error occurred before logger was set up: {e}")
    finally:
        if 'GPUSetup' in locals():
            GPUSetup.cleanup()
            log_info("Cleanup completed.")

# python -m src.procedural_frame_classification.infer_frame
# python -m src.procedural_frame_classification.infer_frame --config_file config.yaml

# Custom Relative weights_path:
# python -m src.procedural_frame_classification.infer_frame --weights_path "../custom_weights/model.pt"
# weights_path: /data/mskscratch/users/ghoyer/Precision_Air/VLS_auto_trimmer/outputs/custom_weights/model.pt

# Custom Absolute weights_path:
# python -m src.procedural_frame_classification.infer_frame --weights_path "/absolute/path/model.pt"
# weights_path: /absolute/path/model.pt

#--------------------------------------------------------
