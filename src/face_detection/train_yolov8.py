
import os
import wandb
import argparse
import logging
from functools import wraps
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from ultralytics import YOLO

from src.utils import gpu_setup as GPUSetup
from src.utils import load_config

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

    logger.info(f"Local Rank {local_rank}: Starting YOLOv8 training script")

    if GPUSetup.is_distributed():
        if rank % ngpus_per_node == 0:
            print("Before DDP initialization:", flush=True)
            os.system("nvidia-smi")

    # ------------- SET UP EXPERIMENT RUN  ------------- #
    output_base_dir = config["paths"]["output_base_dir"]
    if GPUSetup.is_main_process():
        dyn_folder = f"yolov8_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_dir = os.path.join(output_base_dir, dyn_folder)
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(output_base_dir, "latest_yolov8_run.txt"), "w") as f:
            f.write(run_dir)
    else:
        run_dir = None

    if GPUSetup.is_distributed():
        run_dir_tensor = torch.tensor([0], dtype=torch.int).cuda(device) if run_dir is None else torch.tensor([1], dtype=torch.int).cuda(device)
        dist.broadcast(run_dir_tensor, src=0)
        if run_dir_tensor.item() == 1:
            with open(os.path.join(output_base_dir, "latest_yolov8_run.txt"), "r") as f:
                run_dir = f.read().strip()
        dist.barrier()

    # ------------- LOAD DATASETS  ------------- #
    log_info("Preparing YOLOv8 face detection datasets")
    dataset_dir = config["paths"]["dataset_yolov8_dir"]
    logger.info(f"Dataset directory: {dataset_dir}")
    if dataset_dir is None:
        logger.error("Dataset directory is None. Please set 'dataset_yolov8_dir' in yolov8_config.yaml.")
        raise ValueError("Dataset directory is None. Please set 'dataset_yolov8_dir' in yolov8_config.yaml.")
    data_yaml = os.path.join(dataset_dir, 'data.yaml')
    logger.info(f"Data YAML path: {data_yaml}")
    if not os.path.exists(data_yaml):
        logger.error(f"Data YAML file does not exist at: {data_yaml}")
        raise FileNotFoundError(f"Data YAML file not found at: {data_yaml}")

    # Ensure imgsz is an integer
    imgsz = config["model"]["imgsz"]
    if isinstance(imgsz, list):
        imgsz = max(imgsz)  # Use the larger dimension if imgsz is a list
        logger.warning(f"imgsz was provided as a list {config['model']['imgsz']}. Using the larger dimension: {imgsz}")

    # ------------- INITIALIZE MODEL  ------------- #
    model_type = config["model"]["type"]
    log_info(f"Preparing {model_type} on {device}")

    # import pdb; pdb.set_trace()
    model = YOLO(model_type)

    # ------------- TRAINING  ------------- #
    logger.info(f"Local Rank {local_rank}: Starting YOLOv8 training phase...")
    max_epochs = config["training"]["epochs"]

    # Define overrides with custom_train_loader and custom_val_loader
    print(f'data_yaml: {data_yaml}')
    overrides = {
        "model": model_type,
        "data": data_yaml,
        "epochs": max_epochs,
        "batch": config["training"]["batch_size"],
        "imgsz": imgsz,
        "rect": config["model"]["rect"],
        "device": device,
        "workers": config["training"].get("num_workers", 4),
        "project": "face-detection-yolov8", #run_dir,
        "name": dyn_folder, 
        "verbose": True,
        "save_period": 5,
        "exist_ok": True,
        "augment": config.get("augment", True),
        # YOLOv8 Augmentation Configuration
        "hsv_h": config.get("hsv_h", 0.03),  # Increased hue augmentation for more color variation
        "hsv_s": config.get("hsv_s", 0.7),   # Keep saturation (already good)
        "hsv_v": config.get("hsv_v", 0.4),   # Keep value (brightness)
        "degrees": config.get("degrees", 15.0),  # Enable rotation (+/- 15 degrees) for varied object angles
        "translate": config.get("translate", 0.2),  # Increased translation for more positional variation
        "scale": config.get("scale", 0.7),   # Increased scaling for more size variation
        "shear": config.get("shear", 5.0),   # Add shear for slight distortions
        "perspective": config.get("perspective", 0.0001),  # Add subtle perspective transforms
        "flipud": config.get("flipud", 0.5),  # Enable vertical flip (50% probability)
        "fliplr": config.get("fliplr", 0.5),  # Keep horizontal flip (50% probability)
        "mosaic": config.get("mosaic", 0.8),  # Enable mosaic with 80% probability (key for object density)
        "mixup": config.get("mixup", 0.1),   # Enable mixup with 10% probability (blends images)
        "copy_paste": config.get("copy_paste", 0.1),  # Enable copy-paste with 10% probability (pastes objects)
        "auto_augment": config.get("auto_augment", "randaugment"),  # Keep randaugment
        "erasing": config.get("erasing", 0.2),  # Reduced random erasing to avoid obscuring small objects
        "crop_fraction": config.get("crop_fraction", 1.0),  # Keep no cropping

        # Other possible ultralytics yolo updates:
        # "optimizer": "Adam",
        # "lr0": config["training"]["learning_rate"],
        # "patience": config["training"]["early_stopping"]["patience"],
        # "cos_lr": True,
        # "run_dir": output_base_dir,
    }

    # Debug the overrides dictionary
    logger.info(f"Overrides dictionary before model.train(): {overrides}")

    # Pass CustomDetectionTrainer as a class, not an instance
    results = model.train(
        **overrides
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 face detection training script.")
    parser.add_argument(
        "--config_file",
        default="yolov8_config.yaml",
        help="Name of the config YAML file in the ../config/ folder."
    )
    args = parser.parse_args()
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "config",
        args.config_file
    )

    try:
        config = load_config(config_path)
        logger = GPUSetup.setup_logging(
            config_level='DEBUG',
            logger=logging.getLogger(__name__),
            log_file=config["paths"]["log_file"]
        )
        GPUSetup.setup(distributed=False, seed=42)
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

# ------------------------------------------------------------------------
# python -m src.face_detection.train_yolov8
# python -m src.face_detection.train_yolov8 --config_file yolov8_config.yaml