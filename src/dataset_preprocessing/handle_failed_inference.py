import os
import glob
import shutil
import logging
from pathlib import Path

# from utils.config_loader 
from src.utils import load_config, setup_logging

def reintegrate_failed_inference(config):
    """Move failed inference videos back to new_data for reprocessing."""
    failed_dir = config["paths"]["failed_inference_dir"]
    new_data_dir = config["paths"]["new_data_dir"]

    for class_name in ["procedural", "nonprocedural"]:
        failed_class_dir = os.path.join(failed_dir, class_name)
        new_data_class_dir = os.path.join(new_data_dir, class_name)

        if not os.path.exists(failed_class_dir):
            continue

        Path(new_data_class_dir).mkdir(parents=True, exist_ok=True)

        # Move failed videos to new_data
        for video_file in glob.glob(f"{failed_class_dir}/*.mp4") + glob.glob(f"{failed_class_dir}/*.avi"):
            src = video_file
            dst = os.path.join(new_data_class_dir, os.path.basename(video_file))
            shutil.move(src, dst)
            logging.info(f"Moved {os.path.basename(src)} to {new_data_class_dir}")
            print(f"Moved {os.path.basename(src)} to {new_data_class_dir}")

if __name__ == "__main__":
    # Load config and set up logging
    config = load_config("config.yaml")
    setup_logging(config["pipeline"]["log_file"])

    reintegrate_failed_inference(config)
    logging.info("Failed inference data moved to new_data for reprocessing.")
    print("Failed inference data moved to new_data for reprocessing.")

