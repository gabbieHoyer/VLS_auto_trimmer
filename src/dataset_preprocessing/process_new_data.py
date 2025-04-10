
import os
import glob
import shutil
import argparse
import logging
from pathlib import Path

from src.dataset_preprocessing import extract_frames, split_frames
from src.utils import load_config, setup_logging

logger = logging.getLogger(__name__)

def count_frames_in_dataset(dataset_dir, class_name):
    """Count the number of frames in each split for a given class."""
    counts = {}
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(dataset_dir, split, class_name)
        if os.path.exists(split_dir):
            counts[split] = len(glob.glob(os.path.join(split_dir, "*.jpeg")))
        else:
            counts[split] = 0
    return counts

def process_new_data(config):
    """Main function to process new data and integrate it into the dataset."""
    # Extract settings from config
    new_data_dir = config["paths"]["new_data_dir"]
    temp_frames_dir = config["paths"]["temp_frames_dir"]
    dataset_dir = config["paths"]["dataset_dir"]
    frame_interval = config["frame_extraction"]["frame_interval"]
    split_ratios = config["split_ratios"]
    dry_run = config["pipeline"]["dry_run"]

    # Ensure directories exist
    Path(new_data_dir).mkdir(parents=True, exist_ok=True)
    Path(temp_frames_dir).mkdir(parents=True, exist_ok=True)
    Path(dataset_dir).mkdir(parents=True, exist_ok=True)

    # Log initial dataset statistics
    logger.info("Initial dataset statistics:")
    print("Initial dataset statistics:")
    for class_name in ["procedural", "nonprocedural"]:
        counts = count_frames_in_dataset(dataset_dir, class_name)
        stats = f"{class_name}: Train={counts['train']}, Val={counts['val']}, Test={counts['test']}"
        logger.info(stats)
        print(stats)

    # Step 1: Extract frames for both classes
    for class_name in ["procedural", "nonprocedural"]:
        input_video_dir = os.path.join(new_data_dir, class_name)
        temp_frames_dir_class = os.path.join(temp_frames_dir, class_name)

        # Skip if no new data for this class
        if not os.path.exists(input_video_dir):
            logger.info(f"No new data for {class_name}")
            print(f"No new data for {class_name}")
            continue

        # Extract frames
        logger.info(f"Extracting frames for {class_name}...")
        print(f"\nExtracting frames for {class_name}...")
        extract_frames.extract_frames(input_video_dir, temp_frames_dir_class, frame_interval, class_name)

    # Step 2: Split frames across both classes, ensuring subject-level consistency
    logger.info(f"Splitting frames across both classes...")
    print(f"\nSplitting frames across both classes...")
    split_frames.split_frames(temp_frames_dir, dataset_dir, split_ratios, dry_run)

    # Clean up: Remove temp frames directory
    if os.path.exists(temp_frames_dir) and not dry_run:
        shutil.rmtree(temp_frames_dir)
        logger.info(f"Cleaned up temporary frames directory: {temp_frames_dir}")

    # Clean up: Remove processed videos (optional)
    if not dry_run:
        for class_name in ["procedural", "nonprocedural"]:
            input_video_dir = os.path.join(new_data_dir, class_name)
            for video_file in glob.glob(f"{input_video_dir}/*.mp4") + glob.glob(f"{input_video_dir}/*.avi"):
                os.remove(video_file)
                logger.info(f"Removed processed video: {video_file}")

    # Log final dataset statistics
    logger.info("Final dataset statistics:")
    print("\nFinal dataset statistics:")
    for class_name in ["procedural", "nonprocedural"]:
        counts = count_frames_in_dataset(dataset_dir, class_name)
        stats = f"{class_name}: Train={counts['train']}, Val={counts['val']}, Test={counts['test']}"
        logger.info(stats)
        print(stats)


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
        logger = setup_logging(config["pipeline"]["log_file"], logger=logging.getLogger(__name__),)

        process_new_data(config)
        logger.info("New data processing complete!")
        print("\nNew data processing complete!")

    except Exception as e:
        if logger is not None:
            logger.error(f"An error occurred: {e}")
        else:
            print(f"An error occurred before logger was set up: {e}")

# python -m src.dataset_preprocessing.process_new_data --config_file config.yaml

