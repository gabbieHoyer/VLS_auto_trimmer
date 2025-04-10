
import os
import glob
import random
import shutil
from collections import defaultdict
import argparse
from pathlib import Path
import logging

from src.utils import load_config, setup_logging

def split_frames(temp_frames_dir, dataset_dir, split_ratios=None, dry_run=False):
    """
    Split frames into train/val/test sets, ensuring that all frames from the same subject
    (across procedural and nonprocedural classes) are assigned to the same split.
    
    Args:
        temp_frames_dir (str): Directory containing extracted frames (with procedural/ and nonprocedural/ subdirs).
        dataset_dir (str): Base dataset directory (e.g., 'data/').
        split_ratios (dict): Dictionary with split ratios (e.g., {'train': 0.7, 'val': 0.2, 'test': 0.1}).
        dry_run (bool): If True, simulate the process without moving files.
    """
    # Define class names
    class_names = ["procedural", "nonprocedural"]

    # Define output directories for each class and split
    output_dirs = {}
    for class_name in class_names:
        output_dirs[class_name] = {
            "train": os.path.join(dataset_dir, "train", class_name),
            "val": os.path.join(dataset_dir, "val", class_name),
            "test": os.path.join(dataset_dir, "test", class_name)
        }
        if not dry_run:
            for split_dir in output_dirs[class_name].values():
                Path(split_dir).mkdir(parents=True, exist_ok=True)

    # Get all JPEG files for both classes
    all_frames = {}
    for class_name in class_names:
        class_frames_dir = os.path.join(temp_frames_dir, class_name)
        image_files = glob.glob(os.path.join(class_frames_dir, "*.jpeg"))
        all_frames[class_name] = image_files

    # Check if there are frames to process
    total_frames_all_classes = sum(len(files) for files in all_frames.values())
    if total_frames_all_classes == 0:
        logging.info(f"No frames found in {temp_frames_dir}")
        print(f"No frames found in {temp_frames_dir}")
        return

    # Group frames by video (subject) across both classes
    video_groups = defaultdict(lambda: defaultdict(list))
    for class_name in class_names:
        for image_file in all_frames[class_name]:
            filename = os.path.basename(image_file)
            video_name = filename.split("_frame")[0]
            video_groups[video_name][class_name].append(image_file)

    # Prepare video group list with frame counts (total frames per subject across both classes)
    video_group_list = []
    for video_name, class_files in video_groups.items():
        total_frames_for_video = sum(len(files) for files in class_files.values())
        video_group_list.append((video_name, class_files, total_frames_for_video))

    # Shuffle and sort by total frame count
    random.shuffle(video_group_list)
    total_frames = sum(item[2] for item in video_group_list)
    video_group_list.sort(key=lambda x: x[2], reverse=True)  # Sort by total frame count, largest first

    # Target frame counts for each split
    train_target = int(total_frames * split_ratios["train"])
    val_target = int(total_frames * split_ratios["val"])
    test_target = total_frames - train_target - val_target

    # Assign videos (subjects) to splits
    train_groups = []
    val_groups = []
    test_groups = []
    train_frames = 0
    val_frames = 0
    test_frames = 0

    # If train ratio is 1.0, assign everything to train and skip the rest
    if split_ratios["train"] == 1.0:
        for video_name, class_files, num_frames in video_group_list:
            train_groups.append((video_name, class_files))
            train_frames += num_frames
    else:
        for video_name, class_files, num_frames in video_group_list:
            train_percent = (train_frames + num_frames) / total_frames if train_frames + num_frames > 0 else 0
            val_percent = (val_frames + num_frames) / total_frames if val_frames + num_frames > 0 else 0
            test_percent = (test_frames + num_frames) / total_frames if test_frames + num_frames > 0 else 0

            if train_percent <= split_ratios["train"] and (val_percent >= split_ratios["val"] or train_percent < val_percent) and (test_percent >= split_ratios["test"] or train_percent < test_percent):
                train_groups.append((video_name, class_files))
                train_frames += num_frames
            elif val_percent < split_ratios["val"] and (test_percent >= split_ratios["test"] or val_percent < test_percent):
                val_groups.append((video_name, class_files))
                val_frames += num_frames
            else:
                test_groups.append((video_name, class_files))
                test_frames += num_frames

    # Function to move files (or simulate moving for dry run)
    def process_files(group_list, split_name):
        total_files_processed = {class_name: 0 for class_name in class_names}
        for video_name, class_files in group_list:
            for class_name in class_names:
                if class_name not in class_files:
                    continue
                files = class_files[class_name]
                dest_dir = output_dirs[class_name][split_name]
                for file_path in files:
                    dest_path = os.path.join(dest_dir, os.path.basename(file_path))
                    # Check for duplicate filenames
                    base_name, ext = os.path.splitext(os.path.basename(file_path))
                    counter = 1
                    while os.path.exists(dest_path):
                        new_filename = f"{base_name}_{counter}{ext}"
                        dest_path = os.path.join(dest_dir, new_filename)
                        counter += 1
                    if dry_run:
                        logging.info(f"[Dry Run] Would move {os.path.basename(file_path)} to {dest_path}")
                        print(f"[Dry Run] Would move {os.path.basename(file_path)} to {dest_path}")
                    else:
                        shutil.move(file_path, dest_path)
                        logging.info(f"Moved {os.path.basename(file_path)} to {dest_path}")
                    total_files_processed[class_name] += 1
        return total_files_processed

    # Process files for each split
    logging.info(f"Processing splits..." if not dry_run else f"Simulating splits (Dry Run)...")
    print(f"Processing splits..." if not dry_run else f"Simulating splits (Dry Run)...")
    train_files_processed = process_files(train_groups, "train")
    val_files_processed = process_files(val_groups, "val")
    test_files_processed = process_files(test_groups, "test")

    # Print summary
    total_processed = {class_name: train_files_processed[class_name] + val_files_processed[class_name] + test_files_processed[class_name] for class_name in class_names}
    summary = f"""
Total frames across all classes: {total_frames}
Total unique subjects (videos): {len(video_group_list)}

Per-class breakdown:
"""
    for class_name in class_names:
        summary += f"""
{class_name}:
  Total frames: {total_processed[class_name]}
  Training split: {len([g for g in train_groups if class_name in g[1]])} videos, {train_files_processed[class_name]} frames
  Validation split: {len([g for g in val_groups if class_name in g[1]])} videos, {val_files_processed[class_name]} frames
  Test split: {len([g for g in test_groups if class_name in g[1]])} videos, {test_files_processed[class_name]} frames

  Target split percentages:
    Train target: {split_ratios['train']*100:.1f}% ({train_target} frames total)
    Val target: {split_ratios['val']*100:.1f}% ({val_target} frames total)
    Test target: {split_ratios['test']*100:.1f}% ({test_target} frames total)

  Actual split percentages:
    Train: {(train_files_processed[class_name]/total_processed[class_name])*100:.1f}% ({train_files_processed[class_name]} frames)
    Val: {(val_files_processed[class_name]/total_processed[class_name])*100:.1f}% ({val_files_processed[class_name]} frames)
    Test: {(test_files_processed[class_name]/total_processed[class_name])*100:.1f}% ({test_files_processed[class_name]} frames)
"""
    if not dry_run:
        summary += f"""
Files moved to:
  Train: {output_dirs['procedural']['train']}, {output_dirs['nonprocedural']['train']}
  Val: {output_dirs['procedural']['val']}, {output_dirs['nonprocedural']['val']}
  Test: {output_dirs['procedural']['test']}, {output_dirs['nonprocedural']['test']}
"""
    else:
        summary += f"\n[Dry Run] No files were actually moved. Set dry_run = False to perform the move."

    logging.info(summary)
    print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split frames into train/val/test sets.")
    parser.add_argument("--temp_frames_dir", default=None, help="Directory containing extracted frames (with procedural/ and nonprocedural/ subdirs)")
    parser.add_argument("--dataset_dir", default=None, help="Directory containing ML splits (train/val/test)")
    parser.add_argument("--dry_run", default=None, help="dry_run (overrides config)")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),  # Up three levels to root/
        "config",
        args.config
    )

    try:
        # Load config and set up logging
        config = load_config(config_path)
        logger = setup_logging(config["pipeline"]["log_file"], logger=logging.getLogger(__name__))

        # Determine arguments: use command-line args if provided, otherwise fall back to config
        temp_frames_dir = args.temp_frames_dir if args.temp_frames_dir is not None else config["paths"]["temp_frames_dir"]
        dataset_dir = args.dataset_dir if args.dataset_dir is not None else config["paths"]["dataset_dir"]
        dry_run = args.dry_run if args.dry_run is not None else config["pipeline"]["dry_run"]
        
        split_ratios = config["split_ratios"]

        split_frames(temp_frames_dir, dataset_dir, split_ratios, dry_run)

        logger.info("Extracting frames from video complete!")
        print("\nExtracting frames from video complete!")

    except Exception as e:
        if 'logger' in locals():
            logger.error(f"An error occurred: {e}")
        else:
            print(f"An error occurred before logger was set up: {e}")
        raise

#  python -m src.dataset_preprocessing.split_frames --config config.yaml
#  python -m src.dataset_preprocessing.split_frames --config config.yaml --dry_run True


