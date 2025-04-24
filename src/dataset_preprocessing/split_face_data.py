import os
import glob
import random
import shutil
from collections import defaultdict
import argparse
from pathlib import Path
import logging

from src.utils import load_config, setup_logging

def extract_base_name(filename):
    """
    Extract the base video name from a filename by splitting on '_frame'.
    
    Args:
        filename (str): Filename like 'video1_frame000001.jpg'.
    
    Returns:
        str: Base video name (e.g., 'video1').
    """
    base = os.path.splitext(filename)[0]  # Remove .jpg
    return base.split('_frame')[0]  # Take part before '_frame'

def split_face_data(image_dir, label_dir, output_base_dir, split_ratios=None, dry_run=False):
    """
    Split face images and their annotations into train/val/test sets, ensuring all frames
    from the same base video are in the same split.
    
    Args:
        image_dir (str): Directory with face images (.jpg).
        label_dir (str): Directory with corresponding .txt annotations.
        output_base_dir (str): Base directory to create train/val/test splits.
        split_ratios (dict): Dictionary with split ratios (e.g., {'train': 0.7, 'val': 0.15, 'test': 0.15}).
        dry_run (bool): If True, simulate the process without moving files.
    """
    # Define class name (single class for face data)
    class_name = "face"

    # Define output directories for each split
    output_dirs = {
        "train": os.path.join(output_base_dir, "train", class_name),
        "val": os.path.join(output_base_dir, "val", class_name),
        "test": os.path.join(output_base_dir, "test", class_name)
    }
    if not dry_run:
        for split_dir in output_dirs.values():
            Path(split_dir).mkdir(parents=True, exist_ok=True)

    # Get all image files
    image_files = sorted([f for f in glob.glob(f"{image_dir}/*.[jp][pn]g") if os.path.isfile(f)])
    if not image_files:
        logging.info(f"No .jpg or .jpeg files found in {image_dir}")
        print(f"No .jpg or .jpeg files found in {image_dir}")
        return

    # Group image-annotation pairs by base video name
    video_groups = defaultdict(list)
    for img_path in image_files:
        img_basename = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(label_dir, f"{img_basename}.txt")
        if os.path.exists(label_path):
            base_name = extract_base_name(img_basename)
            video_groups[base_name].append((img_path, label_path))
        else:
            logging.warning(f"No annotation found for {img_path}. Skipping.")
            print(f"Warning: No annotation found for {img_path}. Skipping.")

    if not video_groups:
        logging.info("No image-annotation pairs found after checking.")
        print("No image-annotation pairs found after checking.")
        return

    # Prepare video group list with frame counts
    video_group_list = []
    for video_name, pairs in video_groups.items():
        total_frames_for_video = len(pairs)
        video_group_list.append((video_name, pairs, total_frames_for_video))

    # Shuffle and sort by total frame count
    random.shuffle(video_group_list)
    total_frames = sum(item[2] for item in video_group_list)
    video_group_list.sort(key=lambda x: x[2], reverse=True)  # Sort by total frame count, largest first

    # Target frame counts for each split
    train_target = int(total_frames * split_ratios["train"])
    val_target = int(total_frames * split_ratios["val"])
    test_target = total_frames - train_target - val_target

    # Assign videos to splits
    train_groups = []
    val_groups = []
    test_groups = []
    train_frames = 0
    val_frames = 0
    test_frames = 0

    # If train ratio is 1.0, assign everything to train and skip the rest
    if split_ratios["train"] == 1.0:
        for video_name, pairs, num_frames in video_group_list:
            train_groups.append((video_name, pairs))
            train_frames += num_frames
    else:
        for video_name, pairs, num_frames in video_group_list:
            train_percent = (train_frames + num_frames) / total_frames if train_frames + num_frames > 0 else 0
            val_percent = (val_frames + num_frames) / total_frames if val_frames + num_frames > 0 else 0
            test_percent = (test_frames + num_frames) / total_frames if test_frames + num_frames > 0 else 0

            if train_percent <= split_ratios["train"] and (val_percent >= split_ratios["val"] or train_percent < val_percent) and (test_percent >= split_ratios["test"] or train_percent < test_percent):
                train_groups.append((video_name, pairs))
                train_frames += num_frames
            elif val_percent < split_ratios["val"] and (test_percent >= split_ratios["test"] or val_percent < test_percent):
                val_groups.append((video_name, pairs))
                val_frames += num_frames
            else:
                test_groups.append((video_name, pairs))
                test_frames += num_frames

    # Log the split assignments for debugging
    logging.info(f"Train groups: {len(train_groups)} videos, {train_frames} frames")
    logging.info(f"Val groups: {len(val_groups)} videos, {val_frames} frames")
    logging.info(f"Test groups: {len(test_groups)} videos, {test_frames} frames")
    print(f"Train groups: {len(train_groups)} videos, {train_frames} frames")
    print(f"Val groups: {len(val_groups)} videos, {val_frames} frames")
    print(f"Test groups: {len(test_groups)} videos, {test_frames} frames")

    # Function to process files (or simulate for dry run)
    def process_files(group_list, split_name):
        total_files_processed = 0
        for video_name, pairs in group_list:
            for img_path, label_path in pairs:
                img_filename = os.path.basename(img_path)
                label_filename = os.path.basename(label_path)
                dest_img_path = os.path.join(output_dirs[split_name], img_filename)
                dest_label_path = os.path.join(output_dirs[split_name], label_filename)

                # Check for duplicate filenames
                for dest_path in [dest_img_path, dest_label_path]:
                    base_name, ext = os.path.splitext(os.path.basename(dest_path))
                    counter = 1
                    while os.path.exists(dest_path):
                        new_filename = f"{base_name}_{counter}{ext}"
                        dest_path_new = os.path.join(output_dirs[split_name], new_filename)
                        if dest_path == dest_img_path:
                            dest_img_path = dest_path_new
                        else:
                            dest_label_path = dest_path_new
                        counter += 1

                if dry_run:
                    logging.info(f"[Dry Run] Would move {img_filename} to {dest_img_path}")
                    logging.info(f"[Dry Run] Would move {label_filename} to {dest_label_path}")
                    print(f"[Dry Run] Would move {img_filename} to {dest_img_path}")
                    print(f"[Dry Run] Would move {label_filename} to {dest_label_path}")
                else:
                    shutil.copy(img_path, dest_img_path)
                    shutil.copy(label_path, dest_label_path)
                    logging.info(f"Copied {img_filename} to {dest_img_path}")
                    logging.info(f"Copied {label_filename} to {dest_label_path}")
                total_files_processed += 1
        return total_files_processed

    # Process files for each split
    logging.info(f"Processing splits..." if not dry_run else f"Simulating splits (Dry Run)...")
    print(f"Processing splits..." if not dry_run else f"Simulating splits (Dry Run)...")

    # Process train split
    train_files_processed = process_files(train_groups, "train")

    # Process validation split only if there are videos or the target ratio is non-zero
    if val_groups or split_ratios["val"] > 0:
        val_files_processed = process_files(val_groups, "val")
    else:
        val_files_processed = 0
        logging.info("Skipping validation split (0% target)")

    # Process test split only if there are videos or the target ratio is non-zero
    if test_groups or split_ratios["test"] > 0:
        test_files_processed = process_files(test_groups, "test")
    else:
        test_files_processed = 0
        logging.info("Skipping test split (0% target)")

    # Print summary
    total_processed = train_files_processed + val_files_processed + test_files_processed
    summary = f"""
Total frames: {total_frames}
Total unique videos: {len(video_group_list)}

Class '{class_name}' breakdown:
  Total frames: {total_processed}
  Training split: {len(train_groups)} videos, {train_files_processed} frames
  Validation split: {len(val_groups)} videos, {val_files_processed} frames
  Test split: {len(test_groups)} videos, {test_files_processed} frames

  Target split percentages:
    Train target: {split_ratios['train']*100:.1f}% ({train_target} frames total)
    Val target: {split_ratios['val']*100:.1f}% ({val_target} frames total)
    Test target: {split_ratios['test']*100:.1f}% ({test_target} frames total)

  Actual split percentages:
    Train: {(train_files_processed/total_processed)*100:.1f}% ({train_files_processed} frames)
    Val: {(val_files_processed/total_processed)*100:.1f}% ({val_files_processed} frames)
    Test: {(test_files_processed/total_processed)*100:.1f}% ({test_files_processed} frames)
"""
    if not dry_run:
        summary += f"""
Files moved to:
  Train: {output_dirs['train']}
  Val: {output_dirs['val']}
  Test: {output_dirs['test']}
"""
    else:
        summary += f"\n[Dry Run] No files were actually moved. Set dry_run = False to perform the copy."

    logging.info(summary)
    print(summary)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split face images and annotations into train/val/test sets.")
    parser.add_argument("--image_dir", default="/data/mskscratch/users/ghoyer/Precision_Air/0403/face_videos/frame_images_part4",
                        help="Directory containing face images")
    parser.add_argument("--label_dir", default="/data/mskscratch/users/ghoyer/Precision_Air/0403/face_videos/frame_labels_txt_part4",
                        help="Directory containing annotation files")
    parser.add_argument("--config_file", default="face_config.yaml", help="Path to config file")
    parser.add_argument("--dry_run", type=str, default=None, choices=["on", "off", None],
                        help="Dry run mode: 'on' to simulate, 'off' to copy, None to use config")
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

        # Extract settings from config
        output_base_dir = config["paths"]["dataset_dir"]
        split_ratios = config["split_ratios"]
        dry_run = args.dry_run == "on" if args.dry_run is not None else config["pipeline"]["dry_run"]

        print(f"Splitting face data from {args.image_dir} and {args.label_dir} into {output_base_dir} (dry_run={dry_run})")
        split_face_data(args.image_dir, args.label_dir, output_base_dir, split_ratios, dry_run)

        logger.info("New data processing complete!")
        print("\nNew data processing complete!")

    except Exception as e:
        if logger is not None:
            logger.error(f"An error occurred: {e}")
        else:
            print(f"An error occurred before logger was set up: {e}")

