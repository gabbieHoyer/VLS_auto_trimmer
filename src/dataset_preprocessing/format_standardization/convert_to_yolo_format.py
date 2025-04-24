
import os
import cv2
import argparse
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def convert_to_yolo_format(bbox, img_width, img_height):
    """Convert [x_min, y_min, w, h] to YOLO format [class_id, x_center, y_center, width, height]."""
    x_min, y_min, w, h = bbox
    
    if w <= 0 or h <= 0:
        logger.warning(f"Invalid bounding box {bbox} with zero width or height, skipping.")
        return None
    
    x_center = (x_min + w / 2) / img_width
    y_center = (y_min + h / 2) / img_height
    w_norm = w / img_width
    h_norm = h / img_height
    return [0, x_center, y_center, w_norm, h_norm]  # class_id=0 for 'person'

def convert_dataset(input_dir, output_dir):
    """Convert dataset to YOLO format, preserving the existing train/val/test splits."""
    # Check if input directory exists
    if not os.path.exists(input_dir):
        logger.error(f"Input directory {input_dir} does not exist.")
        raise FileNotFoundError(f"Input directory {input_dir} does not exist.")

    # Check and create output directory
    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output directory {output_dir} created or already exists.")
    except PermissionError as e:
        logger.error(f"Permission denied: Cannot create output directory {output_dir}. Error: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to create output directory {output_dir}. Error: {e}")
        raise

    # Define the splits to process
    splits = ['train', 'val'] #, 'test']

    # Create output directories for each split
    for split in splits:
        try:
            os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
            os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)
            logger.debug(f"Created directories for {split} split: {os.path.join(output_dir, 'images', split)} and {os.path.join(output_dir, 'labels', split)}")
        except Exception as e:
            logger.error(f"Failed to create directories for {split} split. Error: {e}")
            raise

    # Process each split separately
    split_counts = {split: 0 for split in splits}
    for split in splits:
        img_dir = os.path.join(input_dir, split, 'face')
        if not os.path.exists(img_dir):
            logger.warning(f"Directory {img_dir} does not exist, skipping.")
            continue

        for fname in os.listdir(img_dir):
            if fname.endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(img_dir, fname)
                label_path = os.path.splitext(img_path)[0] + '.txt'
                if not os.path.exists(label_path):
                    logger.warning(f"No label file found for {img_path}, skipping.")
                    continue

                # Load the image to get its dimensions
                img = cv2.imread(img_path)
                if img is None:
                    logger.warning(f"Could not load image {img_path}, skipping.")
                    continue
                img_height, img_width = img.shape[:2]

                # Load the bounding boxes
                with open(label_path, 'r') as f:
                    boxes = [list(map(float, line.strip().split())) for line in f.readlines() if line.strip()]
                if not boxes:
                    logger.warning(f"No bounding boxes found in {label_path}, skipping.")
                    continue

                # Convert bounding boxes to YOLO format
                yolo_bboxes = [convert_to_yolo_format(bbox, img_width, img_height) for bbox in boxes]

                yolo_bboxes = [bbox for bbox in yolo_bboxes if bbox is not None]
                if not yolo_bboxes:
                    logger.warning(f"No valid bounding boxes after conversion for {img_path}, skipping.")
                    continue

                # Save the image
                img_name = Path(img_path).name
                output_img_path = os.path.join(output_dir, 'images', split, img_name)
                try:
                    cv2.imwrite(output_img_path, img)
                    logger.debug(f"Saved image to {output_img_path}")
                except Exception as e:
                    logger.warning(f"Failed to save image {output_img_path}. Error: {e}")
                    continue

                # Save the label
                label_path = os.path.join(output_dir, 'labels', split, f"{Path(img_name).stem}.txt")
                try:
                    with open(label_path, 'w') as f:
                        for yolo_bbox in yolo_bboxes:
                            f.write(" ".join(map(str, yolo_bbox)) + "\n")
                    logger.debug(f"Saved label to {label_path}")
                except Exception as e:
                    logger.warning(f"Failed to save label {label_path}. Error: {e}")
                    continue

                split_counts[split] += 1

    # # Create data.yaml with relative paths
    # data_yaml_path = os.path.join(output_dir, 'data.yaml')
    # try:
    #     with open(data_yaml_path, 'w') as f:
    #         for split in splits:
    #             if split_counts[split] > 0:  # Only include splits with data
    #                 f.write(f"{split}: ./images/{split}\n")
    #         f.write("nc: 1\n")
    #         f.write("names: ['person']\n")
    #     logger.info(f"Created data.yaml at {data_yaml_path}")
    # except Exception as e:
    #     logger.error(f"Failed to create data.yaml at {data_yaml_path}. Error: {e}")
    #     raise

    # Log the results
    logger.info(f"Dataset converted and saved to {output_dir}")
    for split in splits:
        logger.info(f"{split.capitalize()} split: {split_counts[split]} images")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert dataset to YOLO format.")
    parser.add_argument('--input_dir', type=str, required=True, help="Path to the input dataset directory.")
    parser.add_argument('--output_dir', type=str, default="dataset_yolov8", help="Path to the output YOLO dataset directory.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    convert_dataset(args.input_dir, args.output_dir)

# From root directory:
# python -m src.dataset_preprocessing.format_standardization.convert_to_yolo_format
# python -m src.dataset_preprocessing.format_standardization.convert_to_yolo_format --input_dir /data/mskscratch/users/ghoyer/Precision_Air/0403/face_videos/splits --output_dir dataset_yolov8_test
# python -m src.dataset_preprocessing.format_standardization.convert_to_yolo_format --input_dir /data/mskscratch/users/ghoyer/Precision_Air/0403/face_videos/splits --output_dir /data/mskscratch/users/ghoyer/Precision_Air/0403/face_videos/dataset_yolov8

#  if in ../format_standardization directory:
# python convert_to_yolo_format.py --input_dir /data/mskscratch/users/ghoyer/Precision_Air/0403/face_videos/frame_part4_yolo --output_dir /data/mskscratch/users/ghoyer/Precision_Air/0403/face_videos/dataset_yolo_refined

# ----------------------------------------------------------
