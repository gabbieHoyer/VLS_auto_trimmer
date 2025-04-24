import os
import cv2
import numpy as np
import argparse
import logging
import random
import shutil
import tempfile
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def setup_logging(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def crop_image(
    input_image_path,
    output_image_path,
    left_crop=160,
    right_crop=160
):
    """Crop from both sides and save the result."""
    img = cv2.imread(input_image_path)
    if img is None:
        raise ValueError(f"Could not load image: {input_image_path}")

    orig_height, orig_width = img.shape[:2]
    left_border = left_crop
    right_border = orig_width - right_crop
    cropped_img = img[:, left_border:right_border]

    cropped_height, cropped_width = cropped_img.shape[:2]
    logging.info(
        f"Crop result: Original {orig_width}x{orig_height}, "
        f"Cropped {cropped_width}x{cropped_height}"
    )

    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    cv2.imwrite(output_image_path, cropped_img)
    return cropped_img, cropped_width, cropped_height

def resize_image(
    image,
    target_size=(960, 720),
    output_path=None
):
    """Resize the image to a target size and save it."""
    resized_img = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, resized_img)

    new_h, new_w = resized_img.shape[:2]
    return resized_img, new_w, new_h
def update_yolo_labels(
    input_label_path,
    output_label_path,
    orig_width,
    orig_height,
    new_img_width,
    new_img_height,
    left_crop=0,
    right_crop=0,
    max_box_size=0.5
):
    """
    Update YOLO labels given the old and new image sizes, accounting for
    cropping or resizing. When horizontally cropping 1280->960, do not alter
    vertical dimension or scale in that direction. This keeps box heights the same.
    """
    updated_labels = []

    # If we are purely cropping width from 1280 to 960, the new_img_height is still 720,
    # so we do not scale vertically. For smaller images (640x480 -> 960x720), we do need
    # both width and height scaling.
    if (orig_width == 1280 and orig_height == 720 and new_img_width == 960 and new_img_height == 720):
        # Horizontal crop only, no vertical scaling
        width_scale = new_img_width / orig_width  # e.g., 960/1280 = 0.75
        height_scale = 1.0                       # no change in height
    else:
        # For other cases (e.g., 640x480 -> 960x720), do normal scaling
        width_scale = new_img_width / orig_width if orig_width > 0 else 1.0
        height_scale = new_img_height / orig_height if orig_height > 0 else 1.0

    if not os.path.exists(input_label_path):
        logging.warning(f"Label file not found: {input_label_path}")
        return [], width_scale, height_scale

    with open(input_label_path, 'r') as f:
        lines = f.readlines()
    logging.info(f"Processing {len(lines)} annotations in {input_label_path}")

    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            logging.warning(f"Invalid label format: {line}")
            continue

        try:
            class_id, x_center_n, y_center_n, w_n, h_n = map(float, parts)
        except ValueError:
            logging.warning(f"Failed to parse label: {line}")
            continue

        # Convert from normalized coords to pixel coords
        x_center_px = x_center_n * orig_width
        y_center_px = y_center_n * orig_height
        box_w_px = w_n * orig_width
        box_h_px = h_n * orig_height

        x_left_px = x_center_px - 0.5 * box_w_px
        x_right_px = x_center_px + 0.5 * box_w_px
        y_top_px = y_center_px - 0.5 * box_h_px
        y_bottom_px = y_center_px + 0.5 * box_h_px

        # If left_crop or right_crop > 0, we're cropping horizontally
        if (left_crop > 0) or (right_crop > 0):
            # Example: left_crop=160, right_crop=160 => new width=960, old=1280
            left_border = left_crop
            right_border = orig_width - right_crop

            # If the entire box is to the left or right of the new region, skip
            if x_right_px < left_border or x_left_px > right_border:
                continue

            # Clip left/right edges if partially outside
            clipped_left_px = max(x_left_px, left_border)
            clipped_right_px = min(x_right_px, right_border)

            # Shift these into the new coordinate system (0..new_img_width)
            new_left_px = clipped_left_px - left_border
            new_right_px = clipped_right_px - left_border

            # Box's vertical edges do not change because we do not crop vertically
            new_top_px = y_top_px
            new_bottom_px = y_bottom_px
            new_box_h_px = new_bottom_px - new_top_px  # same as original box_h_px

            # Recompute new center in X
            new_box_w_px = new_right_px - new_left_px
            new_center_x_px = new_left_px + 0.5 * new_box_w_px

            # Center in Y does not shift
            new_center_y_px = y_center_px

            # Convert back to normalized using (new_img_width, new_img_height)
            new_x_center_n = new_center_x_px / new_img_width
            new_box_w_n = new_box_w_px / new_img_width
            new_y_center_n = new_center_y_px / new_img_height
            new_box_h_n = new_box_h_px / new_img_height

            # If you do not want any capping for the big images, skip it here:
            # (comment out or remove lines below for no capping at all)
            #
            # For example:
            #   if new_box_w_n > max_box_size: ...
            #
            # remains optional, but you likely do not want to reduce vertical dimension
            # just because it's above 0.5.
            #
            # Let's keep the capping for the sake of smaller images, but skip it for
            # the horizontal-only scenario:

        else:
            # Resizing (likely 640x480 to 960x720)
            new_center_x_px = x_center_px * width_scale
            new_center_y_px = y_center_px * height_scale
            new_box_w_px = box_w_px * width_scale
            new_box_h_px = box_h_px * height_scale

            new_x_center_n = new_center_x_px / new_img_width
            new_box_w_n = new_box_w_px / new_img_width
            new_y_center_n = new_center_y_px / new_img_height
            new_box_h_n = new_box_h_px / new_img_height

            # If you still want to cap extremely large boxes in the resized case:
            if new_box_w_n > max_box_size:
                logging.warning(
                    f"Box width {new_box_w_n:.4f} > {max_box_size}, capping."
                )
                new_box_w_n = max_box_size
            if new_box_h_n > max_box_size:
                logging.warning(
                    f"Box height {new_box_h_n:.4f} > {max_box_size}, capping."
                )
                new_box_h_n = max_box_size

        # Clip center to [0, 1]
        new_x_center_n = max(0.0, min(1.0, new_x_center_n))
        new_y_center_n = max(0.0, min(1.0, new_y_center_n))

        # Ensure box width/height are never zero
        new_box_w_n = max(new_box_w_n, 0.001)
        new_box_h_n = max(new_box_h_n, 0.001)

        updated_line = (
            f"{int(class_id)} "
            f"{new_x_center_n:.6f} {new_y_center_n:.6f} "
            f"{new_box_w_n:.6f} {new_box_h_n:.6f}\n"
        )
        updated_labels.append(updated_line)

    logging.info(f"Retained {len(updated_labels)} labels in {output_label_path}")
    os.makedirs(os.path.dirname(output_label_path), exist_ok=True)
    with open(output_label_path, 'w') as f_out:
        f_out.writelines(updated_labels)

    return updated_labels, width_scale, height_scale


def draw_bounding_boxes(image, label_path, img_width, img_height):
    """Read YOLO labels from label_path and produce bounding boxes in pixel coords."""
    boxes = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            _, x_c, y_c, w, h = map(float, parts)
            x_center = x_c * img_width
            y_center = y_c * img_height
            box_width = w * img_width
            box_height = h * img_height
            x1 = x_center - box_width / 2.0
            y1 = y_center - box_height / 2.0
            boxes.append((x1, y1, box_width, box_height))
    return boxes

def visualize_changes(
    original_image_path,
    transformed_img,
    original_label_path,
    updated_label_path,
    output_path,
    orig_width,
    orig_height,
    new_img_width,
    new_img_height
):
    """Show original vs transformed images with bounding boxes."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    orig_img = cv2.imread(original_image_path)
    orig_img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    axes[0].imshow(orig_img_rgb)
    orig_boxes = draw_bounding_boxes(orig_img, original_label_path, orig_width, orig_height)
    for x1, y1, w, h in orig_boxes:
        rect = Rectangle((x1, y1), w, h, linewidth=2, edgecolor='r', facecolor='none')
        axes[0].add_patch(rect)
    axes[0].set_title(f'Original {orig_width}x{orig_height}')
    axes[0].axis('off')

    transformed_img_rgb = cv2.cvtColor(transformed_img, cv2.COLOR_BGR2RGB)
    axes[1].imshow(transformed_img_rgb)
    updated_boxes = draw_bounding_boxes(transformed_img, updated_label_path, new_img_width, new_img_height)
    for x1, y1, w, h in updated_boxes:
        rect = Rectangle((x1, y1), w, h, linewidth=2, edgecolor='r', facecolor='none')
        axes[1].add_patch(rect)
    axes[1].set_title(f'Transformed {new_img_width}x{new_img_height}')
    axes[1].axis('off')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

def process_dataset(
    input_dir,
    output_dir,
    crop_width=160,
    target_size=(960, 720),
    dry_run=False,
    num_viz_samples=5,
    dry_run_samples=5
):
    """
    Reads images from input_dir/images and labels from input_dir/labels.
    Outputs processed images/labels to output_dir/images and output_dir/labels.
    If dry_run is enabled, only dry_run_samples images are processed
    and saved to output_dir/dry_run_temp.
    """
    in_image_dir = os.path.join(input_dir, 'images')
    in_label_dir = os.path.join(input_dir, 'labels')

    if not os.path.exists(in_image_dir) or not os.path.exists(in_label_dir):
        raise ValueError(f"Images or labels directory not found in {input_dir}")

    out_image_dir = os.path.join(output_dir, 'images')
    out_label_dir = os.path.join(output_dir, 'labels')
    out_viz_dir = os.path.join(output_dir, 'visualizations')

    os.makedirs(out_image_dir, exist_ok=True)
    os.makedirs(out_label_dir, exist_ok=True)
    os.makedirs(out_viz_dir, exist_ok=True)

    image_files = [
        f for f in os.listdir(in_image_dir)
        if f.lower().endswith(('.jpg', '.jpeg'))
    ]
    logging.info(f"Found {len(image_files)} images in {in_image_dir}")

    # If dry_run, create a temp subfolder and select subset
    if dry_run:
        temp_dir = os.path.join(output_dir, 'dry_run_temp')
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)
        selected_files = random.sample(image_files, min(dry_run_samples, len(image_files)))
        image_files = selected_files
        logging.info(f"Dry run active. Processing {len(selected_files)} images.")
        out_image_dir = os.path.join(temp_dir, 'images')
        out_label_dir = os.path.join(temp_dir, 'labels')
        out_viz_dir = os.path.join(temp_dir, 'visualizations')
        os.makedirs(out_image_dir, exist_ok=True)
        os.makedirs(out_label_dir, exist_ok=True)
        os.makedirs(out_viz_dir, exist_ok=True)

    # Separate images by dimension
    small_images = []
    large_images = []
    for image_file in image_files:
        path_in = os.path.join(in_image_dir, image_file)
        img = cv2.imread(path_in)
        if img is None:
            logging.warning(f"Skipping unreadable file: {image_file}")
            continue
        h, w = img.shape[:2]
        if w == 640 and h == 480:
            small_images.append(image_file)
        elif w == 1280 and h == 720:
            large_images.append(image_file)
        else:
            logging.warning(f"Skipping {image_file} due to unexpected size {w}x{h}")

    # Pick images for visualization
    num_small_viz = min(len(small_images), num_viz_samples // 2)
    num_large_viz = min(len(large_images), num_viz_samples - num_small_viz)
    viz_small = random.sample(small_images, num_small_viz) if small_images else []
    viz_large = random.sample(large_images, num_large_viz) if large_images else []
    viz_files = set(viz_small + viz_large)

    # Process
    for image_file in small_images + large_images:
        in_image_path = os.path.join(in_image_dir, image_file)
        out_image_path = os.path.join(out_image_dir, image_file)
        label_file = os.path.splitext(image_file)[0] + '.txt'
        in_label_path = os.path.join(in_label_dir, label_file)
        out_label_path = os.path.join(out_label_dir, label_file)
        viz_path = os.path.join(out_viz_dir, f"{os.path.splitext(image_file)[0]}_viz.png")

        img = cv2.imread(in_image_path)
        if img is None:
            logging.error(f"Could not load {in_image_path}")
            continue

        orig_h, orig_w = img.shape[:2]
        transformed_img = None
        new_w, new_h = orig_w, orig_h

        try:
            # Crop if 1280x720
            if orig_w == 1280 and orig_h == 720:
                cropped_img, cropped_w, cropped_h = crop_image(
                    in_image_path,
                    out_image_path,
                    left_crop=crop_width,
                    right_crop=crop_width
                )
                transformed_img = cropped_img
                new_w, new_h = cropped_w, cropped_h

                update_yolo_labels(
                    input_label_path=in_label_path,
                    output_label_path=out_label_path,
                    orig_width=orig_w,
                    orig_height=orig_h,
                    new_img_width=cropped_w,
                    new_img_height=cropped_h,
                    left_crop=crop_width,
                    right_crop=crop_width
                )

            # Resize if 640x480
            elif orig_w == 640 and orig_h == 480:
                resized_img, resized_w, resized_h = resize_image(
                    img,
                    target_size,
                    output_path=out_image_path
                )
                transformed_img = resized_img
                new_w, new_h = resized_w, resized_h

                update_yolo_labels(
                    input_label_path=in_label_path,
                    output_label_path=out_label_path,
                    orig_width=orig_w,
                    orig_height=orig_h,
                    new_img_width=resized_w,
                    new_img_height=resized_h
                )
            else:
                # Not an expected dimension
                continue

            # Visualization if selected
            if image_file in viz_files and transformed_img is not None:
                visualize_changes(
                    original_image_path=in_image_path,
                    transformed_img=transformed_img,
                    original_label_path=in_label_path,
                    updated_label_path=out_label_path,
                    output_path=viz_path,
                    orig_width=orig_w,
                    orig_height=orig_h,
                    new_img_width=new_w,
                    new_img_height=new_h
                )
                logging.info(f"Wrote visualization for {image_file}")

        except Exception as e:
            logging.error(f"Error processing {image_file}: {str(e)}")

    if dry_run:
        logging.info(
            "Dry run completed. The subset of processed files is in "
            f"'{os.path.join(output_dir, 'dry_run_temp')}'."
        )

def main():
    parser = argparse.ArgumentParser(
        description="Crop 1280x720 images or resize 640x480 images to 960x720, then update YOLO labels."
    )
    parser.add_argument('--input_dir', required=True, help="Input directory with images and labels")
    parser.add_argument('--output_dir', required=True, help="Output directory for results")
    parser.add_argument('--crop_width', type=int, default=160, help="Pixels to crop from each side of 1280x720 images")
    parser.add_argument('--target_size', type=int, nargs=2, default=[960, 720], help="Target width height for resizing 640x480 images")
    parser.add_argument('--dry_run', action='store_true', help="If set, processes only a small sample into dry_run_temp")
    parser.add_argument('--dry_run_samples', type=int, default=5, help="Number of files to process in dry_run mode")
    parser.add_argument('--num_viz_samples', type=int, default=5, help="Number of images to visualize")
    parser.add_argument('--log_file', default='standardize_log.txt', help="Path for log file")

    args = parser.parse_args()
    setup_logging(args.log_file)
    logging.info(
        f"Starting with input_dir={args.input_dir}, output_dir={args.output_dir}, "
        f"crop_width={args.crop_width}, target_size={args.target_size}, "
        f"dry_run={args.dry_run}, dry_run_samples={args.dry_run_samples}, "
        f"num_viz_samples={args.num_viz_samples}"
    )

    process_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        crop_width=args.crop_width,
        target_size=tuple(args.target_size),
        dry_run=args.dry_run,
        num_viz_samples=args.num_viz_samples,
        dry_run_samples=args.dry_run_samples
    )
    logging.info("All processing completed")

if __name__ == "__main__":
    main()

# From root directory:
# python -m src.dataset_preprocessing.format_standardization.standardize_image_sizes
# python -m src.dataset_preprocessing.format_standardization.standardize_image_sizes --input_dir /data/mskscratch/users/ghoyer/Precision_Air/0403/face_videos/dataset_yolo_refined7/val2 --output_dir /data/mskscratch/users/ghoyer/Precision_Air/0403/face_videos/dataset_yolo_refined7/val --crop_width 160 --dry_run --dry_run_samples 15 --num_viz_samples 15

# From format_standardization directory:
# python standardize_image_sizes.py --input_dir /data/mskscratch/users/ghoyer/Precision_Air/0403/face_videos/dataset_yolo_refined7/val2 --output_dir /data/mskscratch/users/ghoyer/Precision_Air/0403/face_videos/dataset_yolo_refined7/val --crop_width 160 --dry_run --dry_run_samples 15 --num_viz_samples 15


