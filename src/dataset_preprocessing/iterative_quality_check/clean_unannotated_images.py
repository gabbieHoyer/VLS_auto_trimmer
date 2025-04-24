import os
import glob
import argparse
from pathlib import Path

def clean_unannotated_images(image_dir, label_dir, dry_run=True):
    """
    Identify and optionally delete image files without corresponding annotation files.
    
    Args:
        image_dir (str): Directory containing image files (.jpg).
        label_dir (str): Directory containing Pascal VOC annotation files (.xml).
        dry_run (bool): If True, only list unannotated images; if False, delete them.
    """
    # Ensure directories exist
    if not os.path.exists(image_dir):
        print(f"Error: Image directory {image_dir} does not exist.")
        return
    if not os.path.exists(label_dir):
        print(f"Error: Label directory {label_dir} does not exist.")
        return

    # Get all image and annotation files
    image_files = set(os.path.splitext(os.path.basename(f))[0] for f in glob.glob(f"{image_dir}/*.jpg"))
    label_files = set(os.path.splitext(os.path.basename(f))[0] for f in glob.glob(f"{label_dir}/*.xml"))

    # Find images without annotations
    unannotated_images = image_files - label_files

    if not unannotated_images:
        print("No unannotated images found.")
        return

    print(f"Found {len(unannotated_images)} images without annotations:")
    for img_base in sorted(unannotated_images):
        img_path = os.path.join(image_dir, f"{img_base}.jpg")
        if dry_run:
            print(f"Would delete: {img_path}")
        else:
            try:
                os.remove(img_path)
                print(f"Deleted: {img_path}")
            except OSError as e:
                print(f"Error deleting {img_path}: {e}")

    if dry_run:
        print("\nDry run complete. No files were deleted. Use --dry_run off to delete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean up image files without corresponding annotations.")
    parser.add_argument("--image_dir", default="/data/mskscratch/users/ghoyer/Precision_Air/0403/face_videos/frame_images_part3",
                        help="Directory containing image files")
    parser.add_argument("--label_dir", default="/data/mskscratch/users/ghoyer/Precision_Air/0403/face_videos/frame_labels_part3",
                        help="Directory containing annotation files")
    parser.add_argument("--dry_run", type=str, default="on", choices=["on", "off"],
                        help="Dry run mode: 'on' to list only, 'off' to delete")
    args = parser.parse_args()

    # Convert dry_run string to boolean
    dry_run = args.dry_run == "on"

    print(f"Running with image_dir={args.image_dir}, label_dir={args.label_dir}, dry_run={dry_run}")
    clean_unannotated_images(args.image_dir, args.label_dir, dry_run)

# python -m src.dataset_preprocessing.iterative_quality_check.clean_unannotated_images