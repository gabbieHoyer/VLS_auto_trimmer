import os
from pathlib import Path

def cleanup_unpaired_labels(parent_dir, dry_run=True):
    """
    Identifies and optionally deletes .txt label files that don't have corresponding .jpg image files.
    
    Args:
        parent_dir (str): Path to parent directory containing 'images' and 'labels' subfolders
        dry_run (bool): If True, only prints files to be deleted; if False, deletes them
    
    Returns:
        list: List of label files that would be/have been deleted
    """
    # Define paths
    parent_path = Path(parent_dir)
    images_dir = parent_path / 'images'
    labels_dir = parent_path / 'labels'
    
    # Check if directories exist
    if not images_dir.exists():
        print(f"Error: Images directory {images_dir} does not exist")
        return []
    if not labels_dir.exists():
        print(f"Error: Labels directory {labels_dir} does not exist")
        return []
    
    # Get sets of filenames (without extensions)
    image_files = {f.stem for f in images_dir.glob('*.jpg')}
    label_files = {f.stem for f in labels_dir.glob('*.txt')}
    
    # Find labels without corresponding images
    unpaired_labels = label_files - image_files
    files_to_delete = [labels_dir / f"{name}.txt" for name in unpaired_labels]
    
    if not files_to_delete:
        print("No unpaired label files found")
        return []
    
    if dry_run:
        print("Dry run: The following label files would be deleted:")
        for file in files_to_delete:
            print(f"  {file}")
    else:
        print("Deleting unpaired label files:")
        for file in files_to_delete:
            try:
                file.unlink()
                print(f"  Deleted: {file}")
            except Exception as e:
                print(f"  Error deleting {file}: {e}")
    
    return [str(f) for f in files_to_delete]

# Example usage:
if __name__ == "__main__":
    # Dry run
    # path = "/data/mskscratch/users/ghoyer/Precision_Air/0403/face_videos/dataset_yolo_refined/train"
    path = "/data/mskscratch/users/ghoyer/Precision_Air/0403/face_videos/dataset_yolo_refined/val"
    
    # cleanup_unpaired_labels(path, dry_run=True)
    cleanup_unpaired_labels(path, dry_run=False)
    
    # Actual deletion
    # cleanup_unpaired_labels("/path/to/parent/directory", dry_run=False)

# python -m src.dataset_preprocessing.iterative_quality_check.cleanup_unpaired_labels