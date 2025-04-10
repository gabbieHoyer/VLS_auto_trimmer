import os
import glob
from collections import defaultdict

def extract_base_name(filename):
    """
    Extract the base video name from a filename by splitting on '_frame'.
    
    Args:
        filename (str): Filename like 'video1_frame000001.jpg'.
    
    Returns:
        str: Base video name (e.g., 'video1').
    """
    base = os.path.splitext(filename)[0]  # Remove .jpg or .jpeg
    return base.split('_frame')[0]  # Take part before '_frame'

def get_base_names_from_dir(directory):
    """
    Get a set of base video names from all .jpg and .jpeg files in a directory.
    
    Args:
        directory (str): Directory path.
    
    Returns:
        set: Set of base video names.
    """
    base_names = set()
    # Match both .jpg and .jpeg files
    for filepath in glob.glob(f"{directory}/*.[jp][pn]g"):
        filename = os.path.basename(filepath)
        base_name = extract_base_name(filename)
        base_names.add(base_name)
    return base_names

def compare_base_video_names(face_images_dir, splits_base_dir):
    """
    Compare base video names from face images with those in train/val/test splits.
    
    Args:
        face_images_dir (str): Directory with new face frame images.
        splits_base_dir (str): Base directory with train/val/test splits.
    """
    # Get base names from face images (check .jpg and .jpeg)
    face_base_names = get_base_names_from_dir(face_images_dir)
    if not face_base_names:
        print(f"No .jpg or .jpeg files found in {face_images_dir}")
        return

    print(f"Found {len(face_base_names)} unique base video names in {face_images_dir}")

    # Get base names from each split
    splits = ['train', 'val', 'test']
    split_base_names = {}
    
    for split in splits:
        # Combine procedural and nonprocedural for each split
        nonproc_dir = os.path.join(splits_base_dir, split, 'nonprocedural')
        proc_dir = os.path.join(splits_base_dir, split, 'procedural')
        
        nonproc_base_names = get_base_names_from_dir(nonproc_dir)
        proc_base_names = get_base_names_from_dir(proc_dir)
        
        # Union of base names in this split
        split_base_names[split] = nonproc_base_names.union(proc_base_names)
        print(f"Found {len(split_base_names[split])} unique base video names in {split} split")

    # Compare: For each face base name, find which splits it appears in
    base_name_to_splits = defaultdict(list)
    for base_name in face_base_names:
        for split in splits:
            if base_name in split_base_names[split]:
                base_name_to_splits[base_name].append(split)

    # Output results
    print("\nComparison Results:")
    if not base_name_to_splits:
        print("No matching base video names found between face images and splits.")
        return

    for base_name, splits in sorted(base_name_to_splits.items()):
        print(f"Base video '{base_name}' from face images found in splits: {', '.join(splits)}")

    # Summary of unmatched base names
    unmatched = face_base_names - set(base_name_to_splits.keys())
    if unmatched:
        print(f"\n{len(unmatched)} base video names in face images not found in any split:")
        for base_name in sorted(unmatched):
            print(f"  {base_name}")

if __name__ == "__main__":
    face_images_dir = "/data/mskscratch/users/ghoyer/Precision_Air/0403/face_videos/frame_images"
    splits_base_dir = "/data/mskscratch/users/ghoyer/Precision_Air/auto_trimmer/data"

    print(f"Comparing base video names between {face_images_dir} and splits in {splits_base_dir}")
    compare_base_video_names(face_images_dir, splits_base_dir)