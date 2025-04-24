import os
import shutil
import argparse

def copy_nth_jpg(input_dir, output_dir, frequency):
    # Ensure input directory exists
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.")
        return
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get list of JPG files in input directory
    jpg_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.jpeg') or f.lower().endswith('.jpeg')]
    jpg_files.sort()  # Sort files for consistent ordering
    
    if not jpg_files:
        print(f"No JPG files found in '{input_dir}'.")
        return
    
    # Copy every nth JPG file
    for i, filename in enumerate(jpg_files, 1):
        if i % frequency == 0:  # Check if it's the nth file
            source_path = os.path.join(input_dir, filename)
            destination_path = os.path.join(output_dir, filename)
            shutil.copy2(source_path, destination_path)
            print(f"Copied: {filename}")
    
    print(f"Completed. Copied every {frequency}th JPG file to '{output_dir}'.")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Copy every nth JPG file from input to output directory.")
    parser.add_argument("input_dir", help="Path to the input directory containing JPG files")
    parser.add_argument("output_dir", help="Path to the output directory for copied JPG files")
    parser.add_argument("frequency", type=int, help="Copy every nth JPG file (e.g., 5 for every 5th file)")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate frequency
    if args.frequency < 1:
        print("Error: Frequency must be a positive integer.")
        return
    
    # Run the copy function
    copy_nth_jpg(args.input_dir, args.output_dir, args.frequency)

if __name__ == "__main__":
    main()

# From root directory:
# python -m src.dataset_preprocessing.copy_nth_jpg
# python -m src.dataset_preprocessing.copy_nth_jpg /path/to/input /path/to/output 5

# From ../dataset_preprocessing directory:
# python copy_nth_jpg.py /data/mskscratch/users/ghoyer/Precision_Air/VLS_auto_trimmer/data/train/procedural /data/mskscratch/users/ghoyer/Precision_Air/0403/face_videos/dataset_yolo_refined5/train/procedural_images 10
# python copy_nth_jpg.py /data/mskscratch/users/ghoyer/Precision_Air/VLS_auto_trimmer/data/val/procedural /data/mskscratch/users/ghoyer/Precision_Air/0403/face_videos/dataset_yolo_refined5/val/procedural_images 10