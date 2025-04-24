import os
import glob

def update_labels(input_dir, class_mapping):
    """
    Update class indices in YOLO label .txt files and overwrite the original files.
    
    Args:
        input_dir (str): Directory containing the .txt label files.
        class_mapping (dict): Dictionary mapping old class indices to new class indices.
    """
    # Ensure input directory exists
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory {input_dir} does not exist.")
    
    # Find all .txt files in the input directory
    txt_files = glob.glob(os.path.join(input_dir, "*.txt"))
    
    if not txt_files:
        print(f"No .txt files found in {input_dir}.")
        return
    
    # Process each .txt file
    for txt_file in txt_files:
        updated_lines = []
        with open(txt_file, 'r') as f:
            lines = f.readlines()
        
        # Update class indices in each line
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Split the line into components (class_id, x_center, y_center, width, height)
            parts = line.split()
            if len(parts) < 5:
                print(f"Skipping malformed line in {txt_file}: {line}")
                continue
            
            try:
                old_class = int(parts[0])
                # Map old class to new class
                if old_class in class_mapping:
                    new_class = class_mapping[old_class]
                    # Replace old class with new class, keep other parts unchanged
                    updated_line = f"{new_class} {' '.join(parts[1:])}"
                    updated_lines.append(updated_line)
                else:
                    print(f"Warning: Class {old_class} in {txt_file} not found in class_mapping. Skipping line.")
            except ValueError:
                print(f"Skipping invalid class ID in {txt_file}: {line}")
                continue
        
        # Overwrite the original file with updated lines
        with open(txt_file, 'w') as f:
            for line in updated_lines:
                f.write(line + '\n')
        
        print(f"Updated and overwrote {txt_file} ({len(updated_lines)} lines)")

def main():
    # Define the new class order (for reference)
    new_class_order = [
        "epiglottis",
        "vocal cords",
        "arytenoid",
        "trachea rings",
        "esophogous",
        "introducer",
        "endotracheal tube",
        "balloon",
        "blood",
        "NG tube",
        "person",
        "face",
        "patient",
        "clinician"
    ]
    
    # Define class mapping for the dataset with 4 classes
    class_mapping = {
        0: new_class_order.index("person"),   # 0 -> 10
        1: new_class_order.index("face"),     # 1 -> 11
        2: new_class_order.index("patient"),  # 2 -> 12
        3: new_class_order.index("clinician") # 3 -> 13
    }
    
    # Define directory containing the .txt label files to update
    # label_dir = "path/to/dataset/labels"  # Update with your dataset's labels directory
    label_dir = "/data/mskscratch/users/ghoyer/Precision_Air/0403/face_videos/dataset_yolo_refined4/train/labels"

    # Update labels
    print("Processing label files...")
    update_labels(label_dir, class_mapping)
    
    print("Label update complete.")
    print("Next steps: Combine the updated label files with the medical dataset's label files into your train/val splits.")
    print("Update your YOLOv8 data.yaml with the new class list:")
    print("names:")
    for i, name in enumerate(new_class_order):
        print(f"  {i}: {name}")

if __name__ == "__main__":
    main()