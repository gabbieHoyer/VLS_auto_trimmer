import os

def rename_files(input_dir):
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.jpeg'):
            name, ext = os.path.splitext(filename)
            new_name = f"{name}_p2{ext}"
            os.rename(
                os.path.join(input_dir, filename),
                os.path.join(input_dir, new_name)
            )
            print(f"Renamed: {filename} -> {new_name}")

if __name__ == "__main__":
    # input_directory = "path/to/your/directory"  # Replace with your directory path
    # input_directory = "/data/mskscratch/users/ghoyer/Precision_Air/0403/04162025_round_classification_vids_raw/nonprocedural_frames2/nonprocedural"  # Replace with your directory path
    input_directory = "/data/mskscratch/users/ghoyer/Precision_Air/0403/04162025_round_classification_vids_raw/procedural_frames/procedural"  # Replace with your directory path
    
    rename_files(input_directory)

# python -m src.dataset_preprocessing.format_standardization.rename_file