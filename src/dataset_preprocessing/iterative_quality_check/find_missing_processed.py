import os

def find_missing_processed(input_dir, output_dir):
    # Get list of input files (mp4 only)
    input_files = {os.path.splitext(f)[0] for f in os.listdir(input_dir) if f.endswith('.mp4')}
    
    # Get list of output files, removing '_anon' and anything after it
    output_files = set()
    for f in os.listdir(output_dir):
        if f.endswith('.mp4'):
            base_name = f.split('_anon')[0]
            output_files.add(base_name)
    
    # Find input files without corresponding output
    missing = input_files - output_files
    
    # Print results
    if missing:
        print("Input files missing processed versions:")
        for f in sorted(missing):
            print(f"{f}.mp4")
    else:
        print("All input files have processed versions.")
    
    return missing

if __name__ == "__main__":
    # input_directory = "input"  # Replace with your input folder path
    # output_directory = "output"  # Replace with your output folder path

    input_directory = "/data/mskscratch/users/ghoyer/Precision_Air/temp_output/trimmed_videos"  # Replace with your input folder path
    output_directory = "/data/mskscratch/users/ghoyer/Precision_Air/temp_output/clean_trimmed_videos"  # Replace with your output folder path

    find_missing_processed(input_directory, output_directory)

# python -m src.dataset_preprocessing.iterative_quality_check.find_missing_processed