
import cv2
import os
import glob
import argparse
from pathlib import Path
import logging

from src.utils import load_config, setup_logging

def extract_frames(input_video_dir, output_frames_dir, frame_interval=10, class_name=None):
    """
    Extract frames from videos in the input directory and save them to the output directory.
    
    Args:
        input_video_dir (str): Directory containing input videos.
        output_frames_dir (str): Directory to save extracted frames.
        frame_interval (int): Extract every nth frame.
        class_name (str): Class name (e.g., 'procedural', 'nonprocedural') for logging.
    """
    # Ensure output directory exists
    Path(output_frames_dir).mkdir(parents=True, exist_ok=True)

    # Get all video files (.mp4 or .avi)
    video_paths = glob.glob(f"{input_video_dir}/*.mp4") + glob.glob(f"{input_video_dir}/*.avi")
    if not video_paths:
        logging.info(f"No videos found in {input_video_dir}")
        print(f"No videos found in {input_video_dir}")
        return 0

    total_frames_extracted = 0

    for video_path in video_paths:
        cap = cv2.VideoCapture(video_path)
        success, frame_number = True, 0
        
        # Get the base filename without extension
        video_filename = os.path.splitext(os.path.basename(video_path))[0]
        video_frame_count = 0  # Frame counter specific to this video
        
        while success:
            success, frame = cap.read()
            if not success:
                break
            if frame_number % frame_interval == 0:
                # Create filename using original video name and frame number
                frame_filename = os.path.join(output_frames_dir, 
                                            f"{video_filename}_frame{video_frame_count:05d}.jpeg")
                cv2.imwrite(frame_filename, frame)
                video_frame_count += 1
            frame_number += 1

        cap.release()
        total_frames_extracted += video_frame_count
        logging.info(f"Extracted {video_frame_count} frames from {video_filename} ({class_name})")
        print(f"Extracted {video_frame_count} frames from {video_filename} ({class_name})")

    logging.info(f"Total frames extracted for {class_name}: {total_frames_extracted}")
    logging.info(f"Frames saved to {output_frames_dir}")
    print(f"Total frames extracted for {class_name}: {total_frames_extracted}")
    print(f"Frames saved to {output_frames_dir}")
    return total_frames_extracted

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from videos.")
    parser.add_argument("--input_dir", default=None, help="Directory containing input videos (overrides config)")
    parser.add_argument("--output_dir", default=None, help="Directory to save extracted frames (overrides config)")
    parser.add_argument("--class_name", default=None, help="Class name for logging (e.g., 'procedural', 'nonprocedural')")
    parser.add_argument("--frame_interval", type=int, default=None, help="Extract every nth frame (overrides config)")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),  # Up three levels to root/
        "config",
        args.config
    )

    try:
        # Load config and set up logging
        config = load_config(config_path)
        logger = setup_logging(config["pipeline"]["log_file"], logger=logging.getLogger(__name__))

        # Determine arguments: use command-line args if provided, otherwise fall back to config
        input_dir = args.input_dir if args.input_dir is not None else config["paths"]["new_data_dir"]
        output_dir = args.output_dir if args.output_dir is not None else config["paths"]["temp_frames_dir"]
        class_name = args.class_name if args.class_name is not None else "unknown"
        frame_interval = args.frame_interval if args.frame_interval is not None else config["frame_extraction"]["frame_interval"]

        # If class_name is provided or present in config, append it to the directories
        if class_name != "unknown":
            input_dir = os.path.join(input_dir, class_name)
            output_dir = os.path.join(output_dir, class_name)

        extract_frames(input_dir, output_dir, frame_interval, class_name)

        logger.info("Extracting frames from video complete!")
        print("\nExtracting frames from video complete!")

    except Exception as e:
        if 'logger' in locals():
            logger.error(f"An error occurred: {e}")
        else:
            print(f"An error occurred before logger was set up: {e}")
        raise

#  python -m src.dataset_preprocessing.extract_frames --config config.yaml --class_name procedural
# python -m src.dataset_preprocessing.extract_frames --input_dir /custom/input --output_dir /custom/output --class_name custom_class --frame_interval 5 --config config.yaml
# python -m src.dataset_preprocessing.extract_frames --input_dir /custom/input --class_name procedural --config config.yaml

