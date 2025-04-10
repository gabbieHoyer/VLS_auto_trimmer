import cv2
import os
import glob
import argparse
from pathlib import Path
import logging

from src.utils import load_config, setup_logging

def extract_frames(input_video_dir, output_frames_dir, extract_fps=None, quality=95, class_name="face"):
    """
    Extract frames from videos at their native FPS (all frames) by default, or at a specified FPS.
    
    Args:
        input_video_dir (str): Directory containing input videos.
        output_frames_dir (str): Directory to save extracted frames.
        extract_fps (float, optional): Desired frames per second to extract. If None, uses video's native FPS (all frames).
        quality (int): JPEG quality (0-100, higher is better).
        class_name (str): Class name (e.g., 'face') for logging and organization.
    """
    Path(output_frames_dir).mkdir(parents=True, exist_ok=True)
    video_paths = glob.glob(f"{input_video_dir}/*.mp4") + glob.glob(f"{input_video_dir}/*.avi")
    if not video_paths:
        logging.info(f"No videos found in {input_video_dir}")
        print(f"No videos found in {input_video_dir}")
        return 0

    total_frames_extracted = 0

    for video_path in video_paths:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error(f"Failed to open video: {video_path}")
            continue

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_filename = os.path.splitext(os.path.basename(video_path))[0]

        if extract_fps is None:
            # Default to video's native FPS (extract all frames)
            effective_fps = video_fps
            frame_interval = 1
        else:
            # Use specified FPS, capped at video's FPS
            effective_fps = min(extract_fps, video_fps)
            frame_interval = max(1, int(round(video_fps / effective_fps)))

        logging.info(f"Video: {video_filename}, Native FPS: {video_fps}, Extracting at {effective_fps} FPS (interval: {frame_interval})")

        success, frame_number = True, 0
        video_frame_count = 0

        while success:
            success, frame = cap.read()
            if not success:
                break
            if frame_number % frame_interval == 0:
                frame_filename = os.path.join(output_frames_dir, 
                                            f"{video_filename}_frame{video_frame_count:06d}.jpg")
                cv2.imwrite(frame_filename, frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
                video_frame_count += 1
            frame_number += 1

        cap.release()
        total_frames_extracted += video_frame_count
        logging.info(f"Extracted {video_frame_count} frames from {video_filename} ({class_name})")
        print(f"Extracted {video_frame_count} frames from {video_filename} ({class_name})")

    logging.info(f"Total frames extracted for {class_name}: {total_frames_extracted}")
    print(f"Total frames extracted for {class_name}: {total_frames_extracted}")
    return total_frames_extracted

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract high-resolution frames at native video FPS by default.")
    parser.add_argument("--input_dir", required=True, help="Directory containing input videos")
    parser.add_argument("--output_dir", required=True, help="Directory to save extracted frames")
    parser.add_argument("--extract_fps", type=float, default=None, 
                        help="Frames per second to extract (default: video's native FPS, i.e., all frames)")
    parser.add_argument("--quality", type=int, default=95, 
                        help="JPEG quality (0-100, default: 95)")
    parser.add_argument("--config", default="face_config.yaml", help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config["pipeline"]["log_file"])

    extract_frames(args.input_dir, args.output_dir, extract_fps=args.extract_fps, 
                   quality=args.quality, class_name="face")



# Default (All Frames at Native FPS):

# python extract_face_frames.py --input_dir "path/to/face_videos" --output_dir "dataset_dir/train/face" --quality 100

# 30 FPS video: Extracts all 30 frames per second.
# 34 FPS video: Extracts all 34 frames per second.
# Frames saved at maximum JPEG quality.


# Specify Lower FPS (Optional):

# python extract_face_frames.py --input_dir "path/to/face_videos" --output_dir "dataset_dir/train/face" --extract_fps 10

# Extracts 10 FPS, skipping frames as needed.


# Combining Both Goals
# To maximize frame extraction (all frames) and quality:

# python extract_face_frames.py --input_dir "path/to/face_videos" --output_dir "dataset_dir/train/face" --quality 100

# Result: Every frame is extracted (30 or 34 FPS depending on the video), saved as JPGs with maximum quality.


# Alternative: Lossless Saving
# If you want zero quality loss (at the cost of larger files), replace JPG with PNG:

# Modify the cv2.imwrite line to:

# frame_filename = frame_filename.replace('.jpg', '.png')
# cv2.imwrite(frame_filename, frame)

# PNG ignores the quality parameter and saves losslessly.

#------------

# Extract All Frames (Maximum Density)
# python extract_face_frames.py --input_dir "path/to/face_videos" --output_dir "dataset_dir/train/face"

# This extracts every frame from each video, saved as high-quality JPGs (e.g., video1_frame000001.jpg).

# Extract at a Specific FPS (e.g., 10 FPS)
# python extract_face_frames.py --input_dir "path/to/face_videos" --output_dir "dataset_dir/train/face" --extract_fps 10

# Extracts 10 frames per second, adjusting the interval based on the video’s FPS (e.g., if video FPS is 30, saves every 3rd frame).

# Adjust Quality
# python extract_face_frames.py --input_dir "path/to/face_videos" --output_dir "dataset_dir/train/face" --quality 100

# Saves at maximum JPEG quality (100) for minimal compression loss.


# python extract_face_frames.py --input_dir "/data/mskscratch/users/ghoyer/Precision_Air/0403/face_videos/videos/face_trimmed_videos" --output_dir "/data/mskscratch/users/ghoyer/Precision_Air/0403/face_videos/frame_images" --quality 100

# python extract_face_frames.py --input_dir "/data/mskscratch/users/ghoyer/Precision_Air/0403/face_videos/videos_part2/face_trimmed_videos" --output_dir "/data/mskscratch/users/ghoyer/Precision_Air/0403/face_videos/frame_images_part2" --quality 100

# python extract_face_frames.py --input_dir "/data/mskscratch/users/ghoyer/Precision_Air/0403/face_videos/videos_part3/face_trimmed_videos" --output_dir "/data/mskscratch/users/ghoyer/Precision_Air/0403/face_videos/frame_images_part3" --quality 100