import os
import csv
import sys
import cv2
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
import logging

import torch
import torch.nn as nn
from torchvision.models import resnet18
import torchvision.transforms as transforms

try:
    from src.procedural_frame_classification import get_model
    PIPELINE_MODE = True
except ImportError:
    PIPELINE_MODE = False

from src.utils import load_config, setup_logging

# Retrieve a logger for the module
logger = logging.getLogger(__name__)

# Set device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -----------------------------
# Model Definition (for App Mode)
# -----------------------------
def get_app_model(pretrained=True, freeze_backbone=False):
    """Define the model for app mode (embedded to avoid dependencies)."""
    model = resnet18(weights=pretrained)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace the final layer for binary classification
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(model.fc.in_features, 2)
    )
    return model

# -----------------------------
# Model Loading and Preparation
# -----------------------------
def load_model(model_path, device, app_mode=False):
    """Load the model, either from pipeline or app mode."""
    if app_mode or not PIPELINE_MODE:
        model = get_app_model().to(device)
    else:
        model = get_model().to(device)

    state_dict = torch.load(model_path, map_location=device)
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return model, transform

# -----------------------------
# Frame Prediction and Video Processing
# -----------------------------
def is_procedural_frame(frame, model, transform, confidence_threshold=0.5):
    """Determine if a frame is procedural based on model prediction."""
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = transforms.ToPILImage()(img)
    input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        procedural_prob = probs[0, 1].item()  # Probability of procedural class (1)

    return procedural_prob >= confidence_threshold, procedural_prob

def trim_video(input_video_path, output_video_path, model, transform, threshold=5, confidence_threshold=0.5, smoothing_window=5, app_mode=False):
    """Trim a video by identifying procedural frames and saving the trimmed output."""
    cap = cv2.VideoCapture(input_video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_probs = []
    frame_idx = 0
    print("Analyzing frames...")
    if app_mode:
        logger.info("Analyzing frames...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        is_proc, proc_prob = is_procedural_frame(frame, model, transform, confidence_threshold=confidence_threshold)
        frame_probs.append((frame_idx, is_proc, proc_prob))
        frame_idx += 1

    cap.release()

    if not frame_probs:
        msg = f"No frames detected for {input_video_path}."
        print(msg)
        if app_mode:
            logger.warning(msg)
        return

    # Smooth the predictions
    smoothed_procedural = []
    for i in range(len(frame_probs)):
        start = max(0, i - smoothing_window // 2)
        end = min(len(frame_probs), i + smoothing_window // 2 + 1)
        window_probs = [frame_probs[j][2] for j in range(start, end)]
        avg_prob = np.mean(window_probs)
        smoothed_procedural.append(avg_prob >= confidence_threshold)

    # Find procedural frames after smoothing
    procedural_frames = [idx for idx, (frame_idx, _, _) in enumerate(frame_probs) if smoothed_procedural[idx]]

    # Log frame-by-frame predictions
    log_dir = os.path.join(os.path.dirname(output_video_path), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{os.path.splitext(os.path.basename(input_video_path))[0]}_predictions.csv")
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["frame_idx", "is_procedural", "procedural_prob", "smoothed_is_procedural"])
        for i, (frame_idx, is_proc, proc_prob) in enumerate(frame_probs):
            writer.writerow([frame_idx, is_proc, proc_prob, smoothed_procedural[i]])
    msg = f"Saved frame predictions to {log_file}"
    print(msg)
    if app_mode:
        logger.info(msg)

    if not procedural_frames:
        msg = f"No procedural frames detected for {input_video_path} after smoothing."
        print(msg)
        if app_mode:
            logger.warning(msg)
        return

    start_frame = max(procedural_frames[0] - threshold, 0)
    end_frame = procedural_frames[-1] + threshold

    cap = cv2.VideoCapture(input_video_path)
    out = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (frame_width, frame_height)
    )

    msg = f"Trimming video from frame {start_frame} to frame {end_frame} for {input_video_path}"
    print(msg)
    if app_mode:
        logger.info(msg)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if start_frame <= frame_idx <= end_frame:
            out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    msg = f"Trimmed video saved at: {output_video_path}"
    print(msg)
    if app_mode:
        logger.info(msg)

def process_single_video(video_path, output_dir, model, transform, threshold=5, confidence_threshold=0.5, smoothing_window=5, app_mode=False, add_timestamp=False):
    """Process a single video and return the output path."""
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    if add_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_video_path = os.path.join(output_dir, f"{base_name}_{timestamp}.mp4")
    else:
        output_video_path = os.path.join(output_dir, f"{base_name}.mp4")
    
    os.makedirs(output_dir, exist_ok=True)
    
    if os.path.exists(output_video_path):
        msg = f"Skipping {video_path} - output file already exists at {output_video_path}"
        print(msg)
        if app_mode:
            logger.info(msg)
        return None

    trim_video(video_path, output_video_path, model, transform, threshold=threshold, confidence_threshold=confidence_threshold, smoothing_window=smoothing_window, app_mode=app_mode)
    return output_video_path

def process_directory(input_dir, output_dir, model, transform, threshold=5, confidence_threshold=0.5, smoothing_window=5, app_mode=False, add_timestamp=False):
    os.makedirs(output_dir, exist_ok=True)
    valid_exts = (".mp4", ".avi")
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(valid_exts):
            video_path = os.path.join(input_dir, filename)
            process_single_video(video_path, output_dir, model, transform, threshold=threshold, confidence_threshold=confidence_threshold, smoothing_window=smoothing_window, app_mode=app_mode, add_timestamp=add_timestamp)

def process_dataframe(csv_path, output_dir, model, transform, max_rows=None, threshold=5, confidence_threshold=0.5, smoothing_window=5, app_mode=False, add_timestamp=False):
    df = pd.read_csv(csv_path)
    if max_rows is not None:
        df = df.head(max_rows)

    processed_count = 0
    for idx, row in df.iterrows():
        if max_rows is not None and processed_count >= max_rows:
            msg = f"Reached max_rows limit of {max_rows}. Stopping processing."
            print(msg)
            if app_mode:
                logger.info(msg)
            break

        video_path = row["Video_path"]
        result = process_single_video(video_path, output_dir, model, transform, threshold=threshold, confidence_threshold=confidence_threshold, smoothing_window=smoothing_window, app_mode=app_mode, add_timestamp=add_timestamp)
        if result is not None:
            processed_count += 1
            msg = f"Processed video {processed_count}/{max_rows if max_rows is not None else 'unlimited'}: {video_path}"
            print(msg)
            if app_mode:
                logger.info(msg)

# -----------------------------
# Model Path Resolution
# -----------------------------
def get_model_path(config, args_weights_path, config_path):
    """Determine the model path based on config, args, or default latest run."""
    inference_config = config.get("inference", {})
    weights_path = args_weights_path or inference_config.get("weights_path")

    if weights_path:
        # If weights_path is not absolute, resolve it relative to output_base_dir
        if not os.path.isabs(weights_path):
            weights_path = os.path.join(config["paths"]["output_base_dir"], weights_path.lstrip("../"))
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Specified weights path not found: {weights_path}")
        return weights_path

    output_base_dir = config["paths"]["output_base_dir"]
    latest_run_file = os.path.join(output_base_dir, "latest_run.txt")
    if not os.path.exists(latest_run_file):
        raise FileNotFoundError("No training run found in latest_run.txt. Please run training first or specify a weights path.")
    with open(latest_run_file, "r") as f:
        run_dir = f.read().strip()
    checkpoint_path = os.path.join(run_dir, "checkpoints/best_model.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Best checkpoint not found at {checkpoint_path}")
    return checkpoint_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automatic video trimming script.")
    parser.add_argument('--input_path', type=str, default=None,
                        help="Path to a video file OR folder containing videos. Optional if using --csv_path or config.")
    parser.add_argument('--csv_path', type=str, default=None,
                        help="Path to a CSV file containing the 'Video_path' column. Optional if using --input_path or config.")
    parser.add_argument('--output_dir', type=str, default=None,
                        help="Directory for saving trimmed videos (overrides config).")
    parser.add_argument('--threshold', type=int, default=None,
                        help="Number of frames before the first detected frame and after the last detected frame (overrides config).")
    parser.add_argument('--max_rows', type=int, default=None,
                        help="Number of rows from CSV to process (optional, overrides config).")
    parser.add_argument('--confidence_threshold', type=float, default=None,
                        help="Confidence threshold for classifying a frame as procedural (overrides config).")
    parser.add_argument('--smoothing_window', type=int, default=None,
                        help="Window size for smoothing frame predictions (overrides config).")
    parser.add_argument('--weights_path', type=str, default=None,
                        help="Path to the model weights file (overrides config).")
    parser.add_argument('--config_file', default="config.yaml",
                        help="Path to config file.")
    parser.add_argument('--app_mode', action='store_true',
                        help="Run in app mode (for UI app, embeds model logic).")
    parser.add_argument('--add_timestamp', action='store_true',
                        help="Add a timestamp to the output video file name.")

    args = parser.parse_args()

    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),  # Up three levels to root/
        "config",
        args.config_file
    )

    try:
        # Load config and set up logging
        config = load_config(config_path)
        logger = setup_logging(config["pipeline"]["log_file"], logger=logging.getLogger(__name__))

        # Determine model path
        model_path = get_model_path(config, args.weights_path, config_path)

        # Load model and transform
        model, transform = load_model(model_path, DEVICE, app_mode=args.app_mode)

        # Determine parameters: use command-line args if provided, otherwise fall back to config
        output_dir = args.output_dir if args.output_dir is not None else config["paths"]["trimmed_vid_dir"]
        threshold = args.threshold if args.threshold is not None else config.get("auto_trim", {}).get("threshold", 5)
        max_rows = args.max_rows if args.max_rows is not None else config.get("auto_trim", {}).get("max_rows", None)
        confidence_threshold = args.confidence_threshold if args.confidence_threshold is not None else config.get("auto_trim", {}).get("confidence_threshold", 0.5)
        smoothing_window = args.smoothing_window if args.smoothing_window is not None else config.get("auto_trim", {}).get("smoothing_window", 5)

        # Determine input_path and csv_path: use command-line args if provided, otherwise fall back to config
        input_path = args.input_path if args.input_path is not None else config.get("auto_trim", {}).get("input_path", None)
        csv_path = args.csv_path if args.csv_path is not None else config.get("auto_trim", {}).get("csv_path", None)

        # Process based on input
        # Priority: command-line args > config file > error if neither provided
        if input_path and os.path.exists(input_path):
            if os.path.isfile(input_path):
                process_single_video(input_path, output_dir, model, transform, threshold=threshold, confidence_threshold=confidence_threshold, smoothing_window=smoothing_window, app_mode=args.app_mode, add_timestamp=args.add_timestamp)
            elif os.path.isdir(input_path):
                process_directory(input_path, output_dir, model, transform, threshold=threshold, confidence_threshold=confidence_threshold, smoothing_window=smoothing_window, app_mode=args.app_mode, add_timestamp=args.add_timestamp)
            else:
                logger.error("input_path is neither a valid file nor a directory.")
                print("input_path is neither a valid file nor a directory.")
        elif csv_path and os.path.isfile(csv_path):
            process_dataframe(csv_path, output_dir, model, transform, max_rows=max_rows, threshold=threshold, confidence_threshold=confidence_threshold, smoothing_window=smoothing_window, app_mode=args.app_mode, add_timestamp=args.add_timestamp)
        else:
            logger.error("Please provide a valid --input_path (file or folder) or a valid --csv_path, either via command-line arguments or in the config file.")
            print("Please provide a valid --input_path (file or folder) or a valid --csv_path, either via command-line arguments or in the config file.")
            raise ValueError("No valid input path or CSV path provided.")

        logger.info("Video trimming complete!")
        print("Video trimming complete!")

    except Exception as e:
        if 'logger' in locals():
            logger.error(f"An error occurred: {e}")
        else:
            print(f"An error occurred before logger was set up: {e}")
        raise

# python -m src.app.auto_trim_video --config_file config.yaml
# python -m src.app.auto_trim_video --input_path /path/to/video.mp4 --output_dir /path/to/output --config_file config.yaml
# python -m src.app.auto_trim_video --csv_path /path/to/videos.csv --output_dir /path/to/output --config_file config.yaml

# python -m src.app.auto_trim_video --input_path /path/to/input_video.mp4 --output_dir trimmed_videos
# Output: trimmed_videos/input_video.mp4

# python -m src.app.auto_trim_video --input_path /path/to/input_video.mp4 --output_dir trimmed_videos --add_timestamp
# Output: trimmed_videos/input_video_20250410_143022.mp4