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
from ultralytics import YOLO
from collections import deque

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
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(model.fc.in_features, 2)
    )
    return model

# -----------------------------
# Model Loading and Preparation
# -----------------------------
def load_classification_model(model_path, device, app_mode=False):
    """Load the classification model."""
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

def load_yolo_model(model_path):
    """Load the YOLO model."""
    return YOLO(model_path)

# -----------------------------
# Frame Prediction
# -----------------------------
def is_procedural_frame(frame, model, transform, confidence_threshold=0.7):
    """Determine if a frame is procedural based on model prediction."""
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = transforms.ToPILImage()(img)
    input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        procedural_prob = probs[0, 1].item()
    return procedural_prob >= confidence_threshold, procedural_prob

# -----------------------------
# Video Trimming
# -----------------------------
def trim_video_for_detection(input_video_path, output_video_path, model, transform, threshold=5, confidence_threshold=0.7, smoothing_window=10, app_mode=False):
    """Trim a video by identifying procedural frames and save the output. Return frame probabilities."""
    cap = cv2.VideoCapture(input_video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    frame_probs = []
    frame_idx = 0
    print("Analyzing frames for trimming...")
    if app_mode:
        logger.info("Analyzing frames for trimming...")

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
        return None, None, None

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
        return None, None, None

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

    return output_video_path, frame_probs, smoothed_procedural

# -----------------------------
# YOLO Detection and Blurring
# -----------------------------
def detect_and_blur_video(
    yolo_model,
    input_video_path,
    output_video_path,
    frame_probs,
    smoothed_procedural,
    conf_adjust_range=(0.3, 0.99),
    procedural_conf_threshold=0.7,
    smoothing_window=10,
    blur_only=False,
    blur_kernel=(51, 51),
    target_classes=None
):
    """Apply YOLO detection and blur regions, using cached frame probabilities for confidence adjustment."""
    cap = None
    out = None
    try:
        if not os.path.exists(input_video_path):
            raise ValueError(f"Input video file does not exist: {input_video_path}")

        class_names = yolo_model.names
        if target_classes is not None:
            valid_class_ids = []
            for cls in target_classes:
                if isinstance(cls, int) and cls in class_names:
                    valid_class_ids.append(cls)
                elif isinstance(cls, str) and cls in class_names.values():
                    cls_id = [k for k, v in class_names.items() if v == cls][0]
                    valid_class_ids.append(cls_id)
                else:
                    print(f"Warning: Ignoring invalid class {cls}. Valid classes are: {class_names}")
            if not valid_class_ids:
                raise ValueError("No valid classes provided in target_classes.")
        else:
            valid_class_ids = None

        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise ValueError(f"Error opening video file: {input_video_path}")

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        if not out.isOpened():
            raise ValueError(f"Error creating output video file: {output_video_path}")

        prob_window = deque(maxlen=smoothing_window)
        frame_count = 0
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Use cached procedural probability
            if frame_idx < len(frame_probs):
                procedural_prob = frame_probs[frame_idx][2]
            else:
                procedural_prob = 0.0  # Default for out-of-range frames

            prob_window.append(procedural_prob)
            smoothed_prob = np.mean(prob_window) if prob_window else procedural_prob

            min_conf, max_conf = conf_adjust_range
            conf_threshold = min_conf + (max_conf - min_conf) * smoothed_prob

            results = yolo_model.predict(frame, conf=conf_threshold, verbose=False)
            output_frame = frame.copy()
            boxes = results[0].boxes

            filtered_boxes = []
            if valid_class_ids is not None:
                for box in boxes:
                    cls_id = int(box.cls[0])
                    if cls_id in valid_class_ids:
                        filtered_boxes.append(box)
            else:
                filtered_boxes = boxes

            for box in filtered_boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame_width, x2), min(frame_height, y2)
                roi = output_frame[y1:y2, x1:x2]
                if roi.size == 0:
                    continue
                blurred_roi = cv2.GaussianBlur(roi, blur_kernel, 0)
                output_frame[y1:y2, x1:x2] = blurred_roi

            if not blur_only:
                results[0].boxes = filtered_boxes
                output_frame = results[0].plot(img=output_frame)

            out.write(output_frame)

            frame_count += 1
            print(f"Processed frame {frame_count}/{total_frames} (conf_threshold={conf_threshold:.3f}, procedural_prob={smoothed_prob:.3f})", end='\r')
            frame_idx += 1

        print(f"\nBlurred video saved to: {output_video_path}")
        return True

    except Exception as e:
        print(f"Error processing video {input_video_path}: {str(e)}")
        return False

    finally:
        if cap is not None and cap.isOpened():
            cap.release()
        if out is not None and out.isOpened():
            out.release()

# -----------------------------
# Combined Pipeline
# -----------------------------
def process_single_video_with_blur(
    video_path,
    output_dir,
    class_model,
    class_transform,
    yolo_model,
    threshold=5,
    confidence_threshold=0.7,
    smoothing_window=10,
    app_mode=False,
    add_timestamp=False,
    yolo_conf_adjust_range=(0.3, 0.99),
    yolo_blur_only=False,
    yolo_blur_kernel=(51, 51),
    yolo_target_classes=None
):
    """Process a single video: trim and then apply YOLO detection and blurring."""
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    if add_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        trim_output_path = os.path.join(output_dir, f"{base_name}_{timestamp}_trimmed.mp4")
        final_output_path = os.path.join(output_dir, f"{base_name}_{timestamp}_processed.mp4")
    else:
        trim_output_path = os.path.join(output_dir, f"{base_name}_trimmed.mp4")
        final_output_path = os.path.join(output_dir, f"{base_name}_processed.mp4")

    os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(final_output_path):
        msg = f"Skipping {video_path} - final output file already exists at {final_output_path}"
        print(msg)
        if app_mode:
            logger.info(msg)
        return None

    # Step 1: Trim the video
    trim_result, frame_probs, smoothed_procedural = trim_video_for_detection(
        video_path,
        trim_output_path,
        class_model,
        class_transform,
        threshold=threshold,
        confidence_threshold=confidence_threshold,
        smoothing_window=smoothing_window,
        app_mode=app_mode
    )

    if trim_result is None:
        msg = f"Skipping YOLO processing for {video_path} due to trimming failure."
        print(msg)
        if app_mode:
            logger.warning(msg)
        return None

    # Step 2: Apply YOLO detection and blurring on trimmed video
    detect_and_blur_video(
        yolo_model,
        trim_output_path,
        final_output_path,
        frame_probs,
        smoothed_procedural,
        conf_adjust_range=yolo_conf_adjust_range,
        procedural_conf_threshold=confidence_threshold,
        smoothing_window=smoothing_window,
        blur_only=yolo_blur_only,
        blur_kernel=yolo_blur_kernel,
        target_classes=yolo_target_classes
    )

    return final_output_path

def process_directory_with_blur(
    input_dir,
    output_dir,
    class_model,
    class_transform,
    yolo_model,
    threshold=5,
    confidence_threshold=0.7,
    smoothing_window=10,
    app_mode=False,
    add_timestamp=False,
    yolo_conf_adjust_range=(0.3, 0.99),
    yolo_blur_only=False,
    yolo_blur_kernel=(51, 51),
    yolo_target_classes=None
):
    os.makedirs(output_dir, exist_ok=True)
    valid_exts = (".mp4", ".avi")
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(valid_exts):
            video_path = os.path.join(input_dir, filename)
            process_single_video_with_blur(
                video_path,
                output_dir,
                class_model,
                class_transform,
                yolo_model,
                threshold=threshold,
                confidence_threshold=confidence_threshold,
                smoothing_window=smoothing_window,
                app_mode=app_mode,
                add_timestamp=add_timestamp,
                yolo_conf_adjust_range=yolo_conf_adjust_range,
                yolo_blur_only=yolo_blur_only,
                yolo_blur_kernel=yolo_blur_kernel,
                yolo_target_classes=yolo_target_classes
            )

def process_dataframe(
    csv_path,
    output_dir,
    class_model,
    class_transform,
    yolo_model,
    max_rows=None,
    threshold=5,
    confidence_threshold=0.7,
    smoothing_window=10,
    app_mode=False,
    add_timestamp=False,
    yolo_conf_adjust_range=(0.3, 0.99),
    yolo_blur_only=False,
    yolo_blur_kernel=(51, 51),
    yolo_target_classes=None
):
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
        result = process_single_video_with_blur(
            video_path,
            output_dir,
            class_model,
            class_transform,
            yolo_model,
            threshold=threshold,
            confidence_threshold=confidence_threshold,
            smoothing_window=smoothing_window,
            app_mode=app_mode,
            add_timestamp=add_timestamp,
            yolo_conf_adjust_range=yolo_conf_adjust_range,
            yolo_blur_only=yolo_blur_only,
            yolo_blur_kernel=yolo_blur_kernel,
            yolo_target_classes=yolo_target_classes
        )
        if result is not None:
            processed_count += 1
            msg = f"Processed video {processed_count}/{max_rows if max_rows is not None else 'unlimited'}: {video_path}"
            print(msg)
            if app_mode:
                logger.info(msg)

# -----------------------------
# Model Path Resolution
# -----------------------------
def get_model_path(config, args_weights_path, config_path, model_type="classification"):
    """Determine the model path based on config, args, or default latest run."""
    inference_config = config.get("inference", {})
    weights_path = args_weights_path or inference_config.get(f"{model_type}_weights_path")

    if weights_path:
        if not os.path.isabs(weights_path):
            weights_path = os.path.join(config["paths"]["output_base_dir"], weights_path.lstrip("../"))
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Specified {model_type} weights path not found: {weights_path}")
        return weights_path

    output_base_dir = config["paths"]["output_base_dir"]
    latest_run_file = os.path.join(output_base_dir, f"latest_{model_type}_run.txt")
    if not os.path.exists(latest_run_file):
        raise FileNotFoundError(f"No {model_type} training run found in latest_{model_type}_run.txt.")
    with open(latest_run_file, "r") as f:
        run_dir = f.read().strip()
    checkpoint_path = os.path.join(run_dir, "checkpoints/best_model.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Best {model_type} checkpoint not found at {checkpoint_path}")
    return checkpoint_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automatic video trimming and detection pipeline.")
    parser.add_argument('--input_path', type=str, default=None,
                        help="Path to a video file or folder containing videos.")
    parser.add_argument('--csv_path', type=str, default=None,
                        help="Path to a CSV file containing the 'Video_path' column.")
    parser.add_argument('--output_dir', type=str, default=None,
                        help="Directory for saving processed videos.")
    parser.add_argument('--threshold', type=int, default=None,
                        help="Frames before first and after last detected frame.")
    parser.add_argument('--max_rows', type=int, default=None,
                        help="Number of rows from CSV to process.")
    parser.add_argument('--confidence_threshold', type=float, default=None,
                        help="Confidence threshold for classifying a frame as procedural.")
    parser.add_argument('--smoothing_window', type=int, default=None,
                        help="Window size for smoothing frame predictions.")
    parser.add_argument('--class_weights_path', type=str, default=None,
                        help="Path to the classification model weights file.")
    parser.add_argument('--yolo_weights_path', type=str, default=None,
                        help="Path to the YOLO model weights file.")
    parser.add_argument('--yolo_conf_adjust_range', type=float, nargs=2, default=None,
                        help="Min and max YOLO confidence thresholds (e.g., 0.3 0.99).")
    parser.add_argument('--yolo_blur_only', action='store_true',
                        help="Only blur regions without drawing bounding boxes.")
    parser.add_argument('--yolo_blur_kernel', type=int, nargs=2, default=(51, 51),
                        help="Kernel size for Gaussian blur (e.g., 51 51).")
    parser.add_argument('--yolo_target_classes', type=str, nargs='*', default=None,
                        help="List of YOLO class IDs or names to process.")
    parser.add_argument('--config_file', default="config.yaml",
                        help="Path to config file.")
    parser.add_argument('--app_mode', action='store_true',
                        help="Run in app mode (for UI app).")
    parser.add_argument('--add_timestamp', action='store_true',
                        help="Add a timestamp to the output video file name.")

    args = parser.parse_args()

    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "config",
        args.config_file
    )

    try:
        config = load_config(config_path)
        logger = setup_logging(config["pipeline"]["log_file"], logger=logging.getLogger(__name__))

        # Load models
        class_model_path = get_model_path(config, args.class_weights_path, config_path, model_type="classification")
        class_model, class_transform = load_classification_model(class_model_path, DEVICE, app_mode=args.app_mode)
        yolo_model_path = get_model_path(config, args.yolo_weights_path, config_path, model_type="yolo")
        yolo_model = load_yolo_model(yolo_model_path) #YOLO(yolo_model_path)

        # Determine parameters
        output_dir = args.output_dir if args.output_dir is not None else config["paths"]["trimmed_vid_dir"]
        threshold = args.threshold if args.threshold is not None else config.get("auto_trim", {}).get("threshold", 5)
        max_rows = args.max_rows if args.max_rows is not None else config.get("auto_trim", {}).get("max_rows", None)
        confidence_threshold = args.confidence_threshold if args.confidence_threshold is not None else config.get("auto_trim", {}).get("confidence_threshold", 0.7)
        smoothing_window = args.smoothing_window if args.smoothing_window is not None else config.get("auto_trim", {}).get("smoothing_window", 10)
        yolo_conf_adjust_range = tuple(args.yolo_conf_adjust_range) if args.yolo_conf_adjust_range is not None else config.get("yolo", {}).get("conf_adjust_range", (0.3, 0.99))
        yolo_blur_only = args.yolo_blur_only if args.yolo_blur_only else config.get("yolo", {}).get("blur_only", False)
        yolo_blur_kernel = tuple(args.yolo_blur_kernel) if args.yolo_blur_kernel is not None else config.get("yolo", {}).get("blur_kernel", (51, 51))
        yolo_target_classes = args.yolo_target_classes if args.yolo_target_classes is not None else config.get("yolo", {}).get("target_classes", None)

        input_path = args.input_path if args.input_path is not None else config.get("auto_trim", {}).get("input_path", None)
        csv_path = args.csv_path if args.csv_path is not None else config.get("auto_trim", {}).get("csv_path", None)

        if input_path and os.path.exists(input_path):
            if os.path.isfile(input_path):
                process_single_video_with_blur(
                    input_path,
                    output_dir,
                    class_model,
                    class_transform,
                    yolo_model,
                    threshold=threshold,
                    confidence_threshold=confidence_threshold,
                    smoothing_window=smoothing_window,
                    app_mode=args.app_mode,
                    add_timestamp=args.add_timestamp,
                    yolo_conf_adjust_range=yolo_conf_adjust_range,
                    yolo_blur_only=yolo_blur_only,
                    yolo_blur_kernel=yolo_blur_kernel,
                    yolo_target_classes=yolo_target_classes
                )
            elif os.path.isdir(input_path):
                process_directory_with_blur(
                    input_path,
                    output_dir,
                    class_model,
                    class_transform,
                    yolo_model,
                    threshold=threshold,
                    confidence_threshold=confidence_threshold,
                    smoothing_window=smoothing_window,
                    app_mode=args.app_mode,
                    add_timestamp=args.add_timestamp,
                    yolo_conf_adjust_range=yolo_conf_adjust_range,
                    yolo_blur_only=yolo_blur_only,
                    yolo_blur_kernel=yolo_blur_kernel,
                    yolo_target_classes=yolo_target_classes
                )
            else:
                logger.error("input_path is neither a valid file nor a directory.")
                print("input_path is neither a valid file nor a directory.")
        elif csv_path and os.path.isfile(csv_path):
            process_dataframe(
                csv_path,
                output_dir,
                class_model,
                class_transform,
                yolo_model,
                max_rows=max_rows,
                threshold=threshold,
                confidence_threshold=confidence_threshold,
                smoothing_window=smoothing_window,
                app_mode=args.app_mode,
                add_timestamp=args.add_timestamp,
                yolo_conf_adjust_range=yolo_conf_adjust_range,
                yolo_blur_only=yolo_blur_only,
                yolo_blur_kernel=yolo_blur_kernel,
                yolo_target_classes=yolo_target_classes
            )
        else:
            logger.error("Please provide a valid --input_path or --csv_path.")
            print("Please provide a valid --input_path or --csv_path.")
            raise ValueError("No valid input path or CSV path provided.")

        logger.info("Video processing complete!")
        print("Video processing complete!")

    except Exception as e:
        if 'logger' in locals():
            logger.error(f"An error occurred: {e}")
        else:
            print(f"An error occurred before logger was set up: {e}")
        raise

# python -m src.app.auto_trim_and_detect \
#     --csv_path /data/mskscratch/users/ghoyer/Precision_Air/air_loc_0308.csv \
#     --output_dir /data/mskscratch/users/ghoyer/Precision_Air/temp_output/trimmed_videos \
#     --threshold 5 \
#     --confidence_threshold 0.7 \
#     --smoothing_window 10 \
#     --class_weights_path /data/mskscratch/users/ghoyer/Precision_Air/VLS_auto_trimmer/outputs/run_20250417_015644/checkpoints/best_model.pt \
#     --yolo_weights_path /data/mskscratch/users/ghoyer/Precision_Air/VLS_auto_trimmer/face-detection-yolov8/yolov8_run_20250414_225903/weights/best.pt \
#     --yolo_conf_adjust_range 0.3 0.99 \
#     --max_rows 5 \
#     --app_mode

# -----------------------------
