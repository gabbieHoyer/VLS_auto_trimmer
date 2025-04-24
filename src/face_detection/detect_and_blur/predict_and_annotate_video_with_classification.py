import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from ultralytics import YOLO
from torchvision.models import resnet18
import torch.nn as nn
from collections import deque

# Set device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_classification_model(model_path, pretrained=True, freeze_backbone=False):
    """Define and load the classification model (based on autotrimming script)."""
    model = resnet18(weights=pretrained)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(model.fc.in_features, 2)
    )
    
    state_dict = torch.load(model_path, map_location=DEVICE)
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

def predict_and_annotate_video(
    yolo_model_path,
    classification_model_path,
    input_video_path,
    output_video_path,
    base_conf_threshold=0.5,
    conf_adjust_range=(0.3, 0.7),
    procedural_conf_threshold=0.5,
    smoothing_window=5,
    blur_only=False,
    blur_kernel=(51, 51),
    target_classes=None
):
    """
    Use a trained YOLO model to predict bounding boxes on a video, blur those regions, and save the output video.
    Adjust YOLO confidence threshold based on classification model's procedural probability.

    Args:
        yolo_model_path (str): Path to the trained YOLO model weights (e.g., 'best.pt')
        classification_model_path (str): Path to the trained classification model weights
        input_video_path (str): Path to the input video file
        output_video_path (str): Path to save the output video
        base_conf_threshold (float): Baseline confidence threshold for YOLO predictions
        conf_adjust_range (tuple): (min_conf, max_conf) range for dynamic YOLO confidence adjustment
        procedural_conf_threshold (float): Confidence threshold for classifying a frame as procedural
        smoothing_window (int): Window size for smoothing procedural probabilities
        blur_only (bool): If True, only blur regions without drawing bounding boxes
        blur_kernel (tuple): Kernel size for Gaussian blur
        target_classes (list, optional): List of class IDs or names to process
    """
    cap = None
    out = None
    try:
        # Verify input file exists
        if not os.path.exists(input_video_path):
            raise ValueError(f"Input video file does not exist: {input_video_path}")

        # Load models
        yolo_model = YOLO(yolo_model_path)
        class_model, class_transform = get_classification_model(classification_model_path)
        class_model.to(DEVICE)

        # Get class names from YOLO model
        class_names = yolo_model.names

        # Validate and convert target_classes to a list of class IDs
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

        # Open the input video
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise ValueError(f"Error opening video file: {input_video_path}")

        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        if not out.isOpened():
            raise ValueError(f"Error creating output video file: {output_video_path}")

        # Initialize deque for smoothing procedural probabilities
        prob_window = deque(maxlen=smoothing_window)
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Classify frame as procedural
            _, procedural_prob = is_procedural_frame(frame, class_model, class_transform, procedural_conf_threshold)
            prob_window.append(procedural_prob)
            
            # Compute smoothed procedural probability
            smoothed_prob = np.mean(prob_window) if prob_window else procedural_prob
            
            # Adjust YOLO confidence threshold based on smoothed procedural probability
            # Linearly interpolate between conf_adjust_range based on smoothed_prob
            min_conf, max_conf = conf_adjust_range
            conf_threshold = min_conf + (max_conf - min_conf) * smoothed_prob
            
            # Perform YOLO prediction with dynamic confidence threshold
            results = yolo_model.predict(frame, conf=conf_threshold, verbose=False)

            # Create a copy of the frame to modify
            output_frame = frame.copy()

            # Get bounding boxes from results
            boxes = results[0].boxes

            # Filter boxes by target classes if specified
            filtered_boxes = []
            if valid_class_ids is not None:
                for box in boxes:
                    cls_id = int(box.cls[0])
                    if cls_id in valid_class_ids:
                        filtered_boxes.append(box)
            else:
                filtered_boxes = boxes

            # Apply blur to each filtered bounding box region
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

        print(f"\nVideo saved to: {output_video_path}")
        return True

    except Exception as e:
        print(f"Error processing video {input_video_path}: {str(e)}")
        return False

    finally:
        # Ensure resources are released
        if cap is not None and cap.isOpened():
            cap.release()
        if out is not None and out.isOpened():
            out.release()

if __name__ == "__main__":
    # Example usage
    # yolo_model_path = "path/to/yolo/best.pt"
    # classification_model_path = "path/to/classification/best_model.pt"
    # input_video_path = "path/to/input/video.mp4"
    # output_video_path = "path/to/output/video.mp4"

    yolo_model_path = "/data/mskscratch/users/ghoyer/Precision_Air/VLS_auto_trimmer/face-detection-yolov8/yolov8_run_20250414_225903/weights/best.pt"
    # newest classification model with additional proc cases as well as those earlier non-proc cases:
    classification_model_path = "/data/mskscratch/users/ghoyer/Precision_Air/VLS_auto_trimmer/outputs/run_20250417_015644/checkpoints/best_model.pt"  #"/data/mskscratch/users/ghoyer/Precision_Air/VLS_auto_trimmer/outputs/run_20250417_011634/checkpoints/best_model.pt"

    video_paths = [
        {
            'input': '/data/mskscratch/users/ghoyer/Precision_Air/temp_output/trimmed_videos/20200605_232758.mp4',
            'output': '/data/mskscratch/users/ghoyer/Precision_Air/VLS_preprocessing/test_app2/20200605_232758_class_plus_det5.mp4'
        },
        {
            'input': '/data/mskscratch/users/ghoyer/Precision_Air/VLS_preprocessing/test_app2/20200518-171738.mp4',
            'output': '/data/mskscratch/users/ghoyer/Precision_Air/VLS_preprocessing/test_app2/20200518-171738_class_plus_det5.mp4'
        },
        {
            'input': '/data/mskscratch/users/ghoyer/Precision_Air/VLS_preprocessing/test_app2/20200517-074312.mp4',
            'output': '/data/mskscratch/users/ghoyer/Precision_Air/VLS_preprocessing/test_app2/20200517-074312_class_plus_det5.mp4'
        },
    ]

    try:
        # Track processing results
        successful = 0
        failed = 0
    
        for video in video_paths:
            print(f"\nProcessing video: {video['input']}")

            result = predict_and_annotate_video(
                yolo_model_path=yolo_model_path,
                classification_model_path=classification_model_path,
                input_video_path=video['input'],   
                output_video_path=video['output'],
                base_conf_threshold=0.5,
                conf_adjust_range=(0.3, 0.99),  # Updated to reach 99% confidence
                procedural_conf_threshold=0.5,
                smoothing_window=5,
                blur_only=False,
                blur_kernel=(51, 51),
                target_classes=None
            )

            if result:
                print(f"Successfully processed: {video['output']}")
                successful += 1
            else:
                print(f"Failed to process: {video['output']}")
                failed += 1

        print(f"\nProcessing complete. Successfully processed {successful}/{len(video_paths)} videos. Failed: {failed}")

    except Exception as e:
        print(f"Unexpected error in main loop: {str(e)}")
        raise

    # predict_and_annotate_video(
    #     yolo_model_path=yolo_model_path,
    #     classification_model_path=classification_model_path,
    #     input_video_path=input_video_path,
    #     output_video_path=output_video_path,
    #     base_conf_threshold=0.5,
    #     conf_adjust_range=(0.3, 0.7),
    #     procedural_conf_threshold=0.5,
    #     smoothing_window=5,
    #     blur_only=False,
    #     blur_kernel=(51, 51),
    #     target_classes=None
    # )


# python -m src.face_detection.detect_and_blur.predict_and_annotate_video_with_classification