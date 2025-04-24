from ultralytics import YOLO
import cv2
import numpy as np
import os

def predict_and_annotate_video(model_path, input_video_path, output_video_path, conf_threshold=0.5, blur_only=False, blur_kernel=(51, 51), target_classes=None):
    """
    Use a trained YOLO model to predict bounding boxes on a video, blur those regions, and save the output video.
    Optionally include bounding box annotations and filter by specific classes.

    Args:
        model_path (str): Path to the trained YOLO model weights (e.g., 'best.pt')
        input_video_path (str): Path to the input video file
        output_video_path (str): Path to save the output video
        conf_threshold (float): Confidence threshold for predictions
        blur_only (bool): If True, only blur regions without drawing bounding boxes; if False, include both
        blur_kernel (tuple): Kernel size for Gaussian blur (larger values = more blur)
        target_classes (list, optional): List of class IDs (int) or class names (str) to process. If None, process all classes.
    """
    cap = None
    out = None
    try:
        # Verify input file exists
        if not os.path.exists(input_video_path):
            raise ValueError(f"Input video file does not exist: {input_video_path}")

        # Load the trained YOLO model
        model = YOLO(model_path)

        # Get class names from the model
        class_names = model.names

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

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Perform prediction on the frame
            results = model.predict(frame, conf=conf_threshold, verbose=False)

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
            print(f"Processed frame {frame_count}/{total_frames}", end='\r')

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
        # Removed cv2.destroyAllWindows() since no windows are being displayed

if __name__ == "__main__":
    try:
        # model_path = '/data/mskscratch/users/ghoyer/Precision_Air/VLS_auto_trimmer/face-detection-yolov8/yolov8_run_20250415_192503/weights/best.pt'
        # model_path='/data/mskscratch/users/ghoyer/Precision_Air/VLS_auto_trimmer/face-detection-yolov8/yolov8_run_20250415_213507/weights/best.pt' #neg classes not labeled with box

        model_path='/data/mskscratch/users/ghoyer/Precision_Air/VLS_auto_trimmer/face-detection-yolov8/yolov8_run_20250416_144534/weights/best.pt'  # the image sizes were standardized but probs still sucks :,)
        video_paths = [
            {
                'input': '/data/mskscratch/users/ghoyer/Precision_Air/temp_output/trimmed_videos/20200605_232758.mp4',
                'output': '/data/mskscratch/users/ghoyer/Precision_Air/VLS_preprocessing/test_app2/20200605_232758_face10_50conf_blur.mp4'
            },
            {
                'input': '/data/mskscratch/users/ghoyer/Precision_Air/VLS_preprocessing/test_app2/20200405-230345_20250407_234013.mp4',
                'output': '/data/mskscratch/users/ghoyer/Precision_Air/VLS_preprocessing/test_app2/20200405-230345_20250407_234013_face10_50conf_blur.mp4'
            },
            {
                'input': '/data/mskscratch/users/ghoyer/Precision_Air/VLS_preprocessing/test_app2/20200329-051346_20250407_233451.mp4',
                'output': '/data/mskscratch/users/ghoyer/Precision_Air/VLS_preprocessing/test_app2/20200329-051346_20250407_233451_face10_50conf_blur.mp4'
            },
            {
                'input': '/data/mskscratch/users/ghoyer/Precision_Air/temp_output/trimmed_videos/20200611_144511.mp4',
                'output': '/data/mskscratch/users/ghoyer/Precision_Air/VLS_preprocessing/test_app2/20200611_144511_face10_50conf_blur.mp4'
            }
        ]

        target_classes = ['patient', 'face', 'clinician']

        # Track processing results
        successful = 0
        failed = 0

        for video in video_paths:
            print(f"\nProcessing video: {video['input']}")
            result = predict_and_annotate_video(
                model_path,
                video['input'],
                video['output'],
                conf_threshold=0.5,
                blur_only=False,
                blur_kernel=(51, 51),
                target_classes=target_classes
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


# python -m src.face_detection.detect_and_blur.blurred_face_detect

