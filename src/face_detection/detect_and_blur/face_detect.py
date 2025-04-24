from ultralytics import YOLO
import cv2
import os

def predict_and_annotate_video(model_path, input_video_path, output_video_path, conf_threshold=0.5):
    """
    Use a trained YOLO model to predict bounding boxes on a video and save the annotated video.
    
    Args:
        model_path (str): Path to the trained YOLO model weights (e.g., 'best.pt')
        input_video_path (str): Path to the input video file
        output_video_path (str): Path to save the annotated output video
        conf_threshold (float): Confidence threshold for predictions
    """

    cap = None
    out = None
    try:
        # Verify input file exists
        if not os.path.exists(input_video_path):
            raise ValueError(f"Input video file does not exist: {input_video_path}")

        # Load the trained YOLO model
        model = YOLO(model_path)
        
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
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
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
            
            # Draw bounding boxes and labels on the frame
            annotated_frame = results[0].plot()  # Ultralytics provides a built-in plot method
            
            # Write the annotated frame to the output video
            out.write(annotated_frame)
            
            frame_count += 1
            print(f"Processed frame {frame_count}/{total_frames}", end='\r')
        
        print(f"\nAnnotated video saved to: {output_video_path}")

    except Exception as e:
        print(f"Error processing video {input_video_path}: {str(e)}")
        return False

    finally:
        # Ensure resources are released
        if cap is not None and cap.isOpened():
            cap.release()
        if out is not None and out.isOpened():
            out.release()

def main():
    # Define paths and parameters
    # model_path = 'path/to/your/trained_model.pt'  # Replace with your model weight file
    # input_video_path = 'path/to/input_video.mp4'   # Replace with your input video
    # output_video_path = 'path/to/output_video.mp4' # Replace with desired output path

    try:
        model_path='/data/mskscratch/users/ghoyer/Precision_Air/VLS_auto_trimmer/face-detection-yolov8/yolov8_run_20250414_225903/weights/best.pt'
        video_paths = [
            {
                'input': '/data/mskscratch/users/ghoyer/Precision_Air/temp_output/trimmed_videos/20200605_232758.mp4',
                'output': '/data/mskscratch/users/ghoyer/Precision_Air/VLS_preprocessing/test_app2/20200605_232758_face7_70conf.mp4'
            },
            {
                'input': '/data/mskscratch/users/ghoyer/Precision_Air/VLS_preprocessing/test_app2/20200405-230345_20250407_234013.mp4',
                'output': '/data/mskscratch/users/ghoyer/Precision_Air/VLS_preprocessing/test_app2/20200405-230345_20250407_234013_face7_70conf.mp4' 
            },
            {
                'input': '/data/mskscratch/users/ghoyer/Precision_Air/VLS_preprocessing/test_app2/20200329-051346_20250407_233451.mp4', 
                'output': '/data/mskscratch/users/ghoyer/Precision_Air/VLS_preprocessing/test_app2/20200329-051346_20250407_233451_face7_70conf.mp4'
            },

        ]

        # Track processing results
        successful = 0
        failed = 0

        conf_threshold = 0.70 #85  # Adjust confidence threshold if needed
    
        for video in video_paths:
            print(f"\nProcessing video: {video['input']}")

            result = predict_and_annotate_video(
                model_path, 
                video['input'],
                video['output'],
                conf_threshold
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

if __name__ == "__main__":
    main()

# python -m src.face_detection.detect_and_blur.face_detect

# -----------------------
# model_path='/data/mskscratch/users/ghoyer/Precision_Air/VLS_auto_trimmer/face-detection-yolov8/yolov8_run_20250414_184834/weights/best.pt'  # this one might actually be good :,)
# model_path='/data/mskscratch/users/ghoyer/Precision_Air/VLS_auto_trimmer/face-detection-yolov8/yolov8_run_20250414_211928/weights/best.pt'  # same as above but 200 epoch instead of 100

