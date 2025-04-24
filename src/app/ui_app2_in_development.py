import os
import torch
import subprocess
import gradio as gr
import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO

from src.app import load_model, load_yolo_model, process_single_video_with_blur, process_directory_with_blur

# Preset values
THRESHOLD = 5
CONFIDENCE_THRESHOLD = 0.7
SMOOTHING_WINDOW = 10
YOLO_CONF_ADJUST_RANGE = (0.3, 0.99)
YOLO_BLUR_ONLY = False
YOLO_BLUR_KERNEL = (51, 51)
YOLO_TARGET_CLASSES = None  # e.g., ["person", "face"] or [0, 1]
CLASS_MODEL_PATH = "/data/mskscratch/users/ghoyer/Precision_Air/VLS_auto_trimmer/outputs/checkpoints/frame_classification/best_model.pt"
YOLO_MODEL_PATH = "/data/mskscratch/users/ghoyer/Precision_Air/VLS_auto_trimmer/face-detection-yolov8/yolov8_run_20250414_225903/weights/best.pt"

# Load models and transform once
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASS_MODEL, CLASS_TRANSFORM = load_model(CLASS_MODEL_PATH, DEVICE, app_mode=True)
YOLO_MODEL = load_yolo_model(YOLO_MODEL_PATH)

def process_videos(input_files, input_folder, output_path, add_timestamp, blur_only, target_classes, progress=gr.Progress()):
    paths = []
    if input_files:
        paths.extend([file.name for file in input_files])
    if input_folder:
        paths.append(input_folder)
    if not paths:
        return "Please select video file(s) or a folder."

    os.makedirs(output_path, exist_ok=True)
    output_files = []

    # Parse target_classes (comma-separated string to list)
    target_classes_list = [cls.strip() for cls in target_classes.split(",")] if target_classes else None
    if target_classes_list and target_classes_list == [""]:  # Handle empty input
        target_classes_list = None

    total_items = len(paths)
    for i, input_path in enumerate(paths):
        progress((i + 1) / total_items, desc=f"Processing {i + 1}/{total_items}: {os.path.basename(input_path)}")
        if os.path.isfile(input_path):
            result = process_single_video_with_blur(
                input_path,
                output_path,
                CLASS_MODEL,
                CLASS_TRANSFORM,
                YOLO_MODEL,
                threshold=THRESHOLD,
                confidence_threshold=CONFIDENCE_THRESHOLD,
                smoothing_window=SMOOTHING_WINDOW,
                app_mode=True,
                add_timestamp=add_timestamp,
                yolo_conf_adjust_range=YOLO_CONF_ADJUST_RANGE,
                yolo_blur_only=blur_only,
                yolo_blur_kernel=YOLO_BLUR_KERNEL,
                yolo_target_classes=target_classes_list
            )
            if result:
                output_files.append(result)
        elif os.path.isdir(input_path):
            process_directory_with_blur(
                input_path,
                output_path,
                CLASS_MODEL,
                CLASS_TRANSFORM,
                YOLO_MODEL,
                threshold=THRESHOLD,
                confidence_threshold=CONFIDENCE_THRESHOLD,
                smoothing_window=SMOOTHING_WINDOW,
                app_mode=True,
                add_timestamp=add_timestamp,
                yolo_conf_adjust_range=YOLO_CONF_ADJUST_RANGE,
                yolo_blur_only=blur_only,
                yolo_blur_kernel=YOLO_BLUR_KERNEL,
                yolo_target_classes=target_classes_list
            )
            # List all .mp4 files in the output directory
            for filename in os.listdir(input_path):
                if filename.lower().endswith((".mp4", ".avi")):
                    base_name = os.path.splitext(filename)[0]
                    output_file = os.path.join(output_path, f"{base_name}_processed.mp4")
                    if os.path.exists(output_file):
                        output_files.append(output_file)

    if output_files:
        return f"✅ Processing completed! Output files:\n" + "\n".join(output_files)
    return f"✅ Processing completed! Output saved at: {output_path}"

def open_output_folder(output_path):
    if os.path.isdir(output_path):
        if os.name == 'nt':  # Windows
            os.startfile(output_path)
        elif os.name == 'posix':
            subprocess.run(['open' if subprocess.call(['which', 'open'], stdout=subprocess.DEVNULL) == 0 else 'xdg-open', output_path])
        return f"Opened folder: {output_path}"
    else:
        return "Output folder does not exist. Run processing first."

def select_folder():
    root = tk.Tk()
    root.withdraw()
    folder_selected = filedialog.askdirectory()
    return folder_selected

with gr.Blocks(title="Automatic Video Trimming and Detection App") as demo:
    gr.Markdown("# 🎬 **Automatic Video Trimming and Detection App**")

    with gr.Row():
        input_files = gr.Files(label="Select Video File(s)", file_types=[".mp4", ".avi"], interactive=True)
        input_folder = gr.Textbox(label="Selected Input Folder", interactive=False)
        select_folder_btn = gr.Button("📁 Select Input Folder")

    select_folder_btn.click(select_folder, outputs=input_folder)

    with gr.Row():
        output_dir = gr.Textbox(label="Output Directory", value="processed_videos", interactive=True)
        select_output_folder_btn = gr.Button("📂 Select Output Folder")

    select_output_folder_btn.click(select_folder, outputs=output_dir)

    with gr.Row():
        add_timestamp = gr.Checkbox(label="Add Timestamp to Output File Names", value=False, interactive=True)
        blur_only = gr.Checkbox(label="Blur Only (No Bounding Boxes)", value=YOLO_BLUR_ONLY, interactive=True)

    with gr.Row():
        target_classes = gr.Textbox(
            label="Target Classes (comma-separated, e.g., 'person,face' or '0,1')",
            value="" if YOLO_TARGET_CLASSES is None else ",".join(map(str, YOLO_TARGET_CLASSES)),
            interactive=True
        )

    with gr.Row():
        run_btn = gr.Button("Start Processing 🚀", variant="primary")
        open_btn = gr.Button("📂 Open Output Folder")

    status_output = gr.Textbox(label="Status", lines=1, interactive=False)

    run_btn.click(
        fn=process_videos,
        inputs=[input_files, input_folder, output_dir, add_timestamp, blur_only, target_classes],
        outputs=status_output
    )

    open_btn.click(
        fn=open_output_folder,
        inputs=output_dir,
        outputs=status_output
    )

if __name__ == "__main__":
    demo.launch()

# python -m src.app.ui_app2