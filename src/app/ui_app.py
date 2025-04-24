
import os
import torch
import subprocess
import gradio as gr
import tkinter as tk
from tkinter import filedialog

from src.app import load_model, process_single_video, process_directory

# Preset values
THRESHOLD = 5
CONFIDENCE_THRESHOLD = 0.5
SMOOTHING_WINDOW = 5
MODEL_PATH = "/data/mskscratch/users/ghoyer/Precision_Air/VLS_auto_trimmer/outputs/checkpoints/frame_classification/best_model.pt"

# Load model and transform once
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL, TRANSFORM = load_model(MODEL_PATH, DEVICE, app_mode=True)

def trim_videos(input_files, input_folder, output_path, add_timestamp, progress=gr.Progress()):
    paths = []
    if input_files:
        paths.extend([file.name for file in input_files])
    if input_folder:
        paths.append(input_folder)
    if not paths:
        return "Please select video file(s) or a folder."

    os.makedirs(output_path, exist_ok=True)
    output_files = []

    total_items = len(paths)
    for i, input_path in enumerate(paths):
        progress((i + 1) / total_items, desc=f"Processing {i + 1}/{total_items}: {os.path.basename(input_path)}")
        if os.path.isfile(input_path):
            result = process_single_video(
                input_path, 
                output_path, 
                MODEL, 
                TRANSFORM, 
                THRESHOLD, 
                CONFIDENCE_THRESHOLD, 
                SMOOTHING_WINDOW, 
                app_mode=True, 
                add_timestamp=add_timestamp
            )
            if result:
                output_files.append(result)
        elif os.path.isdir(input_path):
            process_directory(
                input_path, 
                output_path, 
                MODEL, 
                TRANSFORM, 
                THRESHOLD, 
                CONFIDENCE_THRESHOLD, 
                SMOOTHING_WINDOW, 
                app_mode=True, 
                add_timestamp=add_timestamp
            )
            # List all .mp4 files in the output directory that match the input directory's videos
            for filename in os.listdir(input_path):
                if filename.lower().endswith((".mp4", ".avi")):
                    output_file = os.path.join(output_path, os.path.splitext(filename)[0] + ".mp4")
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

with gr.Blocks(title="Automatic Video Trimming App") as demo:
    gr.Markdown("# 🎬 **Automatic Video Trimming App**")

    with gr.Row():
        input_files = gr.Files(label="Select Video File(s)", file_types=[".mp4", ".avi"], interactive=True)
        input_folder = gr.Textbox(label="Selected Input Folder", interactive=False)
        select_folder_btn = gr.Button("📁 Select Input Folder")

    select_folder_btn.click(select_folder, outputs=input_folder)

    with gr.Row():
        output_dir = gr.Textbox(label="Output Directory", value="trimmed_videos", interactive=True)
        select_output_folder_btn = gr.Button("📂 Select Output Folder")

    select_output_folder_btn.click(select_folder, outputs=output_dir)

    with gr.Row():
        add_timestamp = gr.Checkbox(label="Add Timestamp to Output File Names", value=False, interactive=True)

    with gr.Row():
        run_btn = gr.Button("Start Processing 🚀", variant="primary")
        open_btn = gr.Button("📂 Open Output Folder")

    status_output = gr.Textbox(label="Status", lines=1, interactive=False)

    run_btn.click(
        fn=trim_videos,
        inputs=[input_files, input_folder, output_dir, add_timestamp], 
        outputs=status_output
    )

    open_btn.click(
        fn=open_output_folder,
        inputs=output_dir,
        outputs=status_output
    )

if __name__ == "__main__":
    demo.launch()


# python -m src.app.ui_app

#  nuitka --standalone --include-package=src.app --include-package=src.utils --output-dir=dist src/app/ui_app.py
