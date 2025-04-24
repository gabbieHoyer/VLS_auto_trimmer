# src/app/__init__.py

# Import the key functions from auto_trim_video.py to make them directly accessible
from .auto_trim_video import (
    load_model,
    process_single_video,
    process_directory,
    process_dataframe,
    trim_video,
    is_procedural_frame
)
# from .auto_trim_video import load_model, process_single_video, process_directory
from .auto_trim_and_detect import (
    load_yolo_model,
    detect_and_blur_video, 
    process_single_video_with_blur, 
    process_directory_with_blur
)

# Add to make manual_tools importable
from . import manual_tools

# Optionally, import ui_app to make it accessible ( not needed)
# from . import ui_app

# Define __all__ to control what gets imported with "from src.app import *"
__all__ = [
    "load_model",
    "process_single_video",
    "process_directory",
    "process_dataframe",
    "trim_video",
    "is_procedural_frame",
    "load_yolo_model",
    "detect_and_blur_video", 
    "process_single_video_with_blur", 
    "process_directory_with_blur",
    "manual_tools",
    # "ui_app",  # Uncomment to include ui_app in the namespace
]