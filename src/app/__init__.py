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

# Optionally, you can import ui_app if you want to make it accessible (though typically not needed)
# from . import ui_app

# Define __all__ to control what gets imported with "from src.app import *"
__all__ = [
    "load_model",
    "process_single_video",
    "process_directory",
    "process_dataframe",
    "trim_video",
    "is_procedural_frame",
    # "ui_app",  # Uncomment if you want to include ui_app in the namespace
]