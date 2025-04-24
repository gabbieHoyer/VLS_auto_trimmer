# src/app

This directory holds the automatic trimming script, the interactive UI, and manual editing tools for VLS video cleaning.

## Contents

- **Automatic Trimming Script**  
  `auto_trim_video.py`  
  Uses a ResNet-18 model to detect procedural frames, smooth predictions, and trim videos with optional timestamping.

- **Trim + Detect Pipeline**  
  `auto_trim_and_detect.py`  
  First trims via the ResNet-18 classifier, then runs a YOLO model to detect faces/persons, blurs or overlays bounding boxes, and saves a final video.

- **Interactive UI**  
  `ui_app.py`  
  Launches a Gradio app for uploading videos, setting trimming options, and running the automatic pipeline.

- **Manual Tools**  
  In `manual_tools/`, scripts for frame-level trimming and face/person blurring.

---

## Usage

### 1. Automatic Trimming

```bash
python -m src.app.auto_trim_video \
  --input_path /path/to/video.mp4 \
  --output_dir /path/to/output \
  [--threshold 5] \
  [--confidence_threshold 0.5] \
  [--smoothing_window 5] \
  [--add_timestamp] \
  [--app_mode] \
  [--csv_path /path/to/videos.csv] \
  [--max_rows 10]
```

- `--input_path` : single video file or folder of videos
- `--csv_path` : CSV with a `Video_path` column
- `--output_dir` : where trimmed videos are saved
- `--threshold` : frames of padding (before/after detection), default 5
- `--confidence_threshold` : min probability for procedural frame, default 0.5
- `--smoothing_window` : window size for smoothing, default 5
- `--add_timestamp` : append timestamp to filename
- `--app_mode` : enable app-mode logging and embedded model
- `--max_rows` : limit rows when using `--csv_path`

To trim a single video using procedural frame classification:
```bash
python -m src.app.auto_trim_video --input_path /path/to/video.mp4 --output_dir trimmed_videos
```

To trim a single video with a timestamp in the output file name:
```bash
python -m src.app.auto_trim_video --input_path /path/to/video.mp4 --output_dir trimmed_videos --add_timestamp
```

### 2. Trim + Detect Pipeline
```bash
python -m src.app.auto_trim_and_detect \
  --input_path /path/to/video.mp4 \
  --output_dir /path/to/output \
  [--threshold 5] \
  [--confidence_threshold 0.7] \
  [--smoothing_window 10] \
  [--add_timestamp] \
  [--app_mode] \
  [--csv_path /path/to/videos.csv] \
  [--max_rows 5] \
  --class_weights_path /path/to/classifier.pt \
  --yolo_weights_path /path/to/yolo.pt \
  [--yolo_conf_adjust_range 0.3 0.99] \
  [--yolo_blur_only] \
  [--yolo_blur_kernel 51 51] \
  [--yolo_target_classes person face]
```
- classifier flags (`--threshold`, `--confidence_threshold`, `--smoothing_window`, `--add_timestamp`, `--app_mode`, `--csv_path`, `--max_rows`) behave as above
- `--class_weights_path` : path to ResNet-18 checkpoint
- `--yolo_weights_path` : path to YOLO model checkpoint
- `--yolo_conf_adjust_range` : min/max confidence for YOLO (default: 0.3 0.99)
- `--yolo_blur_only` : blur regions without drawing boxes
- `--yolo_blur_kernel` : Gaussian blur kernel size (e.g. 51 51)
- `--yolo_target_classes` : list of YOLO class IDs or names to process

### 3. Interactive Interface
```bash
python -m src.app.ui_app
```
1. Select video files or a folder
2. Choose output directory
3. (Optional) Toggle Add Timestamp
4. Click Process to run automatic trimming

![Automatic Trim Interface](assets/auto_vid_trim_tool.png)
*Automatic trimming UI.*

## Manual Tools
Inside `src/app/manual_tools/`:

- `video_trim_tool.py` – frame-by-frame trimming utility
- `video_blur_tool.py` – manual blur selector for privacy

### Frame Trimmer GUI
```bash
python -m src.app.manual_tools.video_trim_tool
```
- **Load Video / Folder**: queue one or many files
- **Slider & Buttons**: step through frames, set start/end trim points
- **Save Trimmed Video**: choose save location and export

*Set start/end frames and export trimmed clip.*

![Manual Trim Tool](assets/manual_trim_tool.png)
*Manual Trim Tool UI.*

### Blur Tool GUI
```bash
python -m src.app.manual_tools.video_blur_tool
```
- **Draw Rectangles**: click-drag to define blur regions per frame
- **Blur Level Slider**: adjust Gaussian kernel size
- **Undo/Redo**: revert or reapply region edits
- **Save Video**: export anonymized video with all blurs applied

*Draw and preview blur regions before saving.*

![Manual Blur Tool](assets/manual_blur_tool.png)
*Manual Blur Tool UI.*

## Packaging for Distribution
```bash
nuitka --standalone \
  --include-package=src.app \
  --include-package=src.utils \
  --output-dir=dist \
  src/app/ui_app.py
```

