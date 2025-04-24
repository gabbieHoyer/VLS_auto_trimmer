# src/dataset_preprocessing

This directory handles preprocessing of raw VLS video data to prepare it for training and inference in both the procedural frame‑classification and face‑person detection pipelines.

## Contents


### Procedural Frame Classification

- **`extract_frames.py`**  
  Extracts evenly spaced frames from videos for procedural frame classification.
- **`split_frames.py`**  
  Splits procedural frames into train/val/test sets, ensuring all frames from the same source video remain together.
- **`process_new_data.py`**  
  End‑to‑end pipeline for procedural and non‑procedural classes:
  1. Extract frames for both `procedural` and `nonprocedural` videos.
  2. Split into train/val/test according to configured ratios.
  3. Clean up temporary frames and processed videos.

### Face/Person Detection

- **`extract_face_frames.py`**  
  Extracts high-resolution frames for face detection at native or specified FPS, with adjustable JPEG quality.
- **`split_face_data.py`**  
  Splits extracted face images and their `.txt` annotations into train/val/test sets, grouping all frames from the same source video.
- **`handle_failed_inference.py`**  
  Moves or logs videos/frames that fail downstream inference for manual review.

## Subdirectories

### `format_standardization/`

Scripts that enforce consistent annotation formats, image sizes, and directory layouts:

- **`convert_script.py`**  
  Converts Pascal VOC XML annotations to plain `.txt` format (`x_min y_min width height`).
- **`convert_to_yolo_format.py`**  
  Converts plain `.txt` boxes and images into YOLO format (`class_id x_center y_center width_norm height_norm`), preserving train/val splits.
- **`standardize_image_sizes.py`**  
  Crops 1280×720 images to 960×720 or resizes 640×480 to 960×720, updates YOLO labels accordingly, and optionally visualizes before/after bounding boxes.

### `iterative_quality_check/`

Utilities (in development) for human-in-the-loop dataset QC, annotation consistency, and corruption checks:

- **`clean_unannotated_images.py`**  
  Identifies image files (`.jpg`) without matching annotation files (`.xml`) and optionally deletes unannotated images to maintain dataset integrity. Usage:
  ```bash
  python -m src.dataset_preprocessing.iterative_quality_check.clean_unannotated_images \
    --image_dir /path/to/images \
    --label_dir /path/to/annotations \
    [--dry_run on|off]
  ```

- **`cleanup_unpaired_labels.py`**  
  Detects YOLO label files (`.txt`) that lack corresponding image files and optionally removes orphaned labels. Usage:
  ```bash
  python -m src.dataset_preprocessing.iterative_quality_check.cleanup_unpaired_labels \
    --parent_dir /path/to/dataset_root \
    [--dry_run on|off]
  ```

- **`compare_base_video_names.py`**  
  Compares base video identifiers across face-image directories and procedural/nonprocedural splits to find mismatches or missing subjects. Usage:
  ```bash
  python -m src.dataset_preprocessing.iterative_quality_check.compare_base_video_names \
    --face_images_dir /path/to/face_images \
    --splits_base_dir /path/to/splits_root
  ```

- **`find_missing_processed.py`**  
  Identifies input video files (`.mp4`) that lack corresponding processed outputs (e.g., anonymized or trimmed) after manual/tools workflows, listing missing pairs for follow‑up. Usage:
  ```bash
  python -m src.dataset_preprocessing.iterative_quality_check.find_missing_processed \
    --input_dir /path/to/input_videos \
    --output_dir /path/to/processed_videos
  ```

- **`copy_nth_jpg.py`**  
  Copies every *n*th `.jpg`/`.jpeg` frame from an input directory to an output directory to balance dataset splits during manual dataset building. Usage:
  ```bash
  python -m src.dataset_preprocessing.copy_nth_jpg \
    /path/to/source_frames \
    /path/to/selected_frames \
    <frequency>
  ```

---

## Setup

See the main [README.md](../../README.md#quickstart) for installation instructions, including Conda and pip options to set up the project dependencies.

---

## Configuration

All scripts read from `config/config.yaml` (or `config/face_config.yaml` for face pipeline). Key sections:

```yaml
paths:
  new_data_dir:         # raw videos for procedural classes
  temp_frames_dir:      # intermediate frame dump for procedural
  dataset_dir:          # train/val/test output for procedural frames
  dataset_yolov8_dir:   # YOLO dataset root for face detection

frame_extraction:
  frame_interval: 10    # every nth frame for procedural extraction

split_ratios:
  train: 0.7
  val:   0.2
  test:  0.1

pipeline:
  dry_run: False
  log_file: logs/preprocess.log
```

For face‑detection pipeline, see `face_config.yaml` (shares `split_ratios` & `pipeline.dry_run`).

---

## Usage

### 1. Clean Unannotated Images

```bash
python -m src.dataset_preprocessing.clean_unannotated_images \
  --img_dir ${PIPELINE.temp_frames_dir}
```

### 2. Extract Frames

#### a) Procedural / Nonprocedural

```bash
python -m src.dataset_preprocessing.extract_frames \
  --input_dir ${PIPELINE.new_data_dir} \
  --output_dir ${PIPELINE.temp_frames_dir} \
  --class_name procedural \
  --frame_interval ${frame_extraction.frame_interval} \
  --config config/config.yaml
```

Use `--class_name nonprocedural` for non‑procedural frames.

#### b) Face Frames

```bash
python -m src.dataset_preprocessing.extract_face_frames \
  --input_dir /path/to/face_videos/ \
  --output_dir /path/to/temp_face_frames/ \
  --extract_fps 10 \
  --quality 95 \
  --config_file face_config.yaml
```

Omit `--extract_fps` to extract at native video FPS (all frames).

### 3. Split into Train/Val/Test

#### a) Procedural

```bash
python -m src.dataset_preprocessing.split_frames \
  --temp_frames_dir ${PIPELINE.temp_frames_dir} \
  --dataset_dir ${PIPELINE.dataset_dir} \
  --dry_run False \
  --config config/config.yaml
```

Add `--dry_run True` to simulate.

#### b) Face

```bash
python -m src.dataset_preprocessing.split_face_data \
  --image_dir /path/to/temp_face_frames/ \
  --label_dir /path/to/temp_face_labels_txt/ \
  --config_file face_config.yaml \
  --dry_run on
```

### 4. Standardize Annotation Formats & Image Sizes

#### a) Pascal VOC → TXT

```bash
python -m src.dataset_preprocessing.format_standardization.convert_script
```

#### b) TXT → YOLO Format

```bash
python -m src.dataset_preprocessing.format_standardization.convert_to_yolo_format \
  --input_dir /path/to/splits/ \
  --output_dir /path/to/dataset_yolov8/
```

#### c) Standardize Image Sizes & Labels

```bash
python -m src.dataset_preprocessing.format_standardization.standardize_image_sizes \
  --input_dir  /path/to/yolo_dataset/images_and_labels \
  --output_dir /path/to/standardized_dataset \
  --crop_width 160        # pixels crop each side for 1280×720 images
  --target_size 960 720   # resize 640×480 images to 960×720
  [--dry_run] [--num_viz_samples 10] [--dry_run_samples 5]
```

Crops or resizes images, updates YOLO labels, and optionally visualizes bounding boxes.

---

### 5. End‑to‑End Procedural Data Processing

```bash
python -m src.dataset_preprocessing.process_new_data \
  --config_file config/config.yaml
```

Performs extract → split → cleanup for both `procedural` & `nonprocedural`.

---

## Next Steps

After preprocessing, proceed to:

- [Procedural Frame Classification](../procedural_frame_classification/README.md)
- [Face Detection & Blurring](../face_detection/README.md)
- [Interactive Trimming & UI](../app/README.md)
