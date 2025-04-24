# src/face_detection

This directory implements the face/person detection and blurring pipeline for VLS video frames, offering two training avenues (Ultralytics YOLOv8 and custom detectors) and post-training tools for annotation and blurring.

## Contents

- **`dataset_face.py`**  
  `FaceDataset` for loading images with YOLO-format labels, applying transforms and optional bbox jitter; plus `custom_collate_fn` for batching.

- **`model_face.py`**  
  Three lightweight detector backbones inheriting from `BaseFaceDetector`:
  - `MobileFaceNet` (MobileNetV2 anchor detector)  
  - `YOLOTinyFaceNet` (tiny-YOLO–style CNN)  
  - `EfficientFaceNet` (EfficientNet-B0 backbone)

- **`train_yolov8.py`**  
  Trains YOLOv8 models via Ultralytics. Reads `dataset_yolov8_dir/data.yaml`, logs to WandB, saves runs under
  `<output_base_dir>/yolov8_run_<timestamp>/`.

- **`train_face_detection.py`**  
  Trains custom detectors with PyTorch (DDP or single‑GPU), logs loss/precision/recall/IoU, applies early stopping, and saves checkpoints in
  `<output_base_dir>/face_det_run_<timestamp>/`.

- **`infer_face.py`**  
  Performs detection on videos or frame folders, applies Gaussian blur or box overlays, and writes output videos to `inference_output_dir`.

- **`detect_and_blur/`**  
  Post‑training scripts for testing detection quality and applying blurs on unseen videos:
  - `face_detect.py` (annotate frames/video)
  - `blurred_face_detect.py` (blur regions)
  - `predict_and_annotate_video_with_classification.py` (dynamic threshold + auto-trim assisted)

## Installation

Ensure Python 3.8+ and install project dependencies:
```bash
pip install -r requirements.txt
```
Key packages:
- `torch`, `torchvision`, `torch.distributed`
- `ultralytics` (YOLOv8)
- `opencv-python`, `Pillow`
- `pandas`, `numpy`, `scikit-learn`
- `wandb` (optional), `pyyaml`, `tqdm`

## Configuration

All scripts read from YAML under `config/`:

### YOLOv8 (Ultralytics)
- **`config/yolov8_config.yaml`**
```yaml
paths:
  output_base_dir: "/path/to/face-detection-yolov8/runs"
  dataset_yolov8_dir: "/path/to/dataset_yolo"
  log_file: "logs/yolov8_train.log"

training:
  epochs: 200
  batch_size: 16
  learning_rate: 0.01
  num_workers: 4
  early_stopping:
    patience: 10
    min_delta: 0.001

model:
  type: "yolov8n.pt"         # pretrained model to fine-tune
  imgsz: 1280                # input image size
  rect: true                 # rectangular training
  data_yaml: "data.yaml"   # YOLO data config

inference:
  detection_weights_path: null  # override at inference time
```  

### Custom Detectors (PyTorch)
- **`config/face_config.yaml`**
```yaml
paths:
  new_data_dir: "/path/to/face_videos/raw"
  temp_frames_dir: "/path/to/temp_frames"
  dataset_dir: "/path/to/splits/frame_part4_yolo"
  output_base_dir: "/path/to/outputs/face_det"
  inference_output_dir: "/path/to/inference_outputs/face_det"
  log_file: "logs/face_det_pipeline.log"

frame_extraction:
  frame_interval: 10

split_ratios:
  train: 0.7
  val:   0.3
  test:  0.0

pipeline:
  dry_run: false

training:
  epochs: 50
  batch_size: 32
  learning_rate: 1e-4
  num_workers: 4
  early_stopping:
    patience: 5
    min_delta: 0.001

model:
  type: "mobilefacenet"   # mobilefacenet | yolotinynet | efficientfacenet
  num_anchors: 9

inference:
  weights_path: null         # e.g. "checkpoints/best_model.pt"
  conf_threshold: 0.5
  blur_only: false
  blur_kernel: [51,51]
  target_classes: ["patient","face","clinician"]
  batch_size: 32
  num_workers: 4
```

## Usage

### 1. Train YOLOv8 Detector
```bash
python -m src.face_detection.train_yolov8 \
  --config_file config/yolov8_config.yaml
```

### 2. Train Custom Detector
```bash
python -m src.face_detection.train_face_detection \
  --config_file config/face_config.yaml
```

### 3. Perform Detection & Blurring
```bash
python -m src.face_detection.infer_face \
  --config_file config/face_config.yaml \
  [--weights_path /path/to/checkpoint.pt]
```

### 4. Annotate Video with Boxes
```bash
python -m src.face_detection.detect_and_blur.face_detect
```
*Configure paths and thresholds inside script or extend to accept CLI arguments.*

### 5. Blur Detected Regions
```bash
python -m src.face_detection.detect_and_blur.blurred_face_detect
```

### 6. Dynamic Threshold & Classification-Assisted Blurring
```bash
python -m src.face_detection.detect_and_blur.predict_and_annotate_video_with_classification
```

## Outputs

- **YOLOv8 Runs**: `<output_base_dir>/yolov8_run_<timestamp>/`
- **Custom Detector Runs**: `<output_base_dir>/face_det_run_<timestamp>/`
- **Checkpoints**: `checkpoints/`
- **Logs & Figures**: `logs/`, `figures/`
- **Annotated/Blurred Videos**: as set in `inference_output_dir`

## Next Steps

After validating detection and blur quality, integrate with the trimming pipeline in [`src/app/README.md`](../app/README.md).  
Proceed to [`src/procedural_frame_classification/README.md`](../procedural_frame_classification/README.md) for frame-based trimming details.

