# src/procedural_frame_classification

This directory implements the procedural frame classification pipeline: training a frame classifier, running inference on videos, and and orchestrating the end‑to‑end frame classification pipeline (preprocessing, training, and inference).

## Contents

- **`dataset_frame.py`**
  Defines a PyTorch dataset class (`FrameDataset`) that loads frame images and labels for train/validation/test splits.

- **`model_frame.py`**
  Builds a ResNet‑18–based binary classifier head for distinguishing procedural vs. non‑procedural frames.

- **`train_frame_classification.py`**
  Trains the frame classifier using PyTorch DDP (optional), logs metrics to WandB, applies early stopping, and saves checkpoints.

- **`infer_frame.py`**
  Loads a trained model, runs inference on a test split or individual videos, generates classification reports, confusion matrices, and sample prediction figures.

- **`run_pipeline.py`**
  Orchestrates data preprocessing, model training, inference, and (in future) failed‑inference handling via subprocess calls.

## Installation

Ensure you have Python 3.8+ and install project requirements:
```bash
pip install -r requirements.txt
```
Key dependencies:
- `torch`, `torchvision`, `torch.distributed`
- `pandas`, `numpy`, `scikit-learn`
- `opencv-python`, `matplotlib`, `seaborn`
- `wandb`, `pyyaml`, `tqdm`

## Configuration

All scripts in this directory read settings from `config/config.yaml`. Below is an example configuration with comments explaining each section:

```yaml
# Directory paths for frame extraction and outputs
paths:
  new_data_dir: "/path/to/new/videos"        # Raw input videos, organized into 'procedural/' and 'nonprocedural/' subfolders
  temp_frames_dir: "/path/to/tmp/frames"     # Temporary directory where extracted frames are saved
  dataset_dir: "/path/to/dataset/frames"     # Directory for final train/val/test frame splits
  failed_inference_dir: "/path/to/failed/"   # Videos for which inference failed, for manual review
  output_base_dir: "/path/to/outputs/"       # Root directory for all run outputs (checkpoints, logs, figures)
  trimmed_vid_dir: "/path/to/trimmed_videos/" # Saved trimmed videos after inference

# Frame extraction settings
frame_extraction:
  frame_interval: 10   # Extract every Nth frame (e.g., 10 → every 10th frame)

# Train/Val/Test split proportions (must sum ≤ 1.0)
split_ratios:
  train: 0.7
  val:   0.2
  test:  0.1

# Pipeline behavior flags
dry_run: False         # If true, simulate file moves without modifying disk
log_file: "logs/preprocess.log"  # Local path for logging pipeline actions

# Training hyperparameters
training:
  epochs:         50     # Maximum epochs for training
  batch_size:     32     # Number of samples per batch
  learning_rate:  1e-4   # Initial learning rate
  lr_scheduler:
    factor: 0.1          # LR reduction factor
    patience: 3         # Epochs to wait before reducing LR
    min_lr: 1e-6        # Minimum learning rate
  early_stopping:
    patience: 5         # Epochs without improvement before stopping
    min_delta: 0.001    # Minimum change in monitored metric to qualify as improvement

# Inference parameters
inference:
  weights_path: null     # Path to a specific checkpoint (otherwise uses latest run)
  batch_size: 32         # Batch size for inference on frames
  num_workers: 4         # Data loader workers
  confidence_threshold: 0.5  # Threshold for classifying frames as procedural
  transforms:
    resize: 256
    crop:   224

# Script entrypoints for orchestrating the pipeline
scripts:
  train_script: "train_frame_classification"
  inference_script: "infer_frame"
```


## Usage

### 1. Train the Frame Classifier

```bash
python -m src.procedural_frame_classification.train_frame_classification \
  --config_file config/config.yaml \
  [--epochs 20] \
  [--batch_size 32] \
  [--learning_rate 1e-4]
```

- Initializes distributed training if `LOCAL_RANK` is set.
- Logs train/val loss, F1, accuracy, confusion matrices to console and WandB.
- Saves per‑epoch checkpoints and a `best_model.pt` under `<output_base_dir>/run_<timestamp>/checkpoints/`.

### 2. Inference on Frames or Videos

```bash
python -m src.procedural_frame_classification.infer_frame \
  --config_file config/config.yaml \
  [--weights_path /path/to/checkpoint.pt]
```

- Loads model from `weights_path` or `<latest_run>/checkpoints/best_model.pt`.
- Runs on the test split defined in `dataset_dir`.
- Prints classification report, saves:
  - `figures/test_confusion_matrix.png`
  - `figures/test_sample_predictions.png`

Override `weights_path` for a custom model or specific video:

```bash
python -m src.procedural_frame_classification.infer_frame \
  --config_file config/config.yaml \
  --weights_path "../custom_weights/model.pt"
```

### 3. Full End‑to‑End Pipeline

```bash
python -m src.procedural_frame_classification.run_pipeline \
  --config_file config/config.yaml
```

Steps executed:
1. Data processing via `src.dataset_preprocessing.process_new_data`
2. Model training (`train_frame_classification.py`)
3. Inference (`infer_frame.py`)
4. (Future) failed‑inference handling

Outputs are organized under `<output_base_dir>/run_<timestamp>/`:
- **Checkpoints**: `checkpoints/`
- **Logs**: `logs/`
- **Figures**: `figures/`
- **Trimmed Videos**: as configured in `trimmed_vid_dir`


## Next Steps

Once familiar with frame classification, explore:
-  [Face/Person Detection & Blurring](../face_detection/README.md)  for face/person detection and blurring.
- [Interactive Trimming & Combined Pipelines](../app/README.md) for interactive trimming and combined pipelines.