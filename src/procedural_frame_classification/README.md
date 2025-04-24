# src/procedural_frame_classification

This directory implements the procedural frame classification pipeline: training a frame classifier, running inference on videos, and orchestrating the end‑to‑end workflow including trimming.

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
- `opencv-python`, `matplotlib, seaborn`
- `wandb, pyyaml`, `tqdm`

## Configuration

All scripts read settings from `config/config.yaml`. Important sections:

- **paths**
-- `dataset_dir`: directory of frame images and labels
-- `output_base_dir`: root for runs, checkpoints, and logs
-- `trimmed_vid_dir`: destination for trimmed videos
- **training**
-- `epochs`, `batch_size`, `learning_rate`, `lr_scheduler`, `early_stopping`
- **inference**
-- `weights_path`, `batch_size`, `confidence_threshold`, `transforms`, etc.
- **pipeline**
-- `log_file`, `scripts.train_script`, `scripts.inference_script`, etc.


## Usage

### 1. Train the Frame Classifier
```bash
python -m src.procedural_frame_classification.train_frame_classification \
  --config_file config.yaml \
  [--epochs 20] \
  [--batch_size 32] \
  [--learning_rate 1e-4]
```

- Initializes distributed training if `LOCAL_RANK` is set
- Logs train/val loss, F1, accuracy, confusion matrices to console and WandB
- Saves per‑epoch checkpoints and a `best_model.pt` under `<output_base_dir>/run_<timestamp>/checkpoints/`

### 2. Inference on Frames or Videos
```bash
python -m src.procedural_frame_classification.infer_frame \
  --config_file config.yaml \
  [--weights_path /path/to/checkpoint.pt]
```

- Loads model from `weights_path` or `<latest_run>/checkpoints/best_model.pt`
- Runs on the test split defined in dataset_dir
- Prints classification report, saves:
-- `figures/test_confusion_matrix.png`
-- `figures/test_sample_predictions.png`

To override weights path for a specific video or custom model:
```bash
python -m src.procedural_frame_classification.infer_frame \
  --config_file config.yaml \
  --weights_path "../custom_weights/model.pt"
```

### 3. Full End‑to‑End Pipeline
```bash
python -m src.procedural_frame_classification.run_pipeline \
  --config_file config.yaml
```

This script calls, in order:
1. Data processing: `src.dataset_preprocessing.process_new_data`
2. Training: `train_frame_classification.py`
3. Inference: `infer_frame.py`
4. (Planned) Handling of failed inference

Log files and outputs share the same <output_base_dir>/run_<timestamp>/ structure.

## Outputs

- **Checkpoints**: `<output_base_dir>/run_<timestamp>/checkpoints/`
- **Logs**: `<output_base_dir>/run_<timestamp>/logs/`
- **Figures**: `<output_base_dir>/run_<timestamp>/figures/`
- **Trimmed Videos**: as configured in `trimmed_vid_dir`

## Next Steps

Once familiar with frame classification, visit:
-  [Face/Person Detection & Blurring](../face_detection/README.md)  for face/person detection and blurring.
- [Interactive Trimming & Combined Pipelines](../app/README.md) for interactive trimming and combined pipelines.