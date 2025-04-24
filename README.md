# VLS Auto Trimmer

The `VLS_auto_trimmer` project automates the cleaning of video laryngoscope (VLS) data by identifying and trimming non-procedural frames and detecting faces/persons for automatic blurring. This ensures that the processed video data is suitable for downstream machine learning tasks and analysis, particularly in medical contexts where privacy and relevance are important.

The project provides two main pipelines:

1. **Procedural Frame Classification**: Identifies and trims non-procedural frames from VLS videos, retaining only the relevant procedural content.  
2. **Face/Person Detection and Blurring**: Detects faces or persons in video frames and applies automatic blurring to protect privacy.

These pipelines are accessible via a single Python script (`run_pipeline.py`) for automated batch processing and a user-friendly Gradio-based UI (`ui_app.py`) for interactive use.

## Project Goals

- **Automated Data Cleaning**: Streamline the process of cleaning VLS video data by removing irrelevant frames and ensuring privacy through face/person blurring.
- **Dual Interface**: Provide both a command-line script (`run_pipeline.py`) and a graphical user interface (`ui_app.py`) to accommodate different user preferences.
- **Modularity**: Organize the codebase into modular components for easy maintenance, extension, and integration into larger workflows.
- **Downstream Compatibility**: Prepare cleaned video data for use in downstream machine learning tasks, such as training models for medical analysis or procedural assessment.

## Project Structure

The `VLS_auto_trimmer` codebase is organized into several directories, each serving a specific purpose.

### `src/`
Main source code directory containing all the Python modules and scripts.

#### `src/app/`
Contains scripts for video trimming and the user interface.

For detailed usage examples, CLI flags and screenshots, see [src/app/README.md](src/app/README.md).

#### `src/dataset_preprocessing/`
Handles preprocessing of video data to prepare it for training and inference.

- **`clean_unannotated_images.py`**: Removes unannotated images from the dataset to ensure data quality.
- **`compare_base_video_names.py`**: Compares video names to identify duplicates or inconsistencies.
- **`convert_script.py`**: Converts video data into a format suitable for processing (for example, frame extraction).
- **`extract_face_frames.py`**: Extracts frames containing faces for face detection tasks.
- **`extract_frames.py`**: Extracts frames from videos for procedural frame classification.
- **`handle_failed_inference.py`**: Moves videos that fail inference to a designated directory for review.
- **`process_new_data.py`**: Processes new video data by extracting frames and preparing them for training or inference.
- **`split_face_data.py`**: Splits face detection data into training, validation, and test sets.
- **`split_frames.py`**: Splits extracted frames into training, validation, and test sets for procedural frame classification.

#### `src/face_detection/`
Implements face/person detection and blurring to ensure privacy in video frames.

- **`dataset_face.py`**: Defines the dataset class for face detection tasks.
- **`infer_face.py`**: Runs inference to detect faces/persons in video frames.
- **`model_face.py`**: Defines the face detection model (for example, using a pre-trained object detection model like YOLO or Faster R-CNN).
- **`train_face_detection.py`**: Trains the face detection model on annotated face data.

#### `src/procedural_frame_classification/`
Implements the procedural frame classification pipeline to identify and trim non-procedural frames.

- **`dataset_frame.py`**: Defines the dataset class for procedural frame classification.
- **`infer_frame.py`**: Runs inference to classify frames as procedural or non-procedural using a pre-trained model.
- **`model_frame.py`**: Defines the procedural frame classification model (ResNet18 with a custom head for binary classification).
- **`train_frame_classification.py`**: Trains the procedural frame classification model on annotated frame data.
- **`run_pipeline.py`**: Orchestrates the full pipeline for procedural frame classification, including data preprocessing, training, inference, and handling failed inference.

  - **Usage**:
    ```bash
    python -m src.procedural_frame_classification.run_pipeline --config_file config.yaml
    ```

#### `src/utils/`
Contains utility modules for shared functionality across the project.

- **`config_loader.py`**: Loads and processes configuration files (`config.yaml`), resolving relative paths to absolute paths for consistency.
- **`gpu_setup.py`**: Sets up GPU/CPU environments and handles distributed training/inference.
- **`losses.py`**: Defines custom loss functions for training models.
- **`metrics.py`**: Defines evaluation metrics for model performance.
- **`onnx_export.py`**: Exports trained models to ONNX format for deployment.
- **`visualization.py`**: Provides functions for visualizing results (for example, confusion matrices and sample predictions).

#### `src/wandb/`
Integrates Weights & Biases (W&B) for experiment tracking and visualization (optional).

## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd VLS_auto_trimmer
    ```

2. **Create a Virtual Environment** (Optional but Recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
    ```
    Ensure you have the following key dependencies:
    - `torch` and `torchvision` for model training and inference.
    - `gradio` for the UI (ui_app.py).
    - `opencv-python` (`cv2`) for video processing.
    - `pandas` for handling CSV data.
    - `nuitka` (optional) for packaging the UI app.

4. **Prepare Configuration:**:
   - Copy the example configuration file:
   ```bash
   cp config/config.yaml.example config/config.yaml
    ```
    - Edit `config.yaml` to specify paths (e.g., `output_base_dir`, `dataset_dir`, `weights_path`) and other parameters.

5. **Download Pre-trained Models** (Optional):
    - Place pre-trained weights for procedural frame classification and face detection in the appropriate directories (e.g., `/path/to/VLS_auto_trimmer/outputs/checkpoints/`).

## Usage

### Command-Line Pipeline
To run the full pipeline (preprocessing, training, inference, and handling failed inference) for procedural frame classification:
```bash
python -m src.procedural_frame_classification.run_pipeline --config_file config.yaml
```

### Face Detection and Blurring
To train the face detection model:
```bash
python -m src.face_detection.train_face_detection --config_file config.yaml
```

To run inference for face detection and apply blurring:
```bash
python -m src.face_detection.infer_face --config_file config.yaml
```

### User Interface
For all trimming + blurring commands and options, see [src/app/README.md](src/app/README.md).


## Configuration
The `config.yaml` file in the `config/` directory controls the behavior of the pipelines. Key sections include:

- **`paths`**: Defines directories for data, outputs, and weights.
  - `output_base_dir`: Base directory for outputs (e.g., /data/mskscratch/users/ghoyer/Precision_Air/VLS_auto_trimmer/outputs/).
  - `dataset_dir`: Directory containing the dataset.
  - `trimmed_vid_dir`: Directory for trimmed videos.
- **`inference`**: Settings for inference (e.g., `weights_path`, `batch_size`).
- **`auto_trim`**: Settings for video trimming (e.g., `threshold`, `confidence_threshold`).
- **`pipeline`**: Settings for the full pipeline (e.g., `log_file`).

## Future Work
- **Performance Optimization**: Optimize frame processing for faster inference, especially for large video datasets.
- **Model Improvements**: Fine-tune models for better accuracy in procedural frame classification and face detection.

## License
This project is licensed under the MIT License. (placeholder)

## Reference
```bibtex
@misc{hoyer2025VLSAutoTrimmer,
  author       = {Hoyer, Gabrielle and Runnels, Sean},
  title        = {VLSAutoTrimmer},
  year         = {2025},
  howpublished = {Computer software},
  version      = {1.0.0},
  note         = {Available at \url{https://github.com/gabbieHoyer/VLS_auto_trimmer}},
}
```

## Contact
- **Gabrielle Hoyer**  
  - UCSF / UC Berkeley  
  - [gabbie.hoyer@ucsf.edu](mailto:gabbie.hoyer@ucsf.edu)  
  - [gabrielle_hoyer@berkeley.edu](mailto:gabrielle_hoyer@berkeley.edu)
  
For questions, suggestions, or collaboration opportunities, feel free to reach out via email.
