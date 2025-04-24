# VLS Auto Trimmer

The `VLS_auto_trimmer` project automates the cleaning of video laryngoscope (VLS) data by:

- **Procedural Frame Classification**: Identifies and trims non‑procedural frames, retaining only relevant procedural content.
- **Face/Person Detection & Blurring**: Detects faces/persons in frames and applies automatic blurring to protect privacy.

Both pipelines can be run in batch via a single entrypoint script or interactively through a Gradio UI.

## Project Goals

- **Automated Data Cleaning**: Remove irrelevant frames and ensure privacy.
- **Flexible Interfaces**: Command‑line and graphical UI options.
- **Modular Design**: Clear separation of preprocessing, modeling, and UI components.
- **Ready for Downstream Use**: Outputs cleaned videos and datasets suitable for machine learning.

## Repository Structure

```text
VLS_auto_trimmer/
├── src/
│   ├── app/                          # Interactive UI and trimming/blurring scripts
│   ├── dataset_preprocessing/       # Raw video → frames → train/val/test splits
│   ├── procedural_frame_classification/ # Train & infer procedural/non‑procedural classifier
│   ├── face_detection/              # Train & infer face/person detection and blurring
│   └── utils/                       # Shared utilities (config, GPU setup, metrics, etc.)
├── config/                          # YAML configuration files
│   ├── config.yaml.example          # Procedural pipeline defaults
│   └── face_config.yaml.example     # Face-detection pipeline defaults
├── requirements.txt
└── README.md                        # This file
```

For detailed instructions and examples, see the README in each major subfolder:

- [Dataset Preprocessing](src/dataset_preprocessing/README.md)
- [Procedural Frame Classification](src/procedural_frame_classification/README.md)
- [Face/Person Detection](src/face_detection/README.md)
- [Interactive App & UI](src/app/README.md)

## Quickstart

1. **Clone & install**:
   ```bash
   git clone https://github.com/gabbieHoyer/VLS_auto_trimmer.git
   cd VLS_auto_trimmer
   ```
  Install dependencies (choose one option):
  
  **Option 1: Using Conda (recommended for cross-platform compatibility and GPU support)**
  Install Miniconda if not already installed, then:
  ```bash
  conda env create -f environment.yml
  conda activate my_project_env
  ```

  **Option 2: Using pip and a virtual environment**
  Ensure Python 3.9+ is installed, then:
  ```bash
  python -m venv venv
  source venv/bin/activate  # On Windows: venv\Scripts\activate
  pip install -r requirements.txt
  ```

2. **Prepare configuration**:
   ```bash
   cp config/config.yaml.example config/config.yaml
   cp config/face_config.yaml.example config/face_config.yaml
   ```
   Edit the copied YAMLs under `config/` to set your paths and parameters.

3. **Run full batch pipeline**:
   ```bash
   python run_pipeline.py --config_file config/config.yaml
   ```
   Executes preprocessing, training, inference, and (optionally) post‑processing steps.

4. **Launch interactive UI**:
   ```bash
   python src/app/ui_app.py --config_file config/config.yaml
   ```

## Configuration Overview

Each pipeline has its own example YAML in `config/`:

- **Procedural Frame Classification**: `config/config.yaml`
  - See [Procedural Frame Classification README](src/procedural_frame_classification/README.md) for details on `paths:`, `frame_extraction:`, `training:`, and `split_ratios:` settings.
- **Face/Person Detection**: `config/face_config.yaml`
  - See [Face/Person Detection README](src/face_detection/README.md) for details on `paths:`, `model:`, `training:`, and `inference:` settings.
- **Dataset Preprocessing**: uses shared sections in `config/config.yaml`
  - See [Dataset Preprocessing README](src/dataset_preprocessing/README.md) for detailed `dataset_preprocessing:` options.

## Next Steps

After setup and preprocessing, follow subfolder READMEs to:

- Train/infer the **procedural frame classification** model.
- Train/infer the **face/person detection** model and apply blurring.
- Use the **interactive UI** for combined trimming & blurring.

## Future Work

- **Performance Optimization**: Speed up frame and video processing for large datasets.
- **Model Improvements**: Enhance accuracy for both procedural classification and face detection.

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
  
For questions or collaboration, feel free to reach out via email.

