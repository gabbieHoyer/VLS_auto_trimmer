# DataFrame and Video Data Usage Instructions

This repository includes a DataFrame (`video_splits_relative_paths.csv`) and a series of curated datasets developed during the creation and refinement of an automated system for video trimming, face/person detection, and anonymization. These datasets were iteratively generated and used for training, validating, and testing the components of this system, including the core scripts and UI applications.

The information below is intended to describe the data organization, usage options, and development flow—not all components need to be downloaded or used unless you wish to replicate the development process or adapt the tools to a new dataset.

---

## Prerequisites

- Python 3.x with `pandas` installed (`pip install pandas`).
- `video_splits_relative_paths.csv` for reference to processed video file paths.

---

**1. Overview of Created Datasets and Processing Flow**

These datasets were generated and refined during development of the trimming and anonymization pipeline. They are available in the shared Box folder: [Shared Box Directory](https://uofu.box.com/s/sxmhp25otspjt9kvplrtu31k0vdpbqi9)

**1.1 frame_classification_data.zip**
- Structure: `data/{train,val,test}/{procedural,nonprocedural}/`
- Each subfolder contains `.jpeg` image frames extracted from original videos.
- Used for training and validating the ResNet-based procedural frame classifier.

**1.2 face_person_detection_data.zip**
- Structure: `dataset_yolo_refined7_clean/{train,val}/{images,labels,visualizations}/`
- Contains `.jpeg` images, YOLO-style `.txt` bounding box annotations, and visualizations.
- Includes `data.yaml` file for YOLOv8. Paths inside must be updated after extraction.
- Used to train a YOLO model for face and person detection to support anonymization.

**1.3 anonymized_trimmed_videos.zip**
- Structure: `clean_trimmed_videos/{.mp4}`
- Final output of the pipeline: anonymized and trimmed video files.
- These files underwent:
  - Trimming based on procedural frame classification.
  - Face/person blurring based on YOLO predictions.
  - Manual quality review for anonymization compliance.
- These are ready for use in downstream tasks or analysis.

---

**2. Optional Reference Dataset (Original and Processed Video Paths)**

The `video_splits_relative_paths.csv` file links original video names to their processed counterparts. While not required for running the trimming pipeline, it provides useful metadata for dataset traceability.

```
Video_path: Data_VL Videos for review (100)/20201029-170946.avi
Processed_video_path: clean_trimmed_videos/20201029-170946_anon.mp4
```

Example directory layout:

```
/path/to/your/data/
├── Data_ VL Videos for review (100)/
│   └── 20201029-170946.avi
├── clean_trimmed_videos/
│   └── 20201029-170946_anon.mp4
└── video_splits_relative_paths.csv
```

**3. Adjust Paths for Your System**

The `video_splits_relative_paths.csv` file contains relative paths to the video files, such as:

```
Data_VL Videos for review (100)/20201029-170946.avi
clean_trimmed_videos/20201029-170946_anon.mp4
```

If your code requires absolute paths, prepend your base directory to the relative paths using this Python script:

```python
import pandas as pd
import os

# Load the DataFrame
split_df = pd.read_csv('video_splits_relative_paths.csv')

# Specify your base directory
base_dir = '/path/to/your/data'  # Replace with your actual base directory

# Prepend the base directory to the relative paths
split_df['Video_path'] = split_df['Video_path'].apply(lambda x: os.path.join(base_dir, x))
split_df['Processed_video_path'] = split_df['Processed_video_path'].apply(lambda x: os.path.join(base_dir, x))

# Save or use the updated DataFrame
split_df.to_csv('video_splits_absolute_paths.csv', index=False)
print(split_df)
```

**4. Verify Paths**

Before running your analysis, verify that the paths in `split_df['Video_path']` and `split_df['Processed_video_path']` point to the correct files on your system. You can check this by:

```python
import os

# Check if files exist
for path in split_df['Video_path']:
    print(f"Checking {path}: {'Exists' if os.path.exists(path) else 'Does not exist'}")
for path in split_df['Processed_video_path']:
    print(f"Checking {path}: {'Exists' if os.path.exists(path) else 'Does not exist'}")
```

---

## DataFrame Columns

The `video_splits_relative_paths.csv` contains 31 columns with metadata and annotations for original and processed videos. Below is a brief description of each column:

### Original Video Metadata
- `Video_path`: Relative path to the original video file (e.g., `Data_ VL Videos for review (100)/20201029-170946.avi`).
- `Duration_sec`: Duration of the original video in seconds.
- `FPS`: Frames per second of the original video.
- `Width`: Width of the original video in pixels.
- `Height`: Height of the original video in pixels.
- `Frame_Count`: Total number of frames in the original video.

### Processed Video Metadata
- `Processed_video_path`: Relative path to the processed video file (e.g., `clean_trimmed_videos/20201029-170946_anon.mp4`).
- `Processed_Duration_sec`: Duration of the processed video in seconds.
- `Processed_FPS`: Frames per second of the processed video.
- `Processed_Width`: Width of the processed video in pixels.
- `Processed_Height`: Height of the processed video in pixels.
- `Processed_Frame_Count`: Total number of frames in the processed video.

### Santiago and Sean Annotations
These columns contain metrics evaluated by two annotators (Santiago and Sean) based on the original videos, related to a medical procedure (intubation).

**Procedure Metrics**
- `Santiago_Total_time_of_attempt`: Time (in seconds) taken by Santiago for the procedure attempt, possibly in binned or inconsistent formats.
- `Sean_Total_time_of_attempt`: Time (in seconds) taken by Sean for the procedure attempt, possibly in binned or inconsistent formats.
- `Santiago_Total_time_of_tracheal_access`: Time (in seconds) taken by Santiago to achieve tracheal access.
- `Sean_Total_time_of_tracheal_access`: Time (in seconds) taken by Sean to achieve tracheal access.
- `Santiago_new_blood_erythema`: Presence or severity of blood or erythema (redness) observed by Santiago, in varied text formats.
- `Sean_new_blood_erythema`: Presence or severity of blood or erythema observed by Sean, in varied text formats.
- `Santiago_Best_CL_grade`: Best Cormack-Lehane grade (a laryngoscopy classification, typically 1–4, assessing glottic view) assigned by Santiago, possibly using Roman numerals (e.g., `I`, `IIa`).
- `Sean_Best_CL_grade`: Best Cormack-Lehane grade assigned by Sean, possibly using mixed formats (e.g., `1`, `2a`).

**Overall Procedure Difficulty (Review Classifications)**
Classification of intubation difficulty by Santiago and Sean, based on procedural success, equipment motion, anatomical displacement, and trauma. Informed by time metrics, CL grade, blood/erythema, and number of attempts. Classes:

- `1`: Easy tracheal access (successful, single smooth motion, minimal displacement).
- `2`: Moderately difficult access (successful, multiple reversals/redirections, moderate displacement, no trauma). Subclasses `2b`, `2c`, `2d` indicate specific challenges.
- `3`: Severely difficult access (successful, multiple reversals, severe displacement, visible trauma). Subclasses `3b` & `3c` indicate specific severe challenges.
- `4`: Impossible access (Unsuccessful placement of ETT into the trachea).
    - `4a`: Unable to pass tracheal access equipment into field of view of VL.
    - `4b`: Unable to reach the glottic opening with tracheal access equipment. 
    - `4c`: Unable to pass tip of tracheal access equipment through the glottic opening.
    - `4d`: Unable to pass tip of tracheal access equipment through subglottic trachea.

- **`Santiago_Review` Unique values**: `1`, `2`, `2b`, `2c`, `2d`, `3`, `3b`, `4b`.

- **`Sean_Review` Unique values**: `1`, `2`, `2b`, `2c`, `3`, `3b`, `3c`, `4b`, `x`.

    *value x: (ETT check).*

**Standardized and Numeric Annotations**
These columns provide standardized (cleaned and unified) or numeric versions of the Santiago and Sean annotations for consistent analysis.

- `Santiago_Total_time_of_attempt_std`: Standardized time (float, in seconds) for Santiago’s total attempt, converting binned formats (e.g., `90+ secs` → `90.0`) or inconsistent entries.
- `Sean_Total_time_of_attempt_std`: Standardized time (float, in seconds) for Sean’s total attempt, converting binned formats or inconsistent entries.
- `Santiago_Total_time_of_tracheal_access_std`: Standardized time (float, in seconds) for Santiago’s tracheal access time.
- `Sean_Total_time_of_tracheal_access_std`: Standardized time (float, in seconds) for Sean’s tracheal access time.
- `Santiago_new_blood_erythema_std`: Standardized blood/erythema observation by Santiago, unified to `Y` (yes), `N` (no), or `1` (ambiguous). Unique values: `1`, `N`, `Y`.
- `Sean_new_blood_erythema_std`: Standardized blood/erythema observation by Sean, unified to `Y`, `N`, `1` (ambiguous), or `??` (uncertain). Unique values: `1`, `??`, `N`, `Y`.
- `Santiago_Best_CL_grade_std`: Standardized Cormack-Lehane grade by Santiago, converting Roman numerals to Arabic with subgrades (e.g., `I` → `1`, `IIa` → `2a`). Unique values: `1`, `2a`, `2b`.
- `Sean_Best_CL_grade_std`: Standardized Cormack-Lehane grade by Sean, unified to Arabic numerals with subgrades (e.g., `1`, `2a`). Unique values: `1`, `2`, `2a`, `2b`.
- `Santiago_Best_CL_grade_num`: Numeric-only version of Santiago’s Cormack-Lehane grade (e.g., `IIa` → `2`). Unique values: `1`, `2`.
- `Sean_Best_CL_grade_num`: Numeric-only version of Sean’s Cormack-Lehane grade (e.g., `2a` → `2`). Unique values: `1`, `2`.

**Data Split**
- `split`: Values include `train`, `test`, `validation`

---

**Troubleshooting**

- **File Not Found Errors**: Confirm that the relevant zip files were extracted into the intended directories.
- **YOLO Data Paths**: Update the `data.yaml` to point to your local directory for training or evaluation.
- **Replicating Model Training**: Only required if you wish to recreate the model training process; otherwise, the final video outputs can be used directly.

---



