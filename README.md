# Sleep Staging Algorithm

A machine learning-based algorithm for automatic sleep stage classification using EEG data from .edf files.

## Overview

This project implements a complete pipeline for sleep staging using EEG data:

1. **Data Loading**: Reads EEG data from .edf files
2. **Preprocessing**: Filters and prepares the EEG signals
3. **Feature Extraction**: Extracts frequency band powers and other relevant features
4. **Model Training**: Trains a machine learning model (Random Forest or SVM) to classify sleep stages
5. **Visualization**: Displays results including confusion matrices, hypnograms, and feature importance
6. **File Upload**: Allows for uploading new annotated .edf files to expand the dataset
7. **Scoring**: Can score unannotated .edf files using the trained model

Sleep stages are classified according to the AASM (American Academy of Sleep Medicine) standards:
- Wake
- N1 (light sleep)
- N2 (intermediate sleep)
- N3 (deep sleep, combining stages 3 and 4 from R&K)
- REM (rapid eye movement)

## Requirements

- Python 3.6+
- Dependencies:
  - numpy
  - pandas
  - matplotlib
  - seaborn
  - scikit-learn
  - mne
  - tqdm
  - joblib

## Installation

1. Clone this repository or download the files.

2. Install the required dependencies:
   ```
   pip install numpy pandas matplotlib seaborn scikit-learn mne tqdm joblib
   ```

## Dataset

The algorithm is designed to work with the Sleep-EDF Database Expanded, which contains .edf files with the following naming convention:
- PSG files: `*-PSG.edf`
- Hypnogram files: `*-Hypnogram.edf`

Each PSG file contains multiple channels of EEG data, and each hypnogram file contains sleep stage annotations.

## Usage

### Command-line Interface

The easiest way to use the algorithm is through the command-line interface provided by `sleep_staging_demo.py`:

```
python sleep_staging_demo.py [--train] [--score FILE] [--upload PSG_FILE HYPNO_FILE]
```

Options:
- `--train`: Train a new model on the dataset
- `--score FILE`: Score an unannotated PSG file
- `--upload PSG HYPNO`: Upload a new annotated PSG and hypnogram file pair
- `--model MODEL`: Path to a trained model to load (optional)
- `--data_dir DIR`: Path to the data directory (default: current directory)
- `--output_dir DIR`: Path to the output directory (default: 'results')

### Examples

1. Train a new model on the dataset:
   ```
   python sleep_staging_demo.py --train
   ```

2. Score an unannotated PSG file:
   ```
   python sleep_staging_demo.py --score new_recording.edf
   ```

3. Upload a new annotated file pair:
   ```
   python sleep_staging_demo.py --upload new_recording.edf new_hypnogram.edf
   ```

4. Train a model and then score a file:
   ```
   python sleep_staging_demo.py --train --score new_recording.edf
   ```

### Using the API

You can also use the `SleepStagingAlgorithm` class directly in your own code:

```python
from sleep_staging_algorithm import SleepStagingAlgorithm

# Initialize the algorithm
sleep_stager = SleepStagingAlgorithm(data_dir="path/to/edf/files")

# Load and process the dataset
sleep_stager.load_dataset()
sleep_stager.process_dataset()

# Train the model
sleep_stager.train_model(model_type='random_forest')  # or 'svm'

# Visualize results
sleep_stager.visualize_results()

# Score a new file
predictions, stage_predictions = sleep_stager.score_file("path/to/new/file.edf")

# Upload a new annotated file pair
features, labels = sleep_stager.upload_annotated_file("path/to/new/psg.edf", "path/to/new/hypnogram.edf")
```

## Algorithm Details

### Feature Extraction

The algorithm extracts the following features from each EEG epoch:
- Power in delta band (0.5-4 Hz)
- Power in theta band (4-8 Hz)
- Power in alpha band (8-13 Hz)
- Power in beta band (13-30 Hz)
- Power in gamma band (30-45 Hz)

These features are calculated for each EEG channel, resulting in a feature vector that captures the spectral characteristics of the EEG signal.

### Model Training

Two machine learning models are supported:
1. **Random Forest**: An ensemble of decision trees that is robust to overfitting and can handle non-linear relationships
2. **Support Vector Machine (SVM)**: A powerful classifier that works well with high-dimensional data

The models are trained on labeled data, where each 30-second epoch of EEG data is associated with a sleep stage label.

### Visualization

The algorithm provides several visualizations:
- **Confusion Matrix**: Shows the performance of the model in terms of true vs. predicted sleep stages
- **Feature Importance**: (For Random Forest) Shows which features are most important for classification
- **Hypnogram**: Displays the sequence of sleep stages over time, comparing true and predicted stages

## File Structure

- `sleep_staging_algorithm.py`: The main implementation of the algorithm
- `sleep_staging_demo.py`: A command-line interface for using the algorithm
- `examine_edf.py`: A utility script for examining .edf files
- `examine_single_pair.py`: A utility script for examining a single pair of PSG and hypnogram files
- `check_files.py`: A utility script for checking if files exist
- `list_files.py`: A utility script for listing files in a directory
- `README.md`: This documentation file

## Results

When the algorithm is run successfully, it will generate:
1. Trained models saved in the output directory
2. Visualizations of the model performance
3. Predictions for scored files

## Limitations and Future Work

- The current implementation focuses on spectral features. Additional features like time-domain statistics or entropy measures could improve performance.
- The algorithm assumes standard 30-second epochs, which is the clinical standard for sleep staging.
- Performance depends on the quality of the EEG recordings and the availability of accurate annotations.

Future improvements could include:
- Deep learning approaches (CNNs, RNNs) for feature extraction and classification
- Multi-modal analysis incorporating other physiological signals (EOG, EMG, ECG)
- Automated artifact detection and removal
- Transfer learning to improve performance with limited labeled data

## License

This project is licensed under the MIT License.

## Acknowledgments

- The Sleep-EDF Database Expanded for providing the dataset
- The MNE-Python library for EEG data processing
- The scikit-learn library for machine learning algorithms
