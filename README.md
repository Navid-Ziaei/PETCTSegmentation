# PETCTSegmentation
## Description

This project focuses on segmenting medical images using the U-Net architecture. The main components of the project include data preprocessing, model training, and evaluation.

## Settings

In the GitHub repository, you will find `settings_sample.yaml` and `device_path_sample.yaml` in the `configs` folder. Copy these files into the `configs` folder and remove `_sample` from the copied file names. This is done because `settings.yaml` and `device_path.yaml` are included in `.gitignore` to prevent conflicts among team members.

### Device Path Configuration

Edit `device_path.yaml` to provide the correct paths to your dataset:

```yaml
raw_dataset_path: \"D:/path_to_dataset/dataset\"
preprocessed_dataset_path: \"D:/path_to_dataset/dataset_preprocessed/\"
model_path: \"D:/path_to_working_directory/saved_model/\"
```

The `raw_dataset_path` should contain two folders with the exact naming:
- `imagesTr`: Contains PET and CT images with suffix `*_0000.nii.gz` and `*_0001.nii.gz`.
- `labelsTr`: Contains GT `*.nii.gz`.
- fdg_metadata.csv
- psma_metadata.csv

### Settings Configuration

Edit `settings.yaml` to modify the training parameters:

```yaml
dataset: "pet_ct"
data_type: "psma" # "psma" "fdg" "both"
use_negative_samples : False
load_preprocessed_data : False

slice_spacing: 3
slice_overlap: 0


model: "unet"

# Training LDGD
load_trained_model : False
batch_size : 4
num_epochs : 5
test_size: 0.2
validation_size: 0.1
learning_rate: 0.0001

# Enable or disable debug mode
debug_mode: true
```

## Running the Project

### Steps to Run the Code

1. **Clone the Repository**:
   ```sh
   git clone https://github.com/yourusername/your-repo.git
   cd your-repo
   ```

2. **Install Dependencies**:
   Ensure you have Python and the necessary packages installed. You can install the required packages using:
   ```sh
   pip install -r requirements.txt
   ```

3. **Prepare Configuration Files**:
   Copy the sample configuration files and remove the `_sample` suffix:
   ```sh
   cp configs/settings_sample.yaml configs/settings.yaml
   cp configs/device_path_sample.yaml configs/device_path.yaml
   ```
   Edit `configs/device_path.yaml` and `configs/settings.yaml` with the appropriate paths and settings.


4. **Run the Main Script**:
   Execute the main script to start training:
   ```sh
   python main.py
   ```

## Experiments

To run experiments, open the Jupyter notebooks in the `experiments` folder.

## Project Structure

```
├── .gitignore
├── configs
│   ├── device_path.yaml
│   └── settings.yaml
├── main.py
├── README.md
├── results
├── saved_model
└── src
    ├── data
    │   ├── data_loader.py
    │   ├── data_preprocessor.py
    │   └── __init__.py
    ├── experiments
    │   ├── data_visualization.ipynb
    │   └── __init__.py
    ├── model
    │   ├── unet_model.py
    │   ├── utils
    │   │   └── __init__.py
    │   └── __init__.py
    ├── settings
    │   ├── paths.py
    │   ├── settings.py
    │   └── __init__.py
    ├── utils
    │   ├── multitapper
    │   │   ├── example.py
    │   │   ├── multitaper_spectrogram_python.py
    │   │   ├── README.md
    │   │   └── __init__.py
    │   ├── utils.py
    │   ├── __init__.py
    │   └── __pycache__
    │       ├── utils.cpython-312.pyc
    │       └── __init__.cpython-312.pyc
    ├── visualization
    │   ├── vizualize_utils.py
    │   ├── __init__.py
    │   └── __pycache__
    │       ├── vizualize_utils.cpython-312.pyc
    │       └── __init__.cpython-312.pyc
    └── __init__.py
