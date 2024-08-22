# PETCTSegmentation
## Description

This project focuses on segmenting medical images using the U-Net architecture. The main components of the project include data preprocessing, model training, and evaluation.

## Settings

In the GitHub repository, you will find \`settings_sample.yaml\` and \`device_path_sample.yaml\` in the \`configs\` folder. Copy these files into the \`configs\` folder and remove \`_sample\` from the copied file names. This is done because \`settings.yaml\` and \`device_path.yaml\` are included in \`.gitignore\` to prevent conflicts among team members.

### Device Path Configuration

Edit \`device_path.yaml\` to provide the correct paths to your dataset:

\`\`\`yaml
raw_dataset_path: \"D:/path_to_dataset/dataset\"
preprocessed_dataset_path: \"D:/path_to_dataset/dataset_preprocessed/\"
model_path: \"D:/path_to_working_directory/saved_model/\"
\`\`\`

The \`raw_dataset_path\` should contain two folders with the exact naming:
- \`imagesTr\`: Contains PET and CT images with suffix \`*_0000.nii.gz\` and \`*_0001.nii.gz\`.
- \`labelsTr\`: Contains GT \`*.nii.gz\`.
- fdg_metadata.csv
- psma_metadata.csv

### Settings Configuration

Edit \`settings.yaml\` to modify the training parameters:

\`\`\`yaml
dataset: \"pet_ct\"
model: \"unet\"
load_trained_model: False
batch_size: 100
num_epochs: 100
test_size: 0.2
validation_size: 0.1
learning_rate: 0.0001
debug_mode: true
\`\`\`

## Running the Project

### Steps to Run the Code

1. **Clone the Repository**:
   \`\`\`sh
   git clone https://github.com/yourusername/your-repo.git
   cd your-repo
   \`\`\`

2. **Install Dependencies**:
   Ensure you have Python and the necessary packages installed. You can install the required packages using:
   \`\`\`sh
   pip install -r requirements.txt
   \`\`\`

3. **Prepare Configuration Files**:
   Copy the sample configuration files and remove the \`_sample\` suffix:
   \`\`\`sh
   cp configs/settings_sample.yaml configs/settings.yaml
   cp configs/device_path_sample.yaml configs/device_path.yaml
   \`\`\`
   Edit \`configs/device_path.yaml\` and \`configs/settings.yaml\` with the appropriate paths and settings.

4. **Run the Preprocessing Script**:
   Preprocess the raw data if not already preprocessed:
   \`\`\`sh
   python preprocess.py
   \`\`\`

5. **Run the Main Script**:
   Execute the main script to start training:
   \`\`\`sh
   python main.py
   \`\`\`

### Example Main Script

To run the project, use the following script in \`main.py\`:

\`\`\`python
import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch import nn, optim

from src.visualization import visualize_slices
from src.data import MedicalDataset
from src.utils import *
from src.model.unet_model import UNet
from src.settings import Settings, Paths

settings = Settings()
settings.load_settings()

paths = Paths(settings=settings)
paths.load_device_paths()

# Create dataset
dataset = MedicalDataset(paths, settings)
train_loader, val_loader, test_loader = dataset.train_test_split()

# Instantiate the model, define the loss function and the optimizer
model = UNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=settings.learning_rate)
model.fit(train_loader, val_loader, optimizer, criterion, num_epochs=settings.num_epochs)

# Training loop
# The model will save npy files in the preprocessed dataset the first time and use saved files subsequently.
\`\`\`

## Experiments

To run experiments, open the Jupyter notebooks in the \`experiments\` folder.

## Project Structure

\`\`\`
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
