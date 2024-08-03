import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch import nn, optim

from src.visualization import visualize_slices
from src.data import MedicalDataset, DataPreprocessor
from src.utils import *
from src.model.unet_model import UNet
from src.settings import Settings, Paths


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device is {device}")


settings = Settings()
settings.load_settings()

paths = Paths(settings=settings)
paths.load_device_paths()


# preprocess data
# data_preprocessor = DataPreprocessor(paths)
# data_preprocessor.preprocess_and_convert_to_npy()

# Create dataset

dataset = MedicalDataset(paths, settings)
train_loader, val_loader, test_loader = dataset.train_test_split()


# Instantiate the saved_model, define the loss function and the optimizer
model = UNet(num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=settings.learning_rate)
model.fit(train_loader, val_loader, optimizer, criterion, num_epochs=settings.num_epochs, device=device)


# Training loop
print("Done!")