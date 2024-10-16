import os
import torch
from torch.utils.data import DataLoader
from torch import nn, optim

from src.model.utils import BCEDiceLoss
from src.visualization import plot_training_history, plot_results
from src.data import MedicalDataset, DataPreprocessor, generate_meta_data
from src.model.unet_model import UNet
from src.settings import Settings, Paths

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device is {device}")

settings = Settings()
settings.load_settings()

paths = Paths(settings=settings)
paths.load_device_paths()

# Just run this for the first time
# generate_meta_data(paths)

# preprocess data
data_preprocessor = DataPreprocessor(paths, settings)
data_preprocessor.preprocess(settings.preprocessing_configs)

# Create dataset
dataset_train = MedicalDataset(paths, settings, data_part='train')
dataset_test = MedicalDataset(paths, settings, data_part='test')
dataset_val = MedicalDataset(paths, settings, data_part='val')

train_loader = DataLoader(dataset_train, batch_size=settings.batch_size, shuffle=True)
val_loader = DataLoader(dataset_val, batch_size=settings.batch_size, shuffle=False)
test_loader = DataLoader(dataset_test, batch_size=settings.batch_size, shuffle=False)


# Instantiate the saved_model, define the loss function and the optimizer
criterion = nn.BCEWithLogitsLoss()
# criterion = BCEDiceLoss()

model = UNet(n_channels=settings.slice_spacing, modality=settings.modality).to(device)
optimizer = optim.Adam(model.parameters(), lr=settings.learning_rate)

history = model.fit(train_loader, val_loader, optimizer, criterion, num_epochs=settings.num_epochs, device=device)
plot_training_history(history, save_path=paths.result_path + 'training_history.png')
model.save_model(paths.model_path + 'model.pt')

df_results = model.evaluate(test_loader, device, criterion)
df_results.to_csv(paths.result_path + "TestResults.csv")
plot_results(df_results, save_dir=paths.result_path)


# Training loop
print("Done!")
