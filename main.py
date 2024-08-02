import os

from torch import nn, optim

from src.visualization import visualize_slices
from src.data import MedicalDataset
from src.utils import *
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from src.model.unet_model import UNet



data_path = "D:\\Datasets\\PetCT\\"

img_path = data_path + 'imagesTr\\'
label_path = data_path + 'labelsTr\\'

file_names_img = os.listdir(img_path)
file_names_label = os.listdir(label_path)

print(f"Number of files: {len(file_names_img)}")

ctres_data, suv_data, label_data = visualize_slices(file_id='89491',
                                                    slice_index=50,
                                                    file_names_img=file_names_img,
                                                    file_names_label=file_names_label,
                                                    img_path=img_path,
                                                    label_path=label_path)
print(ctres_data.shape)
ctres_stats1 = calculate_stats(ctres_data)
suv_stats1 = calculate_stats(suv_data)
label_stats1 = calculate_stats(label_data)

ctres_data, suv_data, label_data = visualize_slices(file_id='29055',
                                                    slice_index=50,
                                                    file_names_img=file_names_img,
                                                    file_names_label=file_names_label,
                                                    img_path=img_path,
                                                    label_path=label_path)

print(ctres_data.shape)



ctres_stats = calculate_stats(ctres_data)
suv_stats = calculate_stats(suv_data)
label_stats = calculate_stats(label_data)
# Identify files containing '33529'

filtered_file_names_img = filter_files(file_names_img, file_names_label, img_path)


# Create dataset
dataset = MedicalDataset(filtered_file_names_img, img_path, label_path)

# Split dataset into train, validation, and test sets
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Create dataloaders
batch_size = 4
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Instantiate the model, define the loss function and the optimizer
model = UNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch in train_loader:
        ctres = batch['ctres'].unsqueeze(1)  # Add channel dimension
        suv = batch['suv'].unsqueeze(1)  # Add channel dimension
        label = batch['label']

        # Concatenate CTres and SUV images along the channel dimension
        inputs = torch.cat((ctres, suv), dim=1)

        optimizer.zero_grad()
        outputs = model(ctres)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            ctres = batch['ctres'].unsqueeze(1)  # Add channel dimension
            suv = batch['suv'].unsqueeze(1)  # Add channel dimension
            label = batch['label']

            # Concatenate CTres and SUV images along the channel dimension
            inputs = torch.cat((ctres, suv), dim=1)

            outputs = model(inputs)
            loss = criterion(outputs, label)

            val_loss += loss.item()

    val_epoch_loss = val_loss / len(val_loader)
    print(f'Validation Loss: {val_epoch_loss:.4f}')