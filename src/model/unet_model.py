import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.evaluation.metrics import compute_metrics, false_neg_pix, false_pos_pix, dice_score


class UNet(nn.Module):
    def __init__(self, n_channels, modality='pet'):
        super(UNet, self).__init__()
        self.modality = modality
        self.encoder = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, n_channels, kernel_size=1)  # Ensure num_classes output channels
        )

    def forward(self, x):
        # From [batch_size, height, width, num_channels] to [batch_size, num_channels, height, width]
        x = x.permute(0, 3, 1, 2)
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def fit(self, train_loader, val_loader, optimizer, criterion, num_epochs, device):
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'train_dice': [],
            'val_dice': []
        }

        for epoch in range(num_epochs):
            self.train()
            running_loss = 0.0
            correct = 0
            total = 0
            dice_score_total = 0.0

            for batch in train_loader:
                ctres = batch['ctres'].to(device)
                suv = batch['suv'].to(device)
                label = batch['label'].to(device)
                label = label.permute(0, 3, 1, 2)

                if self.modality == 'both':
                    inputs = torch.cat((ctres, suv), dim=1)
                elif self.modality == 'ct':
                    inputs = ctres
                elif self.modality == 'pet':
                    inputs = suv
                else:
                    raise ValueError("the data modality is not defined")

                optimizer.zero_grad()
                outputs = self(inputs)

                loss = criterion(outputs, label.float())
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                # Calculate accuracy
                predicted = torch.sigmoid(outputs) > 0.5
                correct += (predicted == label).float().sum().item()
                total += label.numel()

                # Calculate Dice score
                dice_score_total += self.dice_score(outputs, label)

            epoch_loss = running_loss / len(train_loader)
            epoch_accuracy = correct / total
            epoch_dice = dice_score_total / len(train_loader)

            history['train_loss'].append(epoch_loss)
            history['train_accuracy'].append(epoch_accuracy)
            history['train_dice'].append(epoch_dice)

            # print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.5f}, Accuracy: {epoch_accuracy:.5f}, Dice Score: {epoch_dice:.5f}')

            self.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            val_dice_score_total = 0.0

            with torch.no_grad():
                for batch in val_loader:
                    ctres = batch['ctres'].to(device)
                    suv = batch['suv'].to(device)
                    label = batch['label'].to(device)
                    label = label.permute(0, 3, 1, 2)

                    if self.modality == 'both':
                        inputs = torch.cat((ctres, suv), dim=1)
                    elif self.modality == 'ct':
                        inputs = ctres
                    elif self.modality == 'pet':
                        inputs = suv
                    else:
                        raise ValueError("the data modality is not defined")

                    outputs = self(inputs)
                    loss = criterion(outputs, label.float())

                    val_loss += loss.item()

                    predicted = torch.sigmoid(outputs) > 0.5
                    val_correct += (predicted == label).float().sum().item()
                    val_total += label.numel()

                    val_dice_score_total += self.dice_score(outputs, label)

            val_epoch_loss = val_loss / len(val_loader)
            val_epoch_accuracy = val_correct / val_total
            val_epoch_dice = val_dice_score_total / len(val_loader)

            history['val_loss'].append(val_epoch_loss)
            history['val_accuracy'].append(val_epoch_accuracy)
            history['val_dice'].append(val_epoch_dice)

            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.5f}, Accuracy: {epoch_accuracy:.5f}, Dice Score: {epoch_dice:.5f} Validation Loss: {val_epoch_loss:.5f}, Validation Accuracy: {val_epoch_accuracy:.5f}, Validation Dice Score: {val_epoch_dice:.5f}')

        return history
    def evaluate(self, test_loader, device, criterion):
        self.eval()
        test_loss = 0.0
        output_dict, gt_dict = {}, {}
        with torch.no_grad():
            for batch in test_loader:
                ctres = batch['ctres'].to(device)  # Add channel dimension
                suv = batch['suv'].to(device)  # Add channel dimension
                label = batch['label'].to(device)
                label = label.permute(0, 3, 1, 2)
                file_name = batch['file_name']

                # Concatenate CTres and SUV images along the channel dimension
                if self.modality == 'both':
                    inputs = torch.cat((ctres, suv), dim=1)
                elif self.modality == 'ct':
                    inputs = ctres
                elif self.modality == 'pet':
                    inputs = suv
                else:
                    raise ValueError("the data modality is not defined")

                outputs = self(inputs)
                for i in range(len(file_name)):
                    output_dict[file_name[i]] = outputs[i]
                    gt_dict[file_name[i]] = label[i]

                loss = criterion(outputs, label.float())

                test_loss += loss.item()

        file_names = list(np.unique(['_'.join(key.split('_')[:-1]) for key in list(gt_dict.keys())]))
        df_meta = pd.read_csv("./meta data/meta_data.csv")
        results = []
        for f_name in tqdm(file_names):
            slice_idx = [int(key.split('_')[-1]) for key in list(gt_dict.keys()) if f_name in key]
            slice_idx_in_order = list(np.argsort(slice_idx))
            pred_list = [output_dict[key].detach().cpu() for key in list(gt_dict.keys()) if f_name in key]
            gt_list = [gt_dict[key].detach().detach().cpu() for key in list(gt_dict.keys()) if f_name in key]

            pred_list_ordered = [torch.sigmoid(pred_list[idx]) > 0.5 for idx in slice_idx_in_order]
            gt_list_ordered = [gt_list[idx]> 0.5 for idx in slice_idx_in_order]


            pred_array = np.concatenate(pred_list_ordered, axis=0)
            gt_array = np.concatenate(gt_list_ordered, axis=0)

            voxel_vol = df_meta['voxel volume'][df_meta['file name']==f_name].values
            num_slides = df_meta['num of slices'][df_meta['file name'] == f_name].values

            false_neg_vol = false_neg_pix(gt_array, pred_array) * voxel_vol
            false_pos_vol = false_pos_pix(gt_array, pred_array) * voxel_vol
            dice_sc = dice_score(gt_array, pred_array)

            result = {
                'file name': f_name,
                'false_neg_vol': false_neg_vol[0],
                'false_pos_vol': false_pos_vol[0],
                'dice_sc': dice_sc
            }
            results.append(result)
        df_results = pd.DataFrame(results)

        return df_results


    def save_model(self, path):
        """Save the model's state dictionary to the specified path."""
        torch.save(self.state_dict(), path)
        print(f'Model saved to {path}')

    def load_model(self, path, device):
        """Load the model's state dictionary from the specified path."""
        self.load_state_dict(torch.load(path, map_location=device))
        self.to(device)
        print(f'Model loaded from {path}')

    def dice_score(self, outputs, labels):
        smooth = 1e-6
        outputs = torch.sigmoid(outputs) > 0.5
        labels = labels > 0.5

        intersection = (outputs & labels).float().sum((1, 2, 3))
        union = outputs.float().sum((1, 2, 3)) + labels.float().sum((1, 2, 3))
        dice = (2. * intersection + smooth) / (union + smooth)
        return dice.mean().item()
