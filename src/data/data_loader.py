import torch
from torch.utils.data import Dataset, DataLoader, random_split
import nibabel as nib
import os
import numpy as np
from src.utils import filter_files, filter_negative_files
from tqdm import tqdm


class MedicalDataset(Dataset):
    def __init__(self, paths, settings, data_part='train', transform=None):
        self.img_path = paths.preprocessed_dataset_path + f'{settings.data_type}/{data_part}/images/'
        self.label_path = paths.preprocessed_dataset_path + f'{settings.data_type}/{data_part}/labels/'
        self.img_files = list(np.unique(['_'.join(file.split('_')[:-1]) for file in os.listdir(self.img_path)]))
        self.settings = settings
        self.transform = transform
        self.data_part = data_part

        if settings.data_type == 'psma':
            print(f"psma is chosen to be used for {data_part}")
            self.img_files = [img_file for img_file in self.img_files if 'psma_' in img_file]
        elif settings.data_type == 'fdg':
            print(f"fdg is chosen to be used for {data_part}")
            self.img_files = [img_file for img_file in self.img_files if 'fdg_' in img_file]
        else:
            print(f"Both psma and fdg are chosen to be used for {data_part}")
            pass

        if settings.use_negative_samples is False and settings.data_type == 'fdg':
            self.img_files = filter_negative_files(paths, self.img_files)



    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        ctres_img = np.load(os.path.join(self.img_path, f'{self.img_files[idx]}_CT.npy'))
        suv_img = np.load(os.path.join(self.img_path, f'{self.img_files[idx]}_PET.npy'))
        label_img = np.load(os.path.join(self.label_path, f'{self.img_files[idx]}_label.npy'))

        ctres_img = torch.tensor(ctres_img, dtype=torch.float32)
        suv_img = torch.tensor(suv_img, dtype=torch.float32)
        label_img = torch.tensor(label_img, dtype=torch.long)

        sample = {'ctres': ctres_img, 'suv': suv_img, 'label': label_img, 'file_name': self.img_files[idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def train_test_split(self):
        # Split dataset into train, validation, and test sets
        val_size = int(self.settings.validation_size * len(self.img_files))
        test_size = int(self.settings.test_size * len(self.img_files))
        train_size = len(self.img_files) - val_size - test_size

        train_dataset, val_dataset, test_dataset = random_split(self, [train_size, val_size, test_size])

        # Create dataloaders
        batch_size = self.settings.batch_size
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader



def convert_nib_to_npy(paths):
    output_path = paths.preprocessed_dataset_path
    img_path = paths.raw_dataset_path + 'imagesTr/'
    label_path = paths.raw_dataset_path + 'labelsTr/'

    os.makedirs(output_path, exist_ok=True)

    file_names_img = os.listdir(img_path)
    file_names_label = os.listdir(label_path)

    for file_name in tqdm(file_names_img):
        file_id = '_'.join(file_name.split('_')[:-1])
        ctres_file = f'{file_id}_0000.nii.gz'
        suv_file = f'{file_id}_0001.nii.gz'
        label_file = f'{file_id}.nii.gz'

        ctres_npy = os.path.join(output_path, f'{file_id}_ctres.npy')
        suv_npy = os.path.join(output_path, f'{file_id}_suv.npy')
        label_npy = os.path.join(output_path, f'{file_id}_label.npy')

        # Check if preprocessed files already exist
        if os.path.exists(ctres_npy) and os.path.exists(suv_npy) and os.path.exists(label_npy):
            print(f"Preprocessed files for {file_id} already exist. Skipping.")
            continue

        if (ctres_file in file_names_img) and (suv_file in file_names_img) and (label_file in file_names_label):
            # Load and save preprocessed numpy arrays
            ctres_img = nib.load(os.path.join(img_path, ctres_file)).get_fdata()
            suv_img = nib.load(os.path.join(img_path, suv_file)).get_fdata()
            label_img = nib.load(os.path.join(label_path, label_file)).get_fdata()

            np.save(ctres_npy, ctres_img)
            np.save(suv_npy, suv_img)
            np.save(label_npy, label_img)
