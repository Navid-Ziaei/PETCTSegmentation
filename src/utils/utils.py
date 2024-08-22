import os

import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import numpy as np
import nibabel as nib



def split_data(file_names, test_size=0.2, val_size=0.1):
    train_val_files, test_files = train_test_split(file_names, test_size=test_size)
    train_files, val_files = train_test_split(train_val_files, test_size=val_size)
    return train_files, val_files, test_files


def filter_files(file_names_img, file_names_label, img_path):
    """
    Filter image files based on the presence of corresponding CT, PET, and label files
    and a specific shape condition.

    Parameters:
    file_names_img (list): List of image file names.
    file_names_label (list): List of label file names.
    img_path (str): Path to the image files.

    Returns:
    list: List of filtered file names that meet the criteria.
    """
    filtered_files = []

    # Iterate through the list of image files
    for file_name in file_names_img:
        # Extract the file ID from the file name
        file_id = file_name.split('_')[:-1]

        # Construct the corresponding CT, PET, and label file names
        ctres_file = f'fdg_{file_id[1]}_{file_id[2]}_0000.nii.gz'
        suv_file = f'fdg_{file_id[1]}_{file_id[2]}_0001.nii.gz'
        label_file = f'fdg_{file_id[1]}_{file_id[2]}.nii.gz'

        # Check if the corresponding CT, PET, and label files exist
        if (ctres_file in file_names_img) and (suv_file in file_names_img) and (label_file in file_names_label):
            filtered_files.append(file_name)
        else:
            # Print a message if any of the corresponding files are missing
            print(f"File {file_name} or its corresponding CT, PET, or label file is missing.")

    return filtered_files

def get_slices_with_tumor(label_data):
    num_slices = label_data.shape[-1]
    num_pixel = []
    for slice_idx in range(num_slices):
        num_pixel.append(np.sum(label_data[...,slice_idx]))
    slice_list = np.where(np.array(num_pixel)!=0)[0]
    tumor_size_list = [num_pixel[idx] for idx in slice_list]
    if len(tumor_size_list)>0:
        idx_max = np.argmax(np.array(tumor_size_list))
        biggest_tumor = slice_list[idx_max]
    else:
        idx_max = None
        biggest_tumor = None
    return slice_list, tumor_size_list, biggest_tumor




def find_info_by_id(pid, data_info_fdg, data_info_psma):
    diagnosis = data_info_fdg[data_info_fdg['Subject ID']=='PETCT_'+pid]['diagnosis'].unique().tolist()
    age = data_info_fdg[data_info_fdg['Subject ID']=='PETCT_'+pid]['age'].unique().tolist()
    sex = data_info_fdg[data_info_fdg['Subject ID']=='PETCT_'+pid]['sex'].unique().tolist()
    if len(diagnosis)>0:
        print(f"patient id {pid} \t FDG \t diagnosis: {diagnosis} \t sex: {age} \t Age: {sex}")
    else:
        age = data_info_psma[data_info_psma['Subject ID']=='PSMA_'+pid]['age'].unique().tolist()
        pet_radionuclide = data_info_psma[data_info_psma['Subject ID']=='PSMA_'+pid]['pet_radionuclide'].unique().tolist()
        print(f"patient id {pid} \t PSMA \t pet radionuclide: {pet_radionuclide} \t Age: {age}")


def filter_negative_files(paths, file_names):
    data_info_fdg = pd.read_csv(paths.raw_dataset_path + 'fdg_metadata.csv')
    filtered_data_info_fdg = data_info_fdg[data_info_fdg['diagnosis']!='NEGATIVE']
    list_filtered_id = filtered_data_info_fdg['Subject ID'].tolist()
    list_filtered_id = [pid.split('_')[1] for pid in list_filtered_id]
    list_filtered_filenames = [file_name for file_name in file_names if file_name.split('_')[1] in list_filtered_id]
    list_filtered_filenames = list(np.unique(list_filtered_filenames))

    return list_filtered_filenames

def calculate_stats(data):
    stats = {
        'mean': np.mean(data),
        'median': np.median(data),
        'min': np.min(data),
        'max': np.max(data),
        'std': np.std(data)
    }
    return stats

