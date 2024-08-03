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


def calculate_stats(data):
    stats = {
        'mean': np.mean(data),
        'min': np.min(data),
        'max': np.max(data),
        'std': np.std(data)
    }
    return stats

