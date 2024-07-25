from sklearn.model_selection import train_test_split
import numpy as np
import nibabel as nib
def split_data(file_names, test_size=0.2, val_size=0.1):
    train_val_files, test_files = train_test_split(file_names, test_size=test_size)
    train_files, val_files = train_test_split(train_val_files, test_size=val_size)
    return train_files, val_files, test_files


def filter_files(file_names_img, file_names_label, img_path):
    filtered_files = []
    for file_name in file_names_img:
        file_id = file_name.split('_')[:-1]

        # Check for corresponding CT and PET files
        ctres_file = f'fdg_{file_id[1]}_{file_id[2]}_0000.nii.gz'
        suv_file = f'fdg_{file_id[1]}_{file_id[2]}_0001.nii.gz'
        label_file = f'fdg_{file_id[1]}_{file_id[2]}.nii.gz'



        if (ctres_file in file_names_img) and (suv_file in file_names_img) and (label_file in file_names_label):
            ctres_img = nib.load(img_path + ctres_file).get_fdata()
            print(ctres_img.shape)
            if ctres_img.shape[-1] == 326:
                filtered_files.append(file_name)
        else:
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