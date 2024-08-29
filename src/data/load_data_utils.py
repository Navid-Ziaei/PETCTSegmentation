import os
import urllib.request
import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils import get_slices_with_tumor


def download_fdg_by_id(fid, data_info_fdg, raw_dataset_path):
    file_to_download = data_info_fdg[data_info_fdg['Subject ID'] == 'PETCT_' + fid]
    if len(file_to_download['File Location']) == 0:
        print("No file found in FDG dataset")
    else:
        print(f"Found {len(file_to_download['File Location'])} files in FDG dataset")
    for file_name, modality in zip(file_to_download['File Location'], file_to_download['Modality']):

        file_loc = file_name.split('/')
        if modality in ['CT']:
            f_name = 'fdg_' + file_loc[2].split('_')[1] + '_' + file_loc[3] + '_0000.nii.gz'
            f_path = raw_dataset_path + 'imagesTr/'
        elif modality in ['PT']:
            f_name = 'fdg_' + file_loc[2].split('_')[1] + '_' + file_loc[3] + '_0001.nii.gz'
            f_path = raw_dataset_path + 'imagesTr/'
        elif modality in ['SEG']:
            f_name = 'fdg_' + file_loc[2].split('_')[1] + '_' + file_loc[3] + '.nii.gz'
            f_path = raw_dataset_path + 'labelsTr/'
        else:
            raise ValueError("No modality found in the FDG dataset")

        if os.path.exists(f_path + f_name):
            pass
        else:
            add_f_name = f_name.replace(' ', '%20')
            print(f"Downloading {f_name} ...")
            urllib.request.urlretrieve(
                f"https://syncandshare.lrz.de/dl/fiCJ6mQcjefMTQdKJsBSys/imagesTr/{add_f_name}",
                filename=f_path + f_name)


def download_psma_by_id(fid, data_info_psma, raw_dataset_path):
    file_to_download = data_info_psma[data_info_psma['Subject ID'] == 'PSMA_' + fid]


    for file_name, file_date in zip(file_to_download['Subject ID'], file_to_download['Study Date']):

        file_loc = file_name.split('_')[1]
        f_name = 'psma_' + file_loc + '_' + file_date + '_0000.nii.gz'
        f_path = raw_dataset_path + 'imagesTr/'

        if os.path.exists(f_path + f_name):
            pass
        else:
            add_f_name = f_name.replace(' ', '%20')
            print(f"Downloading {f_name} ...")
            urllib.request.urlretrieve(
                f"https://syncandshare.lrz.de/dl/fiCJ6mQcjefMTQdKJsBSys/imagesTr/{add_f_name}",
                filename=f_path + f_name)

        f_name = 'psma_' + file_loc + '_' + file_date + '_0001.nii.gz'
        f_path = raw_dataset_path + 'imagesTr/'

        if os.path.exists(f_path + f_name):
            pass
        else:
            add_f_name = f_name.replace(' ', '%20')
            print(f"Downloading {f_name} ...")
            urllib.request.urlretrieve(
                f"https://syncandshare.lrz.de/dl/fiCJ6mQcjefMTQdKJsBSys/imagesTr/{add_f_name}",
                filename=f_path + f_name)
def load_data_by_file_name(file_name, file_names_img, file_names_label, img_path, label_path):
    ctres_file = None
    suv_file = None
    label_file = None
    file_list = [f for f in file_names_img if file_name in f]
    file_list_label = [f for f in file_names_label if file_name in f]

    # Identify image files containing the specified file_id
    for file_name in file_list:
        if '_0000' in file_name:
            ctres_file = file_name
        elif '_0001' in file_name:
            suv_file = file_name
    label_file = file_list_label[0]

    if ctres_file is not None and suv_file is not None and label_file is not None:
        if len(file_list_label) > 1:
            raise ValueError(f"More than one label found for {file_name}")
        if len(file_list) != 2:
            raise ValueError(f"File found is not correct for {file_name} {file_list}")

        # Load the files using nibabel
        ctres_img = nib.load(os.path.join(img_path, ctres_file))
        suv_img = nib.load(os.path.join(img_path, suv_file))
        label_img = nib.load(os.path.join(label_path, label_file))

        # Get the data from the images
        ctres_data = ctres_img.get_fdata()
        suv_data = suv_img.get_fdata()
        label_data = label_img.get_fdata()

        pixdim = label_img.header['pixdim']
        voxel_vol = pixdim[1] * pixdim[2] * pixdim[3] / 1000

    else:
        raise ValueError(f"File not found {file_name}")
    return ctres_data, suv_data, label_data, voxel_vol
def load_data_by_id(file_id, file_names_img, file_names_label, img_path, label_path):
    ctres_file = None
    suv_file = None
    label_file = None
    file_list = [file_name for file_name in file_names_img if file_id in file_name.split('_')[1]]
    file_list_label = [file_name for file_name in file_names_label if file_id in file_name.split('_')[1]]


    # Identify image files containing the specified file_id
    for file_name in file_list:
        if '_0000' in file_name:
            ctres_file = file_name
        elif '_0001' in file_name:
            suv_file = file_name
    label_file = file_list_label[0]

    if ctres_file is not None and suv_file is not None and label_file is not None:
        if len(file_list_label) > 1:
            raise ValueError(f"More than one label found for {file_id}")
        if len(file_list) != 2:
            raise ValueError(f"File found is not correct for {file_id} {file_list}")


        # Load the files using nibabel
        ctres_img = nib.load(os.path.join(img_path, ctres_file))
        suv_img = nib.load(os.path.join(img_path, suv_file))
        label_img = nib.load(os.path.join(label_path, label_file))

        # Get the data from the images
        ctres_data = ctres_img.get_fdata()
        suv_data = suv_img.get_fdata()
        label_data = label_img.get_fdata()

        pixdim = label_img.header['pixdim']
        voxel_vol = pixdim[1] * pixdim[2] * pixdim[3] / 1000
    else:
        print("No file is found")
        return None, None, None, None
    return ctres_data, suv_data, label_data, label_file.split('.')[0], voxel_vol


def generate_meta_data(paths):
    img_path = paths.raw_dataset_path + 'imagesTr/'
    label_path = paths.raw_dataset_path + 'labelsTr/'

    file_names_img = os.listdir(img_path)
    file_names_label = os.listdir(label_path)

    file_ids = list(np.unique([file_name.split('_')[1] for file_name in file_names_img]))
    file_names = list(np.unique(['_'.join(f_name.split('_')[:-1]) for f_name in file_names_img]))

    data_info_list = []
    for file_name in tqdm(file_names):
        try:
            ctres_data, suv_data, label_data, voxel_vol = load_data_by_file_name(file_name, file_names_img,
                                                                                 file_names_label, img_path, label_path)
            slice_list, tumor_size_list, biggest_tumor = get_slices_with_tumor(label_data)

            data_info = {'file name': file_name,
                         'data type': file_name.split('_')[0],
                         'num of slices': ctres_data.shape[-1],
                         'slice_size': ctres_data.shape[:-1],
                         'voxel volume': voxel_vol,
                         'num_tumor_slices': len(slice_list),
                         'tumor_slices': str(slice_list)}
            data_info_list.append(data_info)
        except:
            print(f"Error in loading {file_name}")

    df_info = pd.DataFrame(data_info_list)
    df_info.to_csv("meta data/meta_data.csv", index=False)
