import os
import shutil
import numpy as np
import nibabel as nib
import pandas as pd

from scipy.ndimage import zoom
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from .load_data_utils import download_fdg_by_id, download_psma_by_id, load_data_by_id, load_data_by_file_name


class DataPreprocessor:
    def __init__(self, paths, settings):
        self.loaded_psma_files = None
        self.loaded_fdg_files = None
        self.settings = settings
        self.raw_dataset_path = paths.raw_dataset_path
        self.preprocessed_dataset_path = paths.preprocessed_dataset_path
        self.check_data()

    def apply_preprocessing(self, data_ct, data_pet, labels, **kwargs):
        for step, params in kwargs.items():
            if step == 'scale_pet':
                data_pet = self.scale_pet(data_pet, **params)
            if step == 'scale_ct':
                data_ct = self.scale_ct(data_ct, **params)
            if step == 'resample_image':
                data_ct = self.resample_image(data_ct, **params)
                data_pet = self.resample_image(data_pet, **params)
                labels = self.resample_image(labels, **params)
                labels = (labels > 0.5) * 1.0

        return data_ct, data_pet, labels

    def preprocess(self, preprocessing_configs):
        img_path = os.path.join(self.raw_dataset_path, 'imagesTr/')
        label_path = os.path.join(self.raw_dataset_path, 'labelsTr/')

        file_names_img = list(np.unique(os.listdir(img_path)))
        file_names_label = list(np.unique(os.listdir(label_path)))

        if self.settings.load_preprocessed_data is True:
            pass
        else:
            # self.clear_train_test_val_folders()
            self.train_test_split()

            for file_name in tqdm(self.file_name_list):
                ctres_img, suv_img, label_img, *_ = load_data_by_file_name(file_name, file_names_img,
                                                                       file_names_label, img_path, label_path)

                ctres_img, suv_img, label_img = self.apply_preprocessing(data_ct=ctres_img,
                                                                         data_pet=suv_img,
                                                                         labels=label_img,
                                                                         **preprocessing_configs)

                self.prepare_data(file_name, ctres_img, suv_img, label_img, file_name=file_name)

                """
                ctres_npy = os.path.join(self.preprocessed_dataset_path, f'{file_id}_ctres.npy')
                suv_npy = os.path.join(self.preprocessed_dataset_path, f'{file_id}_suv.npy')
                label_npy = os.path.join(self.preprocessed_dataset_path, f'{file_id}_label.npy')
                
                if os.path.exists(ctres_npy) and os.path.exists(suv_npy) and os.path.exists(label_npy):
                    print(f"Preprocessed files for {file_id} already exist. Skipping.")
                else:
                    ctres_img = self.load_nifti(ctres_file)
                    suv_img = self.load_nifti(suv_file)
                    label_img = self.load_nifti(label_file)
    
                    ctres_img, suv_img = self.apply_preprocessing(data_ct=ctres_img,
                                                                  data_pet=suv_img, **preprocessing_configs)
    
                    self.save_npy(ctres_img, ctres_npy)
                    self.save_npy(suv_img, suv_npy)
                    self.save_npy(label_img, label_npy)
                """

    def prepare_data(self, file_id, ctres_img, suv_img, label_img, file_name):
        """
        Prepare the data for training, validation, and testing

        :param file_id:
        :param ctres_img:
        :param suv_img:
        :param label_img:
        :return:
        """
        # save the data in the appropriate folder
        if file_id in self.train_files:
            save_path = os.path.join(self.preprocessed_dataset_path, 'train')
        elif file_id in self.val_files:
            save_path = os.path.join(self.preprocessed_dataset_path, 'val')
        elif file_id in self.test_files:
            save_path = os.path.join(self.preprocessed_dataset_path, 'test')
        else:
            save_path = None

        # The images are WxHxD where D is number of slices
        # Here we want to transform each image to M WxHxD1 images where MxD1=D
        # Also we have mode to have overlapping between slides
        if save_path is not None:
            self.split_slices(ctres_img, suv_img, label_img, file_name, file_id, save_path)

    def train_test_split(self):
        # select the data type
        if self.settings.data_type == 'fdg':
            self.file_name_list = self.loaded_fdg_files
        elif self.settings.data_type == 'psma':
            self.file_name_list = self.loaded_psma_files
        elif self.settings.data_type == 'both':
            self.file_name_list = self.loaded_fdg_files + self.loaded_psma_files
        else:
            raise ValueError("Invalid data type. Please select 'fdg', 'psma', or 'both'.")

        # trian/ validation/ test spilit
        train_files, val_files, test_files = self.split_data(self.file_name_list)
        self.train_size = len(train_files)
        self.val_size = len(val_files)
        self.test_size = len(test_files)

        # find if there is any file in file_list that is not in any of the train/val/test
        excluded_files = [file for file in self.file_name_list if file not in train_files + val_files + test_files]
        print(f"Excluded files: {excluded_files}")
        train_files = train_files + excluded_files

        self.train_files = train_files
        self.val_files = val_files
        self.test_files = test_files
        return train_files, val_files, test_files

    def split_slices(self, ctres_img, suv_img, label_img, file_name, file_id, save_path):
        file_type = file_name.split('_')[0]
        ctres_slices = ctres_img.shape[-1]
        suv_slices = suv_img.shape[-1]
        label_slices = label_img.shape[-1]

        if ctres_slices != suv_slices or ctres_slices != label_slices:
            print(f"Number of slices in CT, PET, and label images do not match in {file_name}: "
                  f"labels: {label_slices} , CT: {suv_slices}, PET: {ctres_slices}.")
        else:
            slice_win_size = self.settings.slice_spacing
            slice_overlap = self.settings.slice_overlap

            for slice_idx, idx in enumerate(range(0, ctres_slices, slice_win_size - slice_overlap)):
                ctres_slice = ctres_img[..., slice_idx:slice_idx + slice_win_size]
                suv_slice = suv_img[..., slice_idx:slice_idx + slice_win_size]
                label_slice = label_img[..., slice_idx:slice_idx + slice_win_size]

                # pad if the number of slices in image is less than slice_win_size
                ctres_slice = self.pad_slices(ctres_slice, slice_win_size)
                suv_slice = self.pad_slices(suv_slice, slice_win_size)
                label_slice = self.pad_slices(label_slice, slice_win_size)

                ctres_file = f'{save_path}/images/{file_id}_{idx}_CT.npy'
                suv_file = f'{save_path}/images/{file_id}_{idx}_PET.npy'
                label_file = f'{save_path}/labels/{file_id}_{idx}_label.npy'

                self.save_npy(ctres_slice, ctres_file)
                self.save_npy(suv_slice, suv_file)
                self.save_npy(label_slice, label_file)

    def split_data(self, file_list):
        train_files, test_files = train_test_split(file_list, test_size=self.settings.test_size)
        train_files, val_files = train_test_split(train_files, test_size=self.settings.validation_size)
        return train_files, val_files, test_files

    def pad_slices(self, image_slices, target_slices):
        current_slices = image_slices.shape[-1]
        if current_slices < target_slices:
            print(f"Padding {target_slices - current_slices} slices")
            padding = target_slices - current_slices
            pad_width = ((0, 0), (0, 0), (0, padding))
            padded_image = np.pad(image_slices, pad_width, mode='constant', constant_values=0)
            return padded_image
        return image_slices

    def check_data(self):
        data_info_fdg = pd.read_csv(self.raw_dataset_path + 'fdg_metadata.csv')
        data_info_psma = pd.read_csv(self.raw_dataset_path + 'psma_metadata.csv')

        img_path = self.raw_dataset_path + 'imagesTr/'
        label_path = self.raw_dataset_path + 'labelsTr/'

        file_names = os.listdir(img_path)

        file_names_ct = ['_'.join(file_name.split('_')[:-1]) for file_name in file_names if '_0000' in file_name]
        file_names_pet = ['_'.join(file_name.split('_')[:-1]) for file_name in file_names if '_0001' in file_name]

        mismatch1 = [file_name for file_name in file_names_pet if file_name not in file_names_ct]
        mismatch2 = [file_name for file_name in file_names_ct if file_name not in file_names_pet]

        print(f"Files present in PET not in CT: {mismatch1}")
        print(f"Files present in CT not in PET: {mismatch2}")

        for file_name in mismatch1:
            download_fdg_by_id(file_name.split('_')[1], data_info_fdg, self.raw_dataset_path)
            download_psma_by_id(file_name.split('_')[1], data_info_psma, self.raw_dataset_path)
            file_names_ct.append(file_name)
        for file_name in mismatch2:
            download_fdg_by_id(file_name.split('_')[1], data_info_fdg, self.raw_dataset_path)
            download_psma_by_id(file_name.split('_')[1], data_info_psma, self.raw_dataset_path)
            file_names_pet.append(file_name)

        print(f"Number of total PSMA patients: {len(data_info_psma['Subject ID'].unique())}")
        print(f"Number of total FDG patients: {len(data_info_fdg['Subject ID'].unique())}")

        loaded_file_names = file_names_ct

        fdg_unique_id = data_info_fdg['Subject ID'].unique()
        psma_unique_id = data_info_psma['Subject ID'].unique()

        laded_psma_files = [fid for fid in loaded_file_names if 'PETCT_' + fid.split('_')[1] not in fdg_unique_id]
        laded_fdg_files = [fid for fid in loaded_file_names if 'PSMA_' + fid.split('_')[1] not in psma_unique_id]

        print(f"Number of loaded PSMA patients: {len(laded_psma_files)}")
        print(f"Number of loaded FDG patients: {len(laded_fdg_files)}")

        print(f"=====================================================================================")
        self.loaded_fdg_files = laded_fdg_files
        self.loaded_psma_files = laded_psma_files

        return laded_fdg_files, laded_psma_files

    def create_train_test_val_folders(self):
        if os.path.exists(self.preprocessed_dataset_path + self.settings.data_type + '/train/images/') is False:
            os.makedirs(self.preprocessed_dataset_path + self.settings.data_type + '/train/images/')
        if os.path.exists(self.preprocessed_dataset_path + self.settings.data_type + '/test/images/') is False:
            os.makedirs(self.preprocessed_dataset_path + self.settings.data_type + '/test/images/')
        if os.path.exists(self.preprocessed_dataset_path + self.settings.data_type + '/val/images/') is False:
            os.makedirs(self.preprocessed_dataset_path + self.settings.data_type + '/val/images/')
        if os.path.exists(self.preprocessed_dataset_path + self.settings.data_type + '/train/labels/') is False:
            os.makedirs(self.preprocessed_dataset_path + self.settings.data_type + '/train/labels/')
        if os.path.exists(self.preprocessed_dataset_path + self.settings.data_type + '/test/labels/') is False:
            os.makedirs(self.preprocessed_dataset_path + self.settings.data_type + '/test/labels/')
        if os.path.exists(self.preprocessed_dataset_path + self.settings.data_type + '/val/labels/') is False:
            os.makedirs(self.preprocessed_dataset_path + self.settings.data_type + '/val/labels/')

    def clear_train_test_val_folders(self):
        if os.path.exists(self.preprocessed_dataset_path + self.settings.data_type + '/train/'):
            shutil.rmtree(self.preprocessed_dataset_path + self.settings.data_type + '/train/')
        if os.path.exists(self.preprocessed_dataset_path + self.settings.data_type + '/test/'):
            shutil.rmtree(self.preprocessed_dataset_path + self.settings.data_type + '/test/')
        if os.path.exists(self.preprocessed_dataset_path + self.settings.data_type + '/val/'):
            shutil.rmtree(self.preprocessed_dataset_path + self.settings.data_type + '/val/')

    def pad_slices(self, image_slices, target_slices):
        current_slices = image_slices.shape[0]
        if current_slices < target_slices:
            padding = target_slices - current_slices
            pad_width = ((0, padding), (0, 0), (0, 0))
            padded_image = np.pad(image_slices, pad_width, mode='constant', constant_values=0)
            return padded_image
        return image_slices

    def resample_image(self, image, **kwargs):
        target_shape = kwargs['target_shape'] + (image.shape[-1],)
        zoom_factors = [t / f for t, f in zip(target_shape, image.shape)]
        return zoom(image, zoom_factors, order=1)

    def load_nifti(self, file_path):
        return nib.load(file_path).get_fdata()

    def save_npy(self, data, file_path):
        np.save(file_path, data)

    def scale_pet(self, data_pet, **kwargs):
        if kwargs['mode'] == 'normalization':
            normalized_data = (data_pet - kwargs['min']) / (kwargs['max'] - kwargs['min'])
            normalized_data[normalized_data > 1] = 1
            normalized_data[normalized_data < 0] = 0
            return normalized_data
        elif kwargs['mode'] == 'standardization':
            return (data_pet - np.mean(data_pet)) / np.std(data_pet)
        else:
            raise ValueError(f"Invalid mode: {kwargs['mode']}")

    def scale_ct(self, data_ct, **kwargs):
        if kwargs['mode'] == 'normalization':
            normalized_data = (data_ct - kwargs['min']) / (kwargs['max'] - kwargs['min'])
            normalized_data[normalized_data > 1] = 1
            normalized_data[normalized_data < 0] = 0
            return normalized_data
        elif kwargs['mode'] == 'standardization':
            return (data_ct - np.mean(data_ct)) / np.std(data_ct)
        else:
            raise ValueError(f"Invalid mode: {kwargs['mode']}")
