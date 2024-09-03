import json
import os
import subprocess
from nnunetv2.preprocessing import cropping

import SimpleITK
import torch
from torch.utils.data import DataLoader

from src.data import DataPreprocessor, MedicalDataset
from src.model import UNet
from src.settings import Settings, Paths


class Autopet_mymodel:
    def __init__(self):
        """
        Write your own input validators here
        Initialize your model etc.
        """
        use_win = True
        # set some paths and parameters
        # according to the specified grand-challenge interfaces
        self.input_path = "/input/"
        # according to the specified grand-challenge interfaces
        self.output_path = "/output/images/automated-petct-lesion-segmentation/"
        self.nii_path = "/opt/algorithm/data/imagesTs"
        self.result_path = "/opt/algorithm/nnUNet_raw_data_base/nnUNet_raw_data/Task001_TCIA/result"
        self.model_path = "/opt/algorithm/model/"
        self.nii_seg_file = "TCIA_001.nii.gz"
        self.output_path_category = "/output/data-centric-model.json"

        if use_win is True:
            self.input_path = "D:/Datasets/PetCT" + self.input_path
            self.output_path = "D:/Datasets/PetCT:" + self.output_path
            self.nii_path = "D:/Datasets/PetCT" + self.nii_path
            self.result_path = "D:/Datasets/PetCT" + self.result_path
            self.model_path = "D:/Datasets/PetCT" + self.model_path
            self.output_path_category = "D:/Datasets/PetCT" + self.output_path_category


        pass

    def convert_mha_to_nii(self, mha_input_path, nii_out_path):  # nnUNet specific
        img = SimpleITK.ReadImage(mha_input_path)
        SimpleITK.WriteImage(img, nii_out_path, True)

    def convert_nii_to_mha(self, nii_input_path, mha_out_path):  # nnUNet specific
        img = SimpleITK.ReadImage(nii_input_path)
        SimpleITK.WriteImage(img, mha_out_path, True)
    def check_gpu(self):
        """
        Check if GPU is available
        """
        print("Checking GPU availability")
        is_available = torch.cuda.is_available()
        print("Available: " + str(is_available))
        print(f"Device count: {torch.cuda.device_count()}")
        if is_available:
            print(f"Current device: {torch.cuda.current_device()}")
            print("Device name: " + torch.cuda.get_device_name(0))
            print(
                "Device memory: "
                + str(torch.cuda.get_device_properties(0).total_memory)
            )

    def write_outputs(self, uuid):
        """
        Write to /output/
        Check https://grand-challenge.org/algorithms/interfaces/
        """
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        self.convert_nii_to_mha(
            os.path.join(self.result_path, self.nii_seg_file),
            os.path.join(self.output_path, uuid + ".mha"),
        )
        print("Output written to: " + os.path.join(self.output_path, uuid + ".mha"))

    def load_inputs(self):
        """
        Read from /input/
        Check https://grand-challenge.org/algorithms/interfaces/
        """
        ct_mha = os.listdir(os.path.join(self.input_path, "images/ct/"))
        pet_mha = os.listdir(os.path.join(self.input_path, "images/pet/"))

        uuid = []
        for i in range(len(ct_mha)):
            uuid.append(os.path.splitext(ct_mha[i]))
            self.convert_mha_to_nii(
                os.path.join(self.input_path, "images/ct/", ct_mha[i]),
                os.path.join(self.nii_path, f"TCIA_{i+1:03}_0000.nii.gz"),
            )
            self.convert_mha_to_nii(
                os.path.join(self.input_path, "images/pet/", pet_mha[i]),
                os.path.join(self.nii_path, f"TCIA_{i+1:03}_0001.nii.gz"),
            )
        return uuid

    def process(self):
        self.check_gpu()

        uuid = self.load_inputs()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Start processing")

        settings = Settings()
        settings.load_settings(config_folder=self.model_path)

        paths = Paths(settings=settings)
        paths.load_device_paths(config_folder=self.model_path)

        data_preprocessor = DataPreprocessor(paths, settings)
        data_preprocessor.preprocess(settings.preprocessing_configs, mode='test')

        dataset_test = MedicalDataset(paths, settings, data_part='test')

        test_loader = DataLoader(dataset_test, batch_size=settings.batch_size, shuffle=False)

        model = UNet(n_channels=settings.slice_spacing, modality=settings.modality).to(device)
        model.load_model(paths.model_path + 'model.pt')

        pred_array = model.predict(test_loader, device)

        self.write_outputs(pred_array)

class Autopet_baseline:

    def __init__(self):
        """
        Write your own input validators here
        Initialize your model etc.
        """
        # set some paths and parameters
        # according to the specified grand-challenge interfaces
        self.input_path = "/input/"
        # according to the specified grand-challenge interfaces
        self.output_path = "/output/images/automated-petct-lesion-segmentation/"
        self.nii_path = (
            "/opt/algorithm/nnUNet_raw_data_base/nnUNet_raw_data/Task001_TCIA/imagesTs"
        )
        self.result_path = (
            "/opt/algorithm/nnUNet_raw_data_base/nnUNet_raw_data/Task001_TCIA/result"
        )
        self.nii_seg_file = "TCIA_001.nii.gz"
        self.output_path_category = "/output/data-centric-model.json"

        """
        # set some paths and parameters
        # according to the specified grand-challenge interfaces
        self.input_path = "D:/projects/autoPETIII-main/test/input/"
        # according to the specified grand-challenge interfaces
        self.output_path = "D:/Datasets/PetCT/output/images/automated-petct-lesion-segmentation/"
        self.nii_path = (
            "D:/Datasets/PetCT/opt/algorithm/nnUNet_raw_data_base/nnUNet_raw_data/Task001_TCIA/imagesTs"
        )
        self.result_path = (
            "D:/Datasets/PetCT/opt/algorithm/nnUNet_raw_data_base/nnUNet_raw_data/Task001_TCIA/result"
        )
        self.nii_seg_file = "TCIA_001.nii.gz"
        self.output_path_category = "D:/Datasets/PetCT/output/data-centric-model.json
        """

        pass

    def convert_mha_to_nii(self, mha_input_path, nii_out_path):  # nnUNet specific
        img = SimpleITK.ReadImage(mha_input_path)
        SimpleITK.WriteImage(img, nii_out_path, True)

    def convert_nii_to_mha(self, nii_input_path, mha_out_path):  # nnUNet specific
        img = SimpleITK.ReadImage(nii_input_path)
        SimpleITK.WriteImage(img, mha_out_path, True)

    def check_gpu(self):
        """
        Check if GPU is available
        """
        print("Checking GPU availability")
        is_available = torch.cuda.is_available()
        print("Available: " + str(is_available))
        print(f"Device count: {torch.cuda.device_count()}")
        if is_available:
            print(f"Current device: {torch.cuda.current_device()}")
            print("Device name: " + torch.cuda.get_device_name(0))
            print(
                "Device memory: "
                + str(torch.cuda.get_device_properties(0).total_memory)
            )

    def load_inputs(self):
        """
        Read from /input/
        Check https://grand-challenge.org/algorithms/interfaces/
        """
        ct_mha = os.listdir(os.path.join(self.input_path, "images/ct/"))
        pet_mha = os.listdir(os.path.join(self.input_path, "images/pet/"))

        preprocessing_configs = {
            'scale_ct': {'mode': "normalization", 'min': -1500, 'max': 3500},
            'scale_pett': {'mode': "normalization", 'min': 0, 'max': 120},
            'resample_image': {'target_shape': (200, 200)}
        }
        settings = Settings()

        data_preprocessor = DataPreprocessor(paths, settings)
        data_preprocessor.preprocess(preprocessing_configs)

        uuid = os.path.splitext(ct_mha)[0]

        self.convert_mha_to_nii(
            os.path.join(self.input_path, "images/ct/", ct_mha),
            os.path.join(self.nii_path, "TCIA_001_0000.nii.gz"),
        )
        self.convert_mha_to_nii(
            os.path.join(self.input_path, "images/pet/", pet_mha),
            os.path.join(self.nii_path, "TCIA_001_0001.nii.gz"),
        )
        return uuid

    def write_outputs(self, uuid):
        """
        Write to /output/
        Check https://grand-challenge.org/algorithms/interfaces/
        """
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        self.convert_nii_to_mha(
            os.path.join(self.result_path, self.nii_seg_file),
            os.path.join(self.output_path, uuid + ".mha"),
        )
        print("Output written to: " + os.path.join(self.output_path, uuid + ".mha"))

    def predict(self):
        """
        Your algorithm goes here
        """

        print("nnUNet segmentation starting!")
        cproc = subprocess.run(
            f"nnUNetv2_predict -i {self.nii_path} -o {self.result_path} -d 001 -c 3d_fullres -f all --disable_tta",
            shell=True,
            check=True,
        )
        print(cproc)
        # since nnUNet_predict call is split into prediction and postprocess, a pre-mature exit code is received but
        # segmentation file not yet written. This hack ensures that all spawned subprocesses are finished before being
        # printed.
        print("Prediction finished")

    def save_datacentric(self, value: bool):
        print("Saving datacentric json to " + self.output_path_category)
        with open(self.output_path_category, "w") as json_file:
            json.dump(value, json_file, indent=4)

    def process(self):
        """
        Read inputs from /input, process with your algorithm and write to /output
        """
        # process function will be called once for each test sample
        self.check_gpu()
        print("Start processing")
        uuid = self.load_inputs()
        print("Start prediction")
        self.predict()
        print("Start output writing")
        self.save_datacentric(False)
        self.write_outputs(uuid)


if __name__ == "__main__":
    print("START")
    Autopet_mymodel().process()
