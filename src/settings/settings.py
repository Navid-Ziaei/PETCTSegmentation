from pathlib import Path

import yaml
import os
import warnings


class Settings:
    def __init__(self):

        self.__supported_dataset = ['pet_ct']
        self.__supported_models = ['unet']
        self.__model = None
        self.__debug_mode = False
        self.__load_pretrained_model = False
        self.__test_size = 0.2
        self.__file_format = '.pkl'
        self.__dataset = self.__supported_dataset[0]
        self.__model = None
        self.load_trained_model = False
        self.batch_size = 2
        self.num_epochs = 10
        self.validation_size = 0.1
        self.learning_rate = 0.001


    def load_settings(self):
        """
        This function loads the YAML files for settings and network settings from the working directory and
        creates a Settings object based on the fields in the YAML file. It also loads the local path of the datasets
        from device_path.yaml
        return:
            settings: a Settings object
            network_settings: a dictionary containing settings of the saved_model
            device_path: the path to the datasets on the local device
        """

        """ working directory """
        working_folder = Path(__file__).resolve().parents[2]
        config_folder = working_folder / 'configs'

        """ loading settings from the yaml file """
        try:
            with open(config_folder / "settings.yaml", "r") as file:
                settings_yaml = yaml.safe_load(file)
        except Exception as e:
            raise Exception('Could not load settings.yaml from the working directory!') from e

        for key, value in settings_yaml.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise Exception('{} is not an attribute of the Settings class!'.format(key))

    @property
    def dataset(self):
        return self.__dataset

    @dataset.setter
    def dataset(self, dataset_name):
        if isinstance(dataset_name, str) and dataset_name in self.__supported_dataset:
            print(f"\n Selected Dataset is : {dataset_name}")
            self.__dataset = dataset_name
        else:
            raise ValueError(f"dataset should be selected from supported datasets: {self.__supported_dataset}")

    @property
    def model(self):
        return self.__model

    @model.setter
    def model(self, model_name):
        if isinstance(model_name, str) and model_name in self.__supported_models:
            print(f"\n Selected Dataset is : {model_name}")
            self.__model = model_name
        else:
            raise ValueError(f"dataset should be selected from supported datasets: {self.__supported_models}")

    @property
    def test_size(self):
        return self.__test_size

    @test_size.setter
    def test_size(self, value):
        if 0 < value < 1:
            self.__test_size = value
        else:
            raise ValueError("test_size should be float number between 0 to 1")

    @property
    def load_pretrained_model(self):
        return self.__load_pretrained_model

    @load_pretrained_model.setter
    def load_pretrained_model(self, value):
        if isinstance(value, bool):
            self.__load_pretrained_model = value
        else:
            raise ValueError("load_pretrained_model should be True or False")

    @property
    def debug_mode(self):
        return self.__debug_mode

    @debug_mode.setter
    def debug_mode(self, value):
        if isinstance(value, bool):
            self.__debug_mode = value
        else:
            raise ValueError("The v should be boolean")



