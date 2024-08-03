import yaml
import datetime
from pathlib import Path
import os

class Paths:
    def __init__(self, settings):
        self.result_path = None
        self.model_path = None
        self.raw_dataset_path = None
        self.preprocessed_dataset_path = None
        self.base_path = None
        self.debug_mode = settings.debug_mode

    def load_device_paths(self):
        """ working directory """

        working_folder = Path(__file__).resolve().parents[2]
        config_folder = working_folder / 'configs'


        """ loading device path from the yaml file """
        try:
            with open(config_folder / "device_path.yaml", "r") as file:
                device = yaml.safe_load(file)
        except Exception as e:
            raise Exception('Could not load device_path.yaml from the working directory!') from e

        for key, value in device.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise Exception('{} is not an attribute of the Paths class!'.format(key))

        self.create_paths()

    def create_paths(self):
        dir_path = Path(__file__).resolve().parents[2]

        self.base_path = dir_path / 'results'
        results_base_path = f'{self.base_path}/{self.result_path}/'

        if self.debug_mode is False:
            self.folder_name = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        else:
            self.folder_name = 'debug'
        results_base_path = results_base_path + self.folder_name + '/'

        model_path = os.path.join(results_base_path + 'saved_model/')
        Path(results_base_path).mkdir(parents=True, exist_ok=True)
        Path(model_path).mkdir(parents=True, exist_ok=True)
        self.model_path = model_path
        self.result_path = results_base_path
