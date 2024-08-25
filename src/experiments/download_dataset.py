
import os

import pandas as pd
import torch
from tqdm import tqdm

from src.data import download_fdg_by_id, download_psma_by_id
from src.settings import Settings, Paths

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device is {device}")

settings = Settings()
settings.load_settings()

paths = Paths(settings=settings)
paths.load_device_paths()


data_info_fdg = pd.read_csv(paths.raw_dataset_path + 'fdg_metadata.csv')
data_info_psma = pd.read_csv(paths.raw_dataset_path + 'psma_metadata.csv')

#file_names = [sub_id+'_'+sub_date for sub_id, sub_date in zip(data_info_psma['Subject ID'],data_info_psma['Study Date'])]
file_names = [sub_id.split('_')[1] for sub_id in data_info_psma['Subject ID']]

for file_name in tqdm(file_names):
    try:
        download_psma_by_id(file_name, data_info_psma, paths.raw_dataset_path)
    except:
        print(f"File {file_name} not present in FDG")

