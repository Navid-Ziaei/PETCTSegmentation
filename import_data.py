
import os
from os import listdir
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from tqdm import tqdm

def import_data(folder_dir):


    file_list = os.listdir(folder_dir)
    num_samples = len(file_list)
    print(num_samples)

    matrix_3d = np.zeros((num_samples,144,144))
    print(matrix_3d.shape)
    for idx, im in enumerate(tqdm(file_list)):
        #print(im)
        #print(idx)
        img = Image.open(folder_dir + im)
        img_arr = np.array(img)
        matrix_3d[idx,:,:] = img_arr
    return matrix_3d 