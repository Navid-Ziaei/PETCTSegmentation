from import_data import *
import tensorflow as tf

from keras.losses import binary_crossentropy
from keras.optimizers import Adam

import numpy as np
from model import *
from metrics import *

# get the path directory
device = 'Navid'

if device == 'Navid':
    path_ct = "F:/Datasets/PET_CT/original images/test/CT/"
    path_pet = "F:/Datasets/PET_CT/original images/test/PET/"
    path_seg = "F:/Datasets/PET_CT/original images/test/SEG/"
elif device == 'Raha':
    path_ct = "C:/Users/SadrSystem/Desktop/dissertation_Raha/dataset/new/testt/CT/"
    path_pet = "C:/Users/SadrSystem/Desktop/dissertation_Raha/dataset/new/testt/PET/"
    path_seg = "C:/Users/SadrSystem/Desktop/dissertation_Raha/dataset/new/testt/SEG/"

data_ct = import_data(path_ct)
data_pet = import_data(path_pet)
data_seg = import_data(path_seg)

data_ct = data_ct[:, :, :, np.newaxis]
data_pet = data_pet[:, :, :, np.newaxis]

model = create_model(data_ct)
model.compile(optimizer=Adam(lr=1e-3), loss=binary_crossentropy, metrics=[dice_coef])

hist = model.fit(x=(data_ct, data_pet),
                 y=data_seg[:, :, :, np.newaxis],
                 batch_size=1,
                 epochs=3,
                 validation_split=0.2)
