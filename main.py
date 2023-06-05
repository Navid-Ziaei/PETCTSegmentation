import matplotlib.pyplot as plt

from import_data import *
import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator
from keras.losses import binary_crossentropy
from keras.optimizers import Adam

import numpy as np
from model import *
from metrics import *

# get the path directory
device = 'Raha'

if device == 'Navid':
    path_ct  = "F:/Datasets/PET_CT/original images/test/CT/"
    path_pet = "F:/Datasets/PET_CT/original images/test/PET/"
    path_seg = "F:/Datasets/PET_CT/original images/test/SEG/"
elif device == 'Raha':
    path_ct  = "C:/Users/SadrSystem/Desktop/dissertation_Raha/dataset/new/testt/CT/"
    path_pet = "C:/Users/SadrSystem/Desktop/dissertation_Raha/dataset/new/testt/PET/"
    path_seg = "C:/Users/SadrSystem/Desktop/dissertation_Raha/dataset/new/testt/SEG/"
    
# create generator
datagen = ImageDataGenerator()

# prepare an iterators for each dataset
train_ct  = datagen.flow_from_directory(path_ct   , class_mode = 'binary'  , batch_size = 64)
train_pet = datagen.flow_from_directory(path_pet  , class_mode = 'binary'  , batch_size = 64)
train_seg = datagen.flow_from_directory(path_seg  , class_mode = 'binary'  , batch_size = 64)

##data_ct = import_data(path_ct)
##data_pet = import_data(path_pet)
##data_seg = import_data(path_seg)

# 1- Load all data using flow from directory
# 2- Train model
# 3- Add augmentation
# 4- Visializtion
# 5- EDA
# 6- Add handcrafted features

# 7- Implement state of the art model

##data_ct  = data_ct[:, :, :, np.newaxis]
##data_pet = data_pet[:, :, :, np.newaxis]

model = create_model(data_ct)
model.compile(optimizer=Adam(lr=1e-3), loss=binary_crossentropy, metrics=[dice_coef])

hist = model.fit(x=(train_ct, train_pet),
                 y=train_seg,
                 batch_size=1,
                 epochs=3,
                 validation_split=0.2)

fig, axes = plt.Subplot(1)
axes[0].plot(hist.history['accuracy'])

result = model.evaluate(x_test, y_test)
# save result


