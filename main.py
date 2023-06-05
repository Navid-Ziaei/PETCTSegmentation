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
    path_ct  = "C:/Users/SadrSystem/Desktop/dissertation_Raha/dataset/new/trainn/CT/"
    path_pet = "C:/Users/SadrSystem/Desktop/dissertation_Raha/dataset/new/trainn/PET/"
    path_seg = "C:/Users/SadrSystem/Desktop/dissertation_Raha/dataset/new/trainn/SEG/"
    
# create generator
datagen = ImageDataGenerator()

pet_generator = ImageDataGenerator()
ct_generator = ImageDataGenerator()
mask_generator = ImageDataGenerator()

def dataset_generator():

    # prepare an iterators for each dataset
    train_ct  = ct_generator.flow_from_directory(path_ct, 
                                                 class_mode = None ,
                                                 target_size= (144,144),
                                                 seed = 42,
                                                 shuffle=False,
                                                 batch_size=4)
    train_pet = pet_generator.flow_from_directory(path_pet  ,   
                                                class_mode = None ,
                                                target_size= (144,144),
                                                seed = 42,
                                                shuffle=False,
                                                batch_size=4
                                                  )
    train_seg = mask_generator.flow_from_directory(path_seg  ,
                                                    class_mode = None ,
                                                    target_size= (144,144),
                                                    seed = 42,
                                                    shuffle=False,
                                                    batch_size=4)
    while True:
        pet_img = train_pet.next()
        ct_img = train_ct.next()
        mask = train_seg.next()
        yield [pet_img, ct_img], mask

data_train = dataset_generator()

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

model = create_model(input_shape=(144,144,1))
model.compile(optimizer=Adam(lr=1e-3), loss=binary_crossentropy, metrics=[dice_coef])

hist = model.fit(data_train,
                 batch_size=1,
                 epochs=3)

fig, axes = plt.Subplot(1)
axes[0].plot(hist.history['accuracy'])

result = model.evaluate(x_test, y_test)
# save result


