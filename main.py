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
device = 'Navid-PC'

if device == 'Navid':
    dataset_path = "F:/Datasets/PET_CT/original images/train/"
elif device == 'Navid-PC':
    dataset_path = "D:/Navid/Dataset/Head-Neck-PET-CT/Dataset/train/"
elif device == 'Raha':
    dataset_path = "C:/Users/SadrSystem/Desktop/dissertation_Raha/dataset/new/trainn/"

dataset_path = "D:/Navid/Dataset/Head-Neck-PET-CT/Dataset/train/"

path_ct = dataset_path + "/CT/"
path_pet = dataset_path + "/PET/"
path_seg = dataset_path + "/SEG/"

# create generator
datagen = ImageDataGenerator()

pet_generator = ImageDataGenerator()
ct_generator = ImageDataGenerator()
mask_generator = ImageDataGenerator()

datagen = ImageDataGenerator()

pet_generator = datagen.flow_from_directory(path_pet,
                                            class_mode=None,
                                            color_mode='grayscale',
                                            target_size=(144, 144),
                                            seed=42,
                                            shuffle=False,
                                            batch_size=4)

ct_generator = datagen.flow_from_directory(path_ct,
                                           class_mode=None,
                                           color_mode='grayscale',
                                           target_size=(144, 144),
                                           seed=42,
                                           shuffle=False,
                                           batch_size=4)

mask_generator = datagen.flow_from_directory(path_seg,
                                             class_mode=None,
                                             target_size=(144, 144),
                                             seed=42,
                                             shuffle=False,
                                             batch_size=4)


def dataset_generator():
    while True:
        pet_img = pet_generator.next()
        ct_img = ct_generator.next()
        mask = mask_generator.next()
        yield [pet_img, ct_img], mask

# Test the generator by retrieving one batch
generator = dataset_generator()
inputs, targets = next(generator)

idx=3
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(inputs[0][idx], cmap='gray')
axes[0].set_title("PET")

axes[1].imshow(inputs[1][idx], cmap='gray')
axes[1].set_title("CT")

axes[2].imshow(targets[idx], cmap='gray')
axes[2].set_title("Mask")
plt.tight_layout()
plt.show()

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


model = create_model(input_shape=(144, 144, 1))
model.compile(optimizer=Adam(lr=1e-3), loss=binary_crossentropy, metrics=[dice_coef])

hist = model.fit(data_train,
                 batch_size=1,
                 epochs=3)

fig, axes = plt.Subplot(1)
axes[0].plot(hist.history['accuracy'])

result = model.evaluate(x_test, y_test)
# save result
