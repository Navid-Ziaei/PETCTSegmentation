import os
from os import listdir
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from tqdm import tqdm
from keras.preprocessing.image import ImageDataGenerator


def import_data(folder_dir):
    file_list = os.listdir(folder_dir)
    num_samples = len(file_list)
    print(num_samples)

    matrix_3d = np.zeros((num_samples, 144, 144))
    print(matrix_3d.shape)
    for idx, im in enumerate(tqdm(file_list)):
        # print(im)
        # print(idx)
        img = Image.open(folder_dir + im)
        img_arr = np.array(img)
        matrix_3d[idx, :, :] = img_arr
    return matrix_3d


def image_data_gen(path_pet, path_ct, path_seg, settings, target_size=(144, 144)):
    # create generator
    datagen = ImageDataGenerator()

    pet_generator = ImageDataGenerator()
    ct_generator = ImageDataGenerator()
    mask_generator = ImageDataGenerator()

    datagen = ImageDataGenerator()

    pet_generator = datagen.flow_from_directory(path_pet,
                                                class_mode=None,
                                                color_mode='grayscale',
                                                target_size=target_size,
                                                seed=settings["seed"],
                                                shuffle=False,
                                                batch_size=settings["batch_size"])

    ct_generator = datagen.flow_from_directory(path_ct,
                                               class_mode=None,
                                               color_mode='grayscale',
                                               target_size=target_size,
                                               seed=settings["seed"],
                                               shuffle=False,
                                               batch_size=settings["batch_size"])

    mask_generator = datagen.flow_from_directory(path_seg,
                                                 class_mode=None,
                                                 target_size=target_size,
                                                 color_mode='grayscale',
                                                 seed=settings["seed"],
                                                 shuffle=False,
                                                 batch_size=settings["batch_size"])

    def dataset_generator():
        while True:
            pet_img = pet_generator.next()
            ct_img = ct_generator.next()
            mask = mask_generator.next()
            yield [pet_img, ct_img], mask

    return dataset_generator, len(ct_generator)
