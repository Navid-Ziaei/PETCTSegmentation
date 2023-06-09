import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from import_data import *
from keras.preprocessing.image import ImageDataGenerator
from model import *
from metrics import *
from pathlib import Path
import datetime
settings = {
    "batch_size": 4,
    "epochs": 3,
    "seed": 42
}
# get the path directory
device = 'Navid-PC'

if device == 'Navid':
    dataset_path_train = "F:/Datasets/PET_CT/original images/train/"
    dataset_path_test = "F:/Datasets/PET_CT/original images/train/"
elif device == 'Navid-PC':
    dataset_path_train = "D:/Navid/Dataset/Head-Neck-PET-CT/Dataset/train/"
    dataset_path_test = "D:/Navid/Dataset/Head-Neck-PET-CT/Dataset/test/"

elif device == 'Raha':
    dataset_path_train = "C:/Users/SadrSystem/Desktop/dissertation_Raha/dataset/new/trainn/"
    dataset_path_test = "C:/Users/SadrSystem/Desktop/dissertation_Raha/dataset/new/test/"


dataset_path_train = "D:/Navid/Dataset/Head-Neck-PET-CT/Dataset/train/"

""" ================ Create Save Paths====================="""
# Create Results Path
dir_path = os.path.dirname(os.path.realpath(__file__))  # get the working directory path
results_path = dir_path + '/results/' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') +'/'
model_path = results_path + '/model/'
model_output = results_path + '/predictions/'
Path(results_path).mkdir(parents=True, exist_ok=True)
Path(model_path).mkdir(parents=True, exist_ok=True)
Path(model_output).mkdir(parents=True, exist_ok=True)

""" ================ Load Data====================="""

path_ct_train = dataset_path_train + "/CT/"
path_pet_train = dataset_path_train + "/PET/"
path_seg_train = dataset_path_train + "/SEG/"

path_ct_test = dataset_path_test + "/CT/"
path_pet_test = dataset_path_test + "/PET/"
path_seg_test = dataset_path_test + "/SEG/"

# create generator
datagen = ImageDataGenerator()

train_generator, steps_per_epoch_training = image_data_gen(path_pet=path_pet_train,
                                                           path_ct=path_ct_train,
                                                           path_seg=path_seg_train,
                                                           settings=settings,
                                                           target_size=(144, 144))

test_generator, steps_per_epoch_test = image_data_gen(path_pet=path_pet_test,
                                                      path_ct=path_ct_test,
                                                      path_seg=path_seg_test,
                                                      settings=settings,
                                                      target_size=(144, 144))

""" ================ Visualize Data====================="""
# Test the generator by retrieving one batch
data_train = train_generator()
data_test = test_generator()

inputs, targets = next(data_train)

idx = 3
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(inputs[0][idx], cmap='gray')
axes[0].set_title("PET")

axes[1].imshow(inputs[1][idx], cmap='gray')
axes[1].set_title("CT")

axes[2].imshow(targets[idx], cmap='gray')
axes[2].set_title("Mask")
plt.tight_layout()
plt.show()

# 1- Load all data using flow from directory
# 2- Train model
# 3- Add augmentation
# 4- Visializtion
# 5- EDA
# 6- Add handcrafted features

# 7- Implement state of the art model

""" ================ Create Model====================="""
model = simple_model(input_shape=(144, 144, 1))
model.compile(optimizer=Adam(learning_rate=1e-3), loss=dice_loss, metrics=[dice_coefficient])
model.save(model_path + 'model.h5')

""" ================ Train Model====================="""
hist = model.fit(data_train,
                 batch_size=settings["batch_size"],
                 epochs=settings["epochs"],
                 steps_per_epoch=steps_per_epoch_training,
                 validation_data=data_test,
                 validation_batch_size=settings["batch_size"],
                 validation_steps=steps_per_epoch_test)

fig, axes = plt.subplots(2, 1)
axes[0].plot(hist.history['loss'])
axes[0].plot(hist.history['val_loss'])
axes[0].set_title("Loss")
axes[1].plot(hist.history['dice_coefficient'])
axes[1].plot(hist.history['val_dice_coefficient'])
axes[1].set_title("DIce")
plt.tight_layout()
fig.savefig(results_path + "Training_curves.png")
plt.show()

""" ================ Evaluate Model====================="""
result = model.evaluate(data_test, batch_size=settings['batch_size'], steps=steps_per_epoch_test)
print(result)
with open(results_path+"results.txt","w") as f:
    f.write("Results (loss, DICE): {}".format(result))

""" ================ Save All model outputs====================="""
settings['batch_size'] = 1
test_generator, steps_per_epoch_test = image_data_gen(path_pet=path_pet_test,
                                                      path_ct=path_ct_test,
                                                      path_seg=path_seg_test,
                                                      settings=settings,
                                                      target_size=(144, 144))
data_test = test_generator()
idx = 0
for data, gt in tqdm(data_test):
    idx = idx + 1
    prediction = model.predict(data, verbose=0)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(np.squeeze(gt), cmap='gray')
    axes[0].set_title("Ground Truth")

    axes[1].imshow(np.squeeze(prediction), cmap='gray')
    axes[1].set_title("Prediction")

    plt.tight_layout()
    fig.savefig(model_output + 'pred{}.png'.format(idx))
# save result
