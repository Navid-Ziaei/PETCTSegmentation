from import_data import *
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D
from tensorflow.keras.layers import Flatten, Input, Concatenate, UpSampling2D
import numpy as np
from tensorflow.keras.utils import plot_model
#get the path directory

path_ct  = "C:/Users/SadrSystem/Desktop/dissertation_Raha/dataset/new/testt/CT/"
path_pet = "C:/Users/SadrSystem/Desktop/dissertation_Raha/dataset/new/testt/PET/"
path_seg = "C:/Users/SadrSystem/Desktop/dissertation_Raha/dataset/new/testt/SEG/"
data_ct  = import_data(path_ct)
data_pet = import_data(path_pet)
data_seg = import_data(path_seg)

#fig = plt.figure(figsize=(144,144))
#fig.add_subplot(1, 3, matrix_3d[1])
#plt.show
#data_ct = data_ct[:,:,:,np.newaxis]

#model = Sequential([
        #Conv2D(32, 3, input_shape=data_ct.shape[1:]),
        #MaxPooling2D(2),
        #Flatten(),
        #Dense(2) 
    #])

#in1 = Input(shape=data_ct.shape[1:])
#x   = Conv2D(32, 3, activation='relu', padding="same")(in1)
#x2  = MaxPooling2D()(x)
#x3  = Flatten()(x2)

#in2 = Input(shape=data_ct.shape[1:])
#x11 = Conv2D(32, 3, activation='relu', padding="same")(in2)
#x11_cat = Concatenate()([in2, x11])
#x12 = MaxPooling2D()(x11_cat)
#x13 = Flatten()(x12)

#x4 = Concatenate()([x13, x3])
#out = Dense(2, activation='softmax')(x4)


#model2 = Model(inputs = [in1, in2], outputs=out)
#plot_model(model2, show_shapes=True)

data_ct  = data_ct [:,:,:,np.newaxis]
data_pet = data_pet[:,:,:,np.newaxis]

in1 = Input(shape=data_ct.shape[1:])
x   = Conv2D(32, 3, activation='relu', padding="same")(in1)
x11 = Conv2D(32, 3, activation='relu', padding="same")(x)
x12 = MaxPooling2D()(x11)

x1  = Conv2D(64, 3, activation='relu', padding="same")(x12)
x21 = Conv2D(32, 3, activation='relu', padding="same")(x1)
x22 = MaxPooling2D()(x21)

x2  = Conv2D(128, 3, activation='relu', padding="same")(x22)
x31 = Conv2D(128, 3, activation='relu', padding="same")(x2)
x32 = MaxPooling2D()(x31)

x3  = Conv2D(256, 3, activation='relu', padding="same")(x32)
x31 = Conv2D(256, 3, activation='relu', padding="same")(x3)
x32 = MaxPooling2D()(x31)



in2 = Input(shape=data_pet.shape[1:])
y   = Conv2D(32, 3, activation='relu', padding="same")(in2)
y11 = Conv2D(32, 3, activation='relu', padding="same")(y)
y12 = MaxPooling2D()(y11)

y1  = Conv2D(64, 3, activation='relu', padding="same")(y12)
y21 = Conv2D(32, 3, activation='relu', padding="same")(y1)
y22 = MaxPooling2D()(y21)

y2  = Conv2D(128, 3, activation='relu', padding="same")(y22)
y31 = Conv2D(128, 3, activation='relu', padding="same")(y2)
y32 = MaxPooling2D()(y31)

y3  = Conv2D(256, 3, activation='relu', padding="same")(y32)
y31 = Conv2D(256, 3, activation='relu', padding="same")(y3)
y32 = MaxPooling2D()(y31)


out   = Concatenate()([x32,y32])


z   = UpSampling2D(size = (2,2))(out)
z11 = Conv2D(256, 3, activation='relu', padding="same")(z)
z12 = Conv2D(256, 3, activation='relu', padding="same")(z11)

z1  = UpSampling2D(size = (2,2))(z12)
z21 = Conv2D(128, 3, activation='relu', padding="same")(z1)
z22 = Conv2D(128, 3, activation='relu', padding="same")(z21)

z2  = UpSampling2D(size = (2,2))(z22)
z31 = Conv2D(64, 3, activation='relu', padding="same")(z2)
z32 = Conv2D(64, 3, activation='relu', padding="same")(z31)

z3  = UpSampling2D(size = (2,2))(z32)
z41 = Conv2D(32, 3, activation='relu', padding="same")(z3)
z42 = Conv2D(2 , 3, activation='relu', padding="same")(z41)
z43 = Conv2D(1 , 1, activation = 'softmax')(z42)


model2 = Model(inputs = [in1,in2] , outputs = z43)

#oup_put = model2((data_ct[1:3],data_pet[1:3]))

model2.compile(optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy())


hist = model2.fit(x=(data_ct, data_pet),
                    y=data_seg[:,:,:,np.newaxis],
                    batch_size=1,
                    epochs=3,
                    validation_split = 0.2)



