import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy

layers = tf.keras.layers
models = tf.keras.models


def create_model(input_shape):
    in1 = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, activation='relu', padding="same")(in1)
    x11 = layers.Conv2D(32, 3, activation='relu', padding="same")(x)
    x12 = layers.MaxPooling2D()(x11)

    x1 = layers.Conv2D(64, 3, activation='relu', padding="same")(x12)
    x21 = layers.Conv2D(32, 3, activation='relu', padding="same")(x1)
    x22 = layers.MaxPooling2D()(x21)

    x2 = layers.Conv2D(128, 3, activation='relu', padding="same")(x22)
    x31 = layers.Conv2D(128, 3, activation='relu', padding="same")(x2)
    x32 = layers.MaxPooling2D()(x31)

    x3 = layers.Conv2D(256, 3, activation='relu', padding="same")(x32)
    x31 = layers.Conv2D(256, 3, activation='relu', padding="same")(x3)
    x32 = layers.MaxPooling2D()(x31)
    
    # Second branch 
    in2 = layers.Input(shape=input_shape)
    y = layers.Conv2D(32, 3, activation='relu', padding="same")(in2)
    y11 = layers.Conv2D(32, 3, activation='relu', padding="same")(y)
    y12 = layers.MaxPooling2D()(y11)

    y1 = layers.Conv2D(64, 3, activation='relu', padding="same")(y12)
    y21 = layers.Conv2D(32, 3, activation='relu', padding="same")(y1)
    y22 = layers.MaxPooling2D()(y21)

    y2 = layers.Conv2D(128, 3, activation='relu', padding="same")(y22)
    y31 = layers.Conv2D(128, 3, activation='relu', padding="same")(y2)
    y32 = layers.MaxPooling2D()(y31)

    y3 = layers.Conv2D(256, 3, activation='relu', padding="same")(y32)
    y31 = layers.Conv2D(256, 3, activation='relu', padding="same")(y3)
    y32 = layers.MaxPooling2D()(y31)
    
    # Concat

    out = layers.Concatenate()([x32, y32])
    
    # Decoder

    z = layers.UpSampling2D(size=(2, 2))(out)
    z11 = layers.Conv2D(256, 3, activation='relu', padding="same")(z)
    z12 = layers.Conv2D(256, 3, activation='relu', padding="same")(z11)

    z1 = layers.UpSampling2D(size=(2, 2))(z12)
    z21 = layers.Conv2D(128, 3, activation='relu', padding="same")(z1)
    z22 = layers.Conv2D(128, 3, activation='relu', padding="same")(z21)

    z2 = layers.UpSampling2D(size=(2, 2))(z22)
    z31 = layers.Conv2D(64, 3, activation='relu', padding="same")(z2)
    z32 = layers.Conv2D(64, 3, activation='relu', padding="same")(z31)

    z3 = layers.UpSampling2D(size=(2, 2))(z32)
    z41 = layers.Conv2D(32, 3, activation='relu', padding="same")(z3)
    z42 = layers.Conv2D(2, 3, activation='relu', padding="same")(z41)
    z43 = layers.Conv2D(1, 1, activation='softmax')(z42)

    model2 = models.Model(inputs=[in1, in2], outputs=z43)
    
    return model2


def simple_model(input_shape):
    inputs1 = Input(input_shape)
    inputs2 = Input(input_shape)

    # Encoder 1
    conv11 = Conv2D(32, 3, activation='relu', padding='same')(inputs1)
    conv11 = Conv2D(32, 3, activation='relu', padding='same')(conv11)
    pool11 = MaxPooling2D(pool_size=(2, 2))(conv11)



    # Encoder 1
    conv12 = Conv2D(32, 3, activation='relu', padding='same')(inputs2)
    conv12 = Conv2D(32, 3, activation='relu', padding='same')(conv12)
    pool12 = MaxPooling2D(pool_size=(2, 2))(conv12)

    out = layers.Concatenate()([pool11, pool12])

    # Decoder
    conv2 = Conv2D(64, 3, activation='relu', padding='same')(out)
    conv2 = Conv2D(64, 3, activation='relu', padding='same')(conv2)
    up1 = UpSampling2D(size=(2, 2))(conv2)

    # Output
    output = Conv2D(1, 1, activation='sigmoid')(up1)

    model = Model(inputs=[inputs1, inputs2], outputs=output)
    return model