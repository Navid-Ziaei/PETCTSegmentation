import tensorflow as tf

layers = tf.keras.layers
models = tf.keras.models


def create_model(data_ct):
    in1 = layers.Input(shape=data_ct.shape[1:])
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

    in2 = layers.Input(shape=data_pet.shape[1:])
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

    out = layers.Concatenate()([x32, y32])

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
