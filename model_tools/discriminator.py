import numpy as np

from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, MaxPool2D, Concatenate, Add
from tensorflow.keras.models import Model, load_model


def build_discriminator(embedding_length):

    # inputs
    left = Input((embedding_length,))
    right = Input((embedding_length,))

    combined = Concatenate(axis=1)([left, right])

    # run through a fully connected network
    a = Dense(128, activation="tanh")(combined)
    b = Dense(128, activation="tanh")(a)
    c = Dense(128, activation="tanh")(b)

    output = Dense(1, activation="sigmoid")(c)

    model = Model(inputs=[left, right], outputs=[output])
    return model
