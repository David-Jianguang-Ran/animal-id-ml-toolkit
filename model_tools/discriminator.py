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


def assemble_end_to_end_model(encoder_path, discriminator_path, input_shape):
    """
    returns a combined model that takes two batch of images that
        -- returns calculated identity embeddings for each input as output 0, and 1
        -- returns sameness of between left and right at some index as output 2
    """
    left_image = Input(input_shape)
    right_image = Input(input_shape)

    # feed both images through encoder
    encoder = load_model(encoder_path)
    left_embedding = encoder(left_image)
    right_embedding = encoder(right_image)

    # feed both embeddings through discr.
    discriminator = load_model(discriminator_path)
    sameness = discriminator([left_embedding, right_embedding])

    model = Model(inputs=[left_image, right_image], outputs=[left_embedding, right_embedding, sameness])
    return model



