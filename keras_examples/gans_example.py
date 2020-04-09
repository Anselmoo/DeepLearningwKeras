import keras

print(keras.__version__)

import keras
from keras import layers
import numpy as np
from keras.preprocessing import image

# Global variables
latent_dim = 32
height = 32
width = 32
channels = 3


def generator_input():
    generator_input = keras.Input(shape=(latent_dim,))

    # First, transform the input into a 16x16 128-channels feature map
    x = layers.Dense(128 * 16 * 16)(generator_input)
    x = layers.LeakyReLU()(x)
    x = layers.Reshape((16, 16, 128))(x)

    # Then, add a convolution layer
    x = layers.Conv2D(256, 5, padding="same")(x)
    x = layers.LeakyReLU()(x)

    # Upsample to 32x32
    x = layers.Conv2DTranspose(256, 4, strides=2, padding="same")(x)
    x = layers.LeakyReLU()(x)

    # Few more conv layers
    x = layers.Conv2D(256, 5, padding="same")(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(256, 5, padding="same")(x)
    x = layers.LeakyReLU()(x)

    # Produce a 32x32 1-channel feature map
    x = layers.Conv2D(channels, 7, activation="tanh", padding="same")(x)
    generator = keras.models.Model(generator_input, x)
    generator.summary()
    return generator, x


def discriminator_input(x):
    discriminator_input = layers.Input(shape=(height, width, channels))
    layers.Conv2D(128, 3)(discriminator_input)
    layers.LeakyReLU()(x)
    layers.Conv2D(128, 4, strides=2)(x)
    layers.LeakyReLU()(x)
    layers.Conv2D(128, 4, strides=2)(x)
    layers.LeakyReLU()(x)
    layers.Conv2D(128, 4, strides=2)(x)
    layers.LeakyReLU()(x)
    layers.Flatten()(x)

    # One dropout layer - important trick!
    layers.Dropout(0.4)(x)

    # Classification layer
    layers.Dense(1, activation="sigmoid")(x)

    discriminator = keras.models.Model(discriminator_input, x)
    discriminator.summary()

    # To stabilize training, we use learning rate decay
    # and gradient clipping (by value) in the optimizer.
    discriminator_optimizer = keras.optimizers.RMSprop(
        lr=0.0008, clipvalue=1.0, decay=1e-8
    )
    discriminator.compile(optimizer=discriminator_optimizer, loss="binary_crossentropy")
    return discriminator


def gans_init(discriminator, generator):
    # Set discriminator weights to non-trainable
    # (will only apply to the `gan` model)
    discriminator.trainable = False

    gan_input = keras.Input(shape=(latent_dim,))
    gan_output = discriminator(generator(gan_input))
    gan = keras.models.Model(gan_input, gan_output)

    gan_optimizer = keras.optimizers.RMSprop(lr=0.0004, clipvalue=1.0, decay=1e-8)
    gan.compile(optimizer=gan_optimizer, loss="binary_crossentropy")


if __name__ == "__main__":
    generator, x = generator_input()
    discriminator = discriminator_input(x)
