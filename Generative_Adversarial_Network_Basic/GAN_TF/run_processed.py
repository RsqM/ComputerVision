"""
This module implements a basic GAN training loop using TensorFlow.
It includes generator and discriminator models, loss functions,
and training steps.
"""

import tensorflow as tf
import numpy as np

from DiscriminatorImport import DiscriminatorLoss, DiscriminatorModel
from GeneratorImport import GeneratorLoss, GeneratorModel

GENERATOR_OPTIMIZER = tf.keras.optimizers.Adam(1e-4)
DISCRIMINATOR_OPTIMIZER = tf.keras.optimizers.Adam(1e-3)

generator = GeneratorModel()
discriminator = DiscriminatorModel()

BATCH_SIZE = 100


def training_step(images):
    """
    Performs a single training step for the GAN.

    Args:
        images: A batch of real images.
    """
    noise = np.random.normal(0, 1, (BATCH_SIZE, 100)).astype(np.float32)
    with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:
        generated_images = generator(noise)
        real_output = discriminator(images)
        fake_output = discriminator(generated_images)

        generator_loss = GeneratorLoss(fake_output)
        discriminator_loss = DiscriminatorLoss(real_output, fake_output)

        gradients_generator = generator_tape.gradient(
            generator_loss, generator.trainable_variables
        )
        gradients_discriminator = discriminator_tape.gradient(
            discriminator_loss, discriminator.trainable_variables
        )

        GENERATOR_OPTIMIZER.apply_gradients(
            zip(gradients_generator, generator.trainable_variables)
        )
        DISCRIMINATOR_OPTIMIZER.apply_gradients(
            zip(gradients_discriminator, discriminator.trainable_variables)
        )

        print("Generator Loss:", np.mean(generator_loss), "\n")
        print("Discriminator Loss:", np.mean(discriminator_loss), "\n")


def train(dataset: tf.data.Dataset, epochs: int):
    """
    Trains the GAN for a specified number of epochs.

    Args:
        dataset: The training dataset.
        epochs: The number of training epochs.
    """
    for _ in range(epochs):
        for images in dataset:
            images = tf.cast(images, tf.float32)
            training_step(images)


(train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()

train_x = train_x.reshape(train_x.shape[0], 28, 28, 1)
train_x = (train_x - 127.5) / 127.5

BUFFER = train_x.shape[0]
train_x = tf.data.Dataset.from_tensor_slices(train_x).shuffle(BUFFER).batch(100)

train(train_x, 1)