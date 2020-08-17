import os
import random
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import Model
from tensorflow.keras import callbacks
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import activations
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import InceptionV3

IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 3


# Plot results
def plot_visualize_training(history):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    ax1.plot(history.history['loss'], color='b', label="Training loss")
    ax1.plot(history.history['val_loss'], color='r', label="validation loss")
    ax1.set_xticks(np.arange(1, len(history.history['loss']), 1))
    ax1.set_yticks(np.arange(0, 1, 0.1))

    ax2.plot(history.history['accuracy'], color='b', label="Training accuracy")
    ax2.plot(history.history['val_accuracy'], color='r', label="Validation accuracy")
    ax2.set_xticks(np.arange(1, len(history.history['loss']), 1))

    plt.legend(loc='best', shadow=True)
    plt.tight_layout()
    plt.show()


def plot_sample_images(image_generator, categories=None):
    plt.figure(figsize=(12, 24))

    filenames = np.array(image_generator.filenames)

    if categories is not None:
        filenames = np.column_stack((filenames, np.array(categories)))

    samples = random.choices(filenames, k=18)

    for index, sample in enumerate(samples, start=0):
        if categories is not None:
            filename = sample[0]
            # plt.xlabel(round(float(sample[1])))
        else:
            filename = sample

        image_path = os.path.join(image_generator.directory, filename)
        sample = image.load_img(image_path, target_size=IMAGE_SIZE)

        plt.subplot(6, 3, index + 1)
        plt.imshow(sample)

    plt.tight_layout()
    plt.show()


# Callbacks
class CustomCallback(callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('accuracy') > 0.90:
            print("\nReached 99.5% accuracy so cancelling training!")
            self.model.stop_training = True


def main():
    # Get pre-trained model
    trained_model = InceptionV3(
        input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS),
        include_top=False
    )

    for layer in trained_model.layers:
        layer.trainable = False

    trained_model.summary()
    print('last layer output shape: ', trained_model.get_layer('mixed7').output_shape)

    # Create model using pre-trained model
    x = layers.Flatten()(trained_model.get_layer('mixed7').output)
    x = layers.Dense(1024, activation=activations.relu)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(1, activation=activations.sigmoid)(x)

    model = Model(
        trained_model.input,
        x
    )

    model.compile(
        optimizer=optimizers.RMSprop(learning_rate=0.001),
        loss=losses.BinaryCrossentropy(),
        metrics=['accuracy']
    )

    # Augmented training data image flow
    training_image_data_generation = image.ImageDataGenerator(
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=40,
        horizontal_flip=True,
        rescale=1 / 255.
    )

    training_image_data = training_image_data_generation.flow_from_directory(
        directory=os.path.join(os.getcwd(), 'datasets', 'images', 'horses_or_humans', 'train'),
        target_size=IMAGE_SIZE,
        class_mode='binary'
    )

    plot_sample_images(training_image_data)

    # Validation data should not be augmented
    validation_image_data_generation = image.ImageDataGenerator(
        rescale=1 / 255.
    )
    validation_image_data = validation_image_data_generation.flow_from_directory(
        directory=os.path.join(os.getcwd(), 'datasets', 'images', 'horses_or_humans', 'validation'),
        target_size=IMAGE_SIZE,
        class_mode='binary'
    )

    # Training model with new data
    model_callbacks = [
        CustomCallback()
    ]

    history = model.fit(
        training_image_data,
        epochs=50,
        validation_data=validation_image_data,
        callbacks=model_callbacks,
        verbose=2
    )

    # Predictions for images
    categories = model.predict(validation_image_data)
    # plot_sample_images(validation_image_data, categories)

    # Plot results and check for over-fitting
    plot_visualize_training(history)


if __name__ == '__main__':
    main()
