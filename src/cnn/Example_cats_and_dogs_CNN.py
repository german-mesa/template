import os
import random
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras.preprocessing import image

NUMBER_EPOCHS = 50
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 3


class CustomCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.99:
            print("\nEarly stopping as accuracy is over 99%")
            self.model.stop_training = True


def get_sample_category(sample):
    if 'dog' in sample:
        return 'Dog'
    elif 'cat' in sample:
        return 'Cat'
    else:
        return 'None'


def plot_sample_images(image_generator):
    plt.figure(figsize=(12, 24))

    samples = random.choices(image_generator.filenames, k=18)

    for index, sample in enumerate(samples, start=0):
        category = get_sample_category(sample)

        image_path = os.path.join(image_generator.directory, sample)
        sample = image.load_img(image_path, target_size=IMAGE_SIZE)

        plt.subplot(6, 3, index + 1)
        plt.imshow(sample)
        plt.xlabel(category)

    plt.tight_layout()
    plt.show()


def plot_visualize_training(history):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    ax1.plot(history.history['loss'], color='b', label="Training loss")
    ax1.plot(history.history['val_loss'], color='r', label="validation loss")
    ax1.set_xticks(np.arange(1, NUMBER_EPOCHS, 1))
    ax1.set_yticks(np.arange(0, 1, 0.1))

    ax2.plot(history.history['accuracy'], color='b', label="Training accuracy")
    ax2.plot(history.history['val_accuracy'], color='r', label="Validation accuracy")
    ax2.set_xticks(np.arange(1, NUMBER_EPOCHS, 1))

    legend = plt.legend(loc='best', shadow=True)
    plt.tight_layout()
    plt.show()


def main():
    #
    # Get training and validation images from directory
    #
    image_datagenerator = image.ImageDataGenerator(rescale=1 / 255.)
    training_generator = image_datagenerator.flow_from_directory(
        directory=os.path.join(os.getcwd(), 'datasets', 'images', 'cats_and_dogs', 'train'),
        target_size=IMAGE_SIZE,
        class_mode='binary',
        batch_size=20
    )

    validation_generator = image_datagenerator.flow_from_directory(
        directory=os.path.join(os.getcwd(), 'datasets', 'images', 'cats_and_dogs', 'validation'),
        target_size=IMAGE_SIZE,
        class_mode='binary',
        batch_size=20
    )

    #
    # See sample images
    #
    plot_sample_images(training_generator)

    #
    # Build model
    #
    model = keras.models.Sequential(
        [
            keras.layers.Conv2D(16, (3, 3),
                                input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS),
                                activation=keras.activations.relu),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(32, (3, 3), activation=keras.activations.relu),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation=keras.activations.relu),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation=keras.activations.relu),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(512, activation=keras.activations.relu),
            keras.layers.Dense(1, activation=keras.activations.sigmoid)
        ]
    )

    model.compile(
        loss=keras.losses.BinaryCrossentropy(),
        optimizer=keras.optimizers.RMSprop(learning_rate=0.001),
        metrics=['accuracy']
    )

    model.summary()

    #
    # Fit Model
    #
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            min_delta=1e-3,
            patience=5
        ),
        CustomCallback()
    ]

    history = model.fit(
        training_generator,
        steps_per_epoch=100,
        epochs=NUMBER_EPOCHS,
        validation_data=validation_generator,
        validation_steps=50,
        verbose=2,
        callbacks=callbacks
    )

    plot_visualize_training(history)


if __name__ == '__main__':
    main()
