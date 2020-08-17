import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras.preprocessing import image

NUMBER_EPOCHS = 100

IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 3

callbacks = [

]


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
    model = keras.models.Sequential(
        [
            keras.layers.Conv2D(16, (3, 3), input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS),
                                activation=keras.activations.relu),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(32, (3, 3), activation=keras.activations.relu),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation=keras.activations.relu),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation=keras.activations.relu),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(units=512, activation=keras.activations.relu),
            keras.layers.Dense(units=1, activation=keras.activations.sigmoid)
        ]
    )
    model.compile(
        loss=keras.losses.BinaryCrossentropy(),
        optimizer=keras.optimizers.RMSprop(learning_rate=0.001),
        metrics=['accuracy']
    )

    model.summary()

    image_data_generator = image.ImageDataGenerator(
        rotation_range=90,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        rescale=1 / 255.
    )

    training_dataset = image_data_generator.flow_from_directory(
        directory=os.path.join(os.getcwd(), 'datasets', 'images', 'cats_and_dogs', 'train'),
        target_size=IMAGE_SIZE,
        class_mode='binary',
        batch_size=20
    )

    testing_dataset = image_data_generator.flow_from_directory(
        directory=os.path.join(os.getcwd(), 'datasets', 'images', 'cats_and_dogs', 'validation'),
        target_size=IMAGE_SIZE,
        class_mode='binary',
        batch_size=20
    )

    history = model.fit(
        training_dataset,
        steps_per_epoch=100,
        epochs=NUMBER_EPOCHS,
        validation_data=testing_dataset,
        validation_steps=50,
        verbose=2,
        callbacks=callbacks
    )

    plot_visualize_training(history)


if __name__ == '__main__':
    main()
