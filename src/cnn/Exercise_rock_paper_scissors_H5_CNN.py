import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras.preprocessing import image

IMAGE_WIDTH = 150
IMAGE_HEIGHT = 150
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)

IMAGE_CHANNELS = 3

# Callbacks
custom_callbacks = []


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


def main():
    # Create CNN topology
    model = keras.models.Sequential(
        [
            keras.layers.Conv2D(64, (3, 3),
                                input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS),
                                activation=keras.activations.relu),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation=keras.activations.relu),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(128, (3, 3), activation=keras.activations.relu),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(128, (3, 3), activation=keras.activations.relu),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(512, activation=keras.activations.relu),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(3, activation=keras.activations.softmax)
        ]
    )

    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=0.001),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy']
    )

    model.summary()

    # Get augmented dataset - training
    training_data_generator = image.ImageDataGenerator(
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=40,
        horizontal_flip=True,
        rescale=1 / 255.
    )
    training_image_dataset = training_data_generator.flow_from_directory(
        directory=os.path.join(os.getcwd(), 'datasets', 'images', 'rps', 'train'),
        class_mode='categorical',
        batch_size=126,
        target_size=IMAGE_SIZE
    )

    # Get dataset - validation
    validation_data_generator = image.ImageDataGenerator(
        rescale=1 / 255.
    )
    validation_image_dataset = validation_data_generator.flow_from_directory(
        directory=os.path.join(os.getcwd(), 'datasets', 'images', 'rps', 'validation'),
        class_mode='categorical',
        batch_size=126,
        target_size=IMAGE_SIZE
    )

    # Train the model
    history = model.fit(
        training_image_dataset,
        steps_per_epoch=20,
        epochs=20,
        validation_data=validation_image_dataset,
        validation_steps=3,
        callbacks=custom_callbacks,
        verbose=1
    )

    model.save('rps.h5')

    # Plot training results
    plot_visualize_training(history)


if __name__ == '__main__':
    main()
