import os
import csv
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras.preprocessing import image

IMAGE_PATH = 'datasets/images/sign'
IMAGE_CATEGORIES = 25
IMAGE_SIZE = 28
IMAGE_CHANNELS = 1
IMAGE_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS)

BATCH_SIZE = 32

# Callback functions
custom_callback = [
    keras.callbacks.EarlyStopping(
        monitor='accuracy',
        min_delta=1e-2,
        patience=3
    )
]


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


# Get image data from CSV files
def get_data(filename):
    labels = []
    images = []

    with open(filename, 'r') as file:
        reader = csv.reader(file, delimiter=',')
        next(reader)

        for row in reader:
            labels.append(row[0])
            images.append(np.array_split(row[1:], IMAGE_SIZE))

    return np.array(images, dtype='float'), np.array(labels, dtype='int')


def main():
    # Get data from CSV files
    training_images, training_labels = get_data(os.path.join(os.getcwd(), IMAGE_PATH, 'sign_mnist_train.csv'))
    testing_images, testing_labels = get_data(os.path.join(os.getcwd(), IMAGE_PATH, 'sign_mnist_test.csv'))

    print(training_images.shape)
    print(training_labels.shape)
    print(testing_images.shape)
    print(testing_labels.shape)

    # Expand the vectors in one extra dimension
    training_images = np.expand_dims(training_images, axis=3)
    testing_images = np.expand_dims(testing_images, axis=3)

    print(training_images.shape)
    print(testing_images.shape)

    # Get augmented data for training
    training_images_generator = image.ImageDataGenerator(
        width_shift_range=0.2,
        height_shift_range=0.2,
        rotation_range=40,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        rescale=1 / 255.,
        fill_mode='nearest'
    )
    training_images_flow = training_images_generator.flow(training_images,
                                                          training_labels,
                                                          batch_size=BATCH_SIZE)

    testing_images_generator = image.ImageDataGenerator(
        rescale=1 / 255.
    )
    validation_images_flow = testing_images_generator.flow(testing_images,
                                                           testing_labels,
                                                           batch_size=BATCH_SIZE)

    # Create CNN topology
    model = keras.Sequential(
        [
            keras.layers.Conv2D(64, (3, 3),
                                input_shape=(IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS),
                                activation=keras.activations.relu),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(128, (3, 3), activation=keras.activations.relu),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(1024, activation=keras.activations.relu),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(IMAGE_CATEGORIES, activation=keras.activations.softmax)
        ]
    )

    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=0.001),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    # Print the model summary
    model.summary()

    # Train the model
    history = model.fit(
        training_images_flow,
        steps_per_epoch=len(training_labels) / BATCH_SIZE,
        epochs=20,
        validation_data=validation_images_flow,
        validation_steps=len(testing_images) / BATCH_SIZE,
        verbose=1
    )

    # Save model
    model.save('signs.h5')

    # Plot training results
    plot_visualize_training(history)


if __name__ == '__main__':
    main()
