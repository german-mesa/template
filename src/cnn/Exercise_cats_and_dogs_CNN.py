import os
import zipfile
import random
import urllib.request

import numpy as np
import matplotlib.pyplot as plt

from shutil import copyfile
from tensorflow import keras
from tensorflow.keras.preprocessing import image

NUMBER_EPOCHS = 100

DATASET_URL = "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip"

DATASET_PATH = "datasets/images"
DATASET_FILE = "kagglecatsanddogs_3367a.zip"

SPLIT_SIZE = 0.9

IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 3

callbacks = [

]


def get_dataset_folder():
    return os.path.join(os.getcwd(), DATASET_PATH)


def get_dataset_file():
    return os.path.join(get_dataset_folder(), DATASET_FILE)


def get_cat_source_dir():
    return os.path.join(get_dataset_folder(), 'PetImages', 'Cat')


def get_cat_training_destination_dir():
    return os.path.join(get_dataset_folder(), 'cats_and_dogs', 'train', 'cats')


def get_cat_testing_destination_dir():
    return os.path.join(get_dataset_folder(), 'cats_and_dogs', 'validation', 'cats')


def get_dog_source_dir():
    return os.path.join(get_dataset_folder(), 'PetImages', 'Dog')


def get_dog_training_destination_dir():
    return os.path.join(get_dataset_folder(), 'cats_and_dogs', 'train', 'dogs')


def get_dog_testing_destination_dir():
    return os.path.join(get_dataset_folder(), 'cats_and_dogs', 'validation', 'dogs')


def get_image_dataset():
    if os.path.isfile(get_dataset_file()):
        return

    urllib.request.urlretrieve(DATASET_URL, get_dataset_file())

    with zipfile.ZipFile(get_dataset_file(), 'r') as zp:
        zp.extractall(get_dataset_folder())
        zp.close()


def copy_files_list(file_list, source, destination):
    print(f'Copying {len(file_list)} files from {source} to {destination}')
    for item in file_list:
        copyfile(os.path.join(source, item), os.path.join(destination, item))


def split_image_dataset(source, training_destination, testing_destination, split_size):
    file_list = [item for item in os.listdir(path=source) if os.path.getsize(os.path.join(source, item)) > 0]

    random.shuffle(file_list)

    training_list = file_list[: int(len(file_list) * split_size)]
    testing_list = file_list[int(len(file_list) * split_size):]

    copy_files_list(training_list, source, training_destination)
    copy_files_list(testing_list, source, testing_destination)


def get_image_generator(dataset_type):
    image_data_generator = image.ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        rescale=1 / 255.,
        fill_mode='nearest'
    )

    return image_data_generator.flow_from_directory(
        directory=os.path.join(os.getcwd(), 'datasets', 'images', 'cats_and_dogs', dataset_type),
        target_size=IMAGE_SIZE,
        batch_size=20,
        class_mode='binary'
    )


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
    # Download image dataset
    get_image_dataset()

    # Split image dataset
    split_image_dataset(get_cat_source_dir(), get_cat_training_destination_dir(), get_cat_testing_destination_dir(),
                        SPLIT_SIZE)
    split_image_dataset(get_dog_source_dir(), get_dog_training_destination_dir(), get_dog_testing_destination_dir(),
                        SPLIT_SIZE)

    # Image generators for our model
    training_generator = get_image_generator('train')
    validation_generator = get_image_generator('validation')

    # Model definition and training
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
            keras.layers.Dense(512, activation=keras.activations.relu),
            keras.layers.Dense(1, activation=keras.activations.sigmoid)
        ]
    )

    model.compile(
        loss=keras.losses.BinaryCrossentropy(),
        optimizer=keras.optimizers.RMSprop(learning_rate=0.001),
        metrics=['accuracy']
    )

    history = model.fit(
        training_generator,
        epochs=100,
        validation_data=validation_generator,
        callbacks=callbacks
    )

    # Plot training and validation history
    plot_visualize_training(history)


if __name__ == '__main__':
    main()
