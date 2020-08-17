import os

from tensorflow import keras
from tensorflow.keras.preprocessing import image


class CustomCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.999:
            print("\nReached 99.9% accuracy so cancelling training!")
            self.model.stop_training = True


callbaks = [
    CustomCallback()
]

model = keras.models.Sequential(
    [
        keras.layers.Conv2D(16, (3, 3), input_shape=(150, 150, 3), activation=keras.activations.relu),
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

train_datagen = image.ImageDataGenerator(rescale=1/255)
train_generator = train_datagen.flow_from_directory(
    directory=os.path.join(os.getcwd(), 'datasets'),
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

history = model.fit(
    train_generator,
    epochs=50,
    steps_per_epoch=2,
    verbose=2,
    callbacks=callbaks
)