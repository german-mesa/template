import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras, train

FRESH_RUN = True
NUMBER_EPOCHS = 50
CONVOLUTION_NUMBER = 2


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


def visualizing_convolutions_and_pooling(model, images):
    layer_outputs = [layer.output for layer in model.layers]
    activation_model = keras.models.Model(inputs=model.input, outputs=layer_outputs)

    f, axarr = plt.subplots(len(images), 4)

    for x in range(0, 4):
        for i in range(0, len(images)):
            figure = activation_model.predict(test_images[i].reshape(1, 28, 28, 1))[x]
            axarr[i, x].imshow(figure[0, :, :, CONVOLUTION_NUMBER], cmap='inferno')
            axarr[i, x].grid(False)

    plt.show()


class CustomCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.998:
            self.model.stop_training = True


checkpoint_path = "checkpoints/mnist-fashion-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

callbacks = [
    CustomCallback(),
    keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1,
        save_weights_only=True,
        period=10)
]

(training_images, training_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()

training_images = np.reshape(training_images, [len(training_images), 28, 28, 1])
test_images = np.reshape(test_images, [len(test_images), 28, 28, 1])

training_images = training_images / 255.0
test_images = test_images / 255.0

model = keras.models.Sequential(
    [
        keras.layers.Conv2D(64, (3, 3), activation=keras.activations.relu, input_shape=(28, 28, 1)),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation=keras.activations.relu),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation=keras.activations.relu),
        keras.layers.Dense(10, activation=keras.activations.softmax)
    ]
)

print(model.summary())

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

if FRESH_RUN:
    history = model.fit(
        training_images,
        training_labels,
        validation_data=(test_images, test_labels),
        epochs=NUMBER_EPOCHS,
        callbacks=callbacks
    )

    plot_visualize_training(history)

else:
    model.load_weights(filepath=train.latest_checkpoint(checkpoint_dir))

# Visualization
visualizing_convolutions_and_pooling(model, [0, 7, 26])

# Predictions
predictions = model.predict(test_images)
print(predictions)

# Image 0 prediction and actual
prediction = predictions[0][test_labels[0]]
print(f'Label is: {test_labels[0]}')
print(f'Prediction is: {prediction}')


model.save('Exercise_MNIST_Fashion_CNN.h5')