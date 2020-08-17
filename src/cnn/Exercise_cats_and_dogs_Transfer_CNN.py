import os
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3

IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 3

# Callbacks
callbacks = [

]

# Import a previously trained model
local_weights_file = os.path.join(os.getcwd(), 'transfer', 'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')

pre_trained_model = InceptionV3(
    input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS),
    include_top=False,
    weights=None
)

pre_trained_model.load_weights(local_weights_file)

for layer in pre_trained_model.layers:
    layer.trainable = False

pre_trained_model.summary()

print('last layer output shape: ', pre_trained_model.get_layer('mixed7').output_shape)

# Use previously trained model as input for new model
x = keras.layers.Flatten()(pre_trained_model.get_layer('mixed7').output)
x = keras.layers.Dense(units=1024, activation=keras.activations.relu)(x)
x = keras.layers.Dropout(0.2)(x)
x = keras.layers.Dense(units=1, activation=keras.activations.sigmoid)(x)

model = keras.Model(
    pre_trained_model.input,
    x
)

model.compile(
    optimizer=keras.optimizers.RMSprop(learning_rate=0.001),
    loss=keras.losses.BinaryCrossentropy(),
    metrics=['accuracy']
)

model.summary()

# Image Load from local directory - Data augmentation used to increase test data
image_data_generator = image.ImageDataGenerator(
    height_shift_range=0.2,
    width_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=40,
    horizontal_flip=True,
    rescale=1/255.
)

training_image_data = image_data_generator.flow_from_directory(
    directory=os.path.join(os.getcwd(),'datasets','images', 'cats_and_dogs', 'train'),
    class_mode='binary',
    target_size=IMAGE_SIZE,
    batch_size=20
)
validation_image_data = image_data_generator.flow_from_directory(
    directory=os.path.join(os.getcwd(), 'datasets', 'images', 'cats_and_dogs', 'validation'),
    class_mode='binary',
    target_size=IMAGE_SIZE,
    batch_size=20
)

# Training and validate new model
history = model.fit(
    training_image_data,
    steps_per_epoch=100,
    epochs=20,
    validation_data=validation_image_data,
    validation_steps=50,
    verbose=2,
    callbacks=callbacks
)

# Plot results and check for overfiting
epochs = range(len(history.history['accuracy']))

plt.plot(epochs, history.history['accuracy'], 'r', label='Training accuracy')
plt.plot(epochs, history.history['val_accuracy'], 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()

plt.show()
