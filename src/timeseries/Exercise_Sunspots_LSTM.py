import os
import csv
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras.utils as utils

SPLIT_TIME = 2500
WINDOW_SIZE = 64
BATCH_SIZE = 256
SHUFFLE_BUFFER_SIZE = 1000

SEED = 51

OPTIMIZE_LR = False
FRESH_RUN = True


def get_data_from_file(filename):
    utils.get_file(fname=filename,
                   origin="https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv",
                   cache_dir='.')

    step = 0
    time = []
    temperature = []

    with open(os.path.join(os.getcwd(), 'datasets', filename)) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)

        for row in reader:
            temperature.append(float(row[1]))
            time.append(step)
            step = step + 1

    return np.array(temperature), np.array(time)


def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)

    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[1:]))
    ds = ds.batch(batch_size).prefetch(1)

    return ds


def model_forecast(model, series, window_size, batch_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(batch_size).prefetch(1)

    forecast = model.predict(ds)

    return forecast


def plot_history(history):
    mae = history.history['mae']
    loss = history.history['loss']
    epochs = range(len(loss))  # Get number of epochs

    plt.plot(epochs, mae, 'r')
    plt.plot(epochs, loss, 'b')
    plt.title('MAE and Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(["MAE", "Loss"])


def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)


def main():
    # Get data from external file
    series, time = get_data_from_file('daily-min-temperatures.csv')

    plt.figure(figsize=(10, 6))
    plot_series(time, series)
    plt.show()

    # Prepare training and validation sets
    train_time = time[:SPLIT_TIME]
    train_serie = series[:SPLIT_TIME]

    valid_time = time[SPLIT_TIME:]
    valid_serie = series[SPLIT_TIME:]

    # Dataset is composed of batches of:
    # - Several temperatures (Window size) that will act as features
    # - 1 temperature that will be the Y value (expected result)
    train_set = windowed_dataset(train_serie, WINDOW_SIZE, BATCH_SIZE, SHUFFLE_BUFFER_SIZE)

    # Model generation - Calculate optimal learning rate
    if OPTIMIZE_LR:
        tf.keras.backend.clear_session()
        tf.random.set_seed(SEED)

        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Conv1D(
                    filters=32,
                    kernel_size=5,
                    strides=1,
                    padding='causal',
                    activation=tf.keras.activations.relu,
                    input_shape=[None, 1]
                ),
                tf.keras.layers.LSTM(64, return_sequences=True),
                tf.keras.layers.LSTM(64, return_sequences=True),
                tf.keras.layers.Dense(30, activation=tf.keras.activations.relu),
                tf.keras.layers.Dense(10, activation=tf.keras.activations.relu),
                tf.keras.layers.Dense(1),
                tf.keras.layers.Lambda(lambda x: x * 400)
            ])

        model.summary()

        model.compile(
            loss=tf.keras.losses.Huber(),
            optimizer=tf.keras.optimizers.SGD(learning_rate=1e-8, momentum=0.9),
            metrics=['mae']
        )

        history = model.fit(
            train_set,
            epochs=100,
            callbacks=[
                tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10 ** (epoch / 20))
            ]
        )

        plt.semilogx(history.history["lr"], history.history["loss"])
        plt.axis([1e-8, 1e-4, 0, 60])
        plt.show()

    # Model generation - Execute with pre-calculated leearning rate
    checkpoint_path = "checkpoints/sun-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    tf.keras.backend.clear_session()
    tf.random.set_seed(SEED)

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv1D(
                filters=128,
                kernel_size=5,
                strides=1,
                padding='causal',
                activation=tf.keras.activations.relu,
                input_shape=[None, 1]
            ),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.Dense(30, activation=tf.keras.activations.relu),
            tf.keras.layers.Dense(10, activation=tf.keras.activations.relu),
            tf.keras.layers.Dense(1),
            tf.keras.layers.Lambda(lambda x: x * 400)
        ])

    model.summary()

    model.compile(
        loss=tf.keras.losses.Huber(),
        optimizer=tf.keras.optimizers.SGD(learning_rate=1e-5, momentum=0.9),
        metrics=['mae']
    )

    if FRESH_RUN:
        history = model.fit(
            train_set,
            epochs=200,
            callbacks=[
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=checkpoint_path,
                    verbose=1,
                    save_weights_only=True,
                    period=25)
            ]
        )

        # Plot MAE & Loss vs epoch
        plot_history(history)
        plt.show()

    else:
        model.load_weights(filepath=tf.train.latest_checkpoint(checkpoint_dir))

    # Check model forecast
    rnn_forecast = model_forecast(model, series[..., np.newaxis], WINDOW_SIZE, 32)
    rnn_forecast = rnn_forecast[SPLIT_TIME - WINDOW_SIZE:-1, -1, 0]

    plt.figure(figsize=(10, 6))
    plot_series(valid_time, valid_serie)
    plot_series(valid_time, rnn_forecast)
    plt.show()

    mae = tf.keras.metrics.mean_absolute_error(valid_serie, rnn_forecast).numpy()
    print('Mean Absolute Error for this model is {0}'.format(mae))


if __name__ == '__main__':
    main()
