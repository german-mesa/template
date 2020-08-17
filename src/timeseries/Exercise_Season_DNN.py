import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

AMPLITUDE = 40
BASELINE = 10
SLOPE = 0.005
PERIOD = 365
NOISE_LEVEL = 3
SEED = 51

SPLIT_TIME = 3000
WINDOW_SIZE = 20
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000

FRESH_RUN = False


def plot_series(time, series, format='-', start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.grid(True)


def trend(time, slope):
    return time * slope


def seasonal_pattern(season_time):
    return np.where(season_time < 0.1,
                    np.cos(season_time * 6 * np.pi),
                    2 / np.exp(9 * season_time))


def seasonality(time, period, amplitude=1, phase=0):
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)


def noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level


def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(size=window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(buffer_size=shuffle_buffer).map(lambda window: (window[:-1], window[-1]))
    dataset = dataset.batch(batch_size=batch_size).prefetch(1)
    return dataset


def main():
    # Prepare training and validation data from synthetic function
    time = np.arange(10 * PERIOD + 1, dtype='float32')

    series = BASELINE + \
             trend(time, SLOPE) + \
             seasonality(time, period=PERIOD, amplitude=AMPLITUDE) + \
             noise(time, noise_level=NOISE_LEVEL, seed=SEED)

    plot_series(time, series)
    plt.show()

    train_time = time[:SPLIT_TIME]
    train_series = series[:SPLIT_TIME]

    validation_time = time[SPLIT_TIME:]
    validation_series = series[SPLIT_TIME:]

    # Prepare dataset from training data
    dataset = windowed_dataset(train_series,
                               window_size=WINDOW_SIZE,
                               batch_size=BATCH_SIZE,
                               shuffle_buffer=SHUFFLE_BUFFER_SIZE)

    # Define model
    checkpoint_path = "checkpoints/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            verbose=1,
            save_weights_only=True,
            period=5)
    ]

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(100, activation=tf.keras.activations.relu, input_shape=[WINDOW_SIZE]),
            tf.keras.layers.Dense(10, activation=tf.keras.activations.relu),
            tf.keras.layers.Dense(1)
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=1e-6, momentum=0.9),
        loss=tf.keras.losses.MeanSquaredError()
    )

    if FRESH_RUN:
        model.fit(dataset, epochs=100, verbose=2, callbacks=callbacks)
    else:
        model.load_weights(filepath=tf.train.latest_checkpoint(checkpoint_dir))

    # Plot results
    print('Forecasting the results...')
    forecast = []
    for time in range(len(series) - WINDOW_SIZE):
        forecast.append(model.predict(series[time:time + WINDOW_SIZE][np.newaxis]))

    forecast = forecast[SPLIT_TIME - WINDOW_SIZE:]
    results = np.array(forecast)[:, 0, 0]

    plt.figure(figsize=(10, 6))

    plot_series(validation_time, validation_series)
    plot_series(validation_time, results)
    plt.show()

    print(f'MAE for this model is: {tf.keras.metrics.mean_absolute_error(validation_series, results).numpy()}')


if __name__ == '__main__':
    main()
