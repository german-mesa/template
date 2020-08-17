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

FRESH_RUN = True


def plot_series(time, series, format='-', start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.grid(True)


def trend(time, slope):
    return time * slope


def seasonal_pattern(season_time):
    return np.where(season_time < 0.4,
                    np.cos(season_time * 2 * np.pi),
                    2 / np.exp(9 * season_time))


def seasonality(time, period, amplitude=1, phase=0):
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)


def noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level


def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, 1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset


def scheduler(epoch):
    if epoch < 10:
        return 1e-6
    else:
        return 1e-6 * tf.math.exp(epoch / 20)


def main():
    # Datasets
    time = np.arange(10 * PERIOD + 1, dtype='float32')
    series = BASELINE + trend(time, SLOPE) + seasonality(time, PERIOD, AMPLITUDE) + noise(time, NOISE_LEVEL, SEED)

    train_time = time[:SPLIT_TIME]
    train_series = series[:SPLIT_TIME]

    validation_time = time[SPLIT_TIME:]
    validation_series = series[SPLIT_TIME:]

    dataset = windowed_dataset(train_series, WINDOW_SIZE, BATCH_SIZE, SHUFFLE_BUFFER_SIZE)

    # Define the model
    checkpoint_path = "checkpoints/lstm-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    callbacks = [
        #    tf.keras.callbacks.LearningRateScheduler(
        #        lambda epoch: 1e-8 * 10 ** (epoch / 20)),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            verbose=1,
            save_weights_only=True,
            period=25)
    ]

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1), input_shape=[None]),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
            tf.keras.layers.Dense(1),
            tf.keras.layers.Lambda(lambda x: x * 100.0)
        ]
    )

    model.compile(
        #    optimizer=tf.keras.optimizers.SGD(learning_rate=1e-8, momentum=0.9),
        optimizer=tf.keras.optimizers.SGD(learning_rate=1e-5, momentum=0.9),
        loss=tf.keras.losses.Huber(),
        metrics=['mae']
    )

    if FRESH_RUN:
        history = model.fit(
            dataset,
            #    epochs=100,
            epochs=500,
            callbacks=callbacks
        )

        # plt.semilogx(history.history["lr"], history.history["loss"])
        # plt.axis([1e-8, 1e-4, 0, 30])
        # plt.show()
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

    # -----------------------------------------------------------
    # Retrieve a list of list results on training and test data
    # sets for each training epoch
    # -----------------------------------------------------------
    mae = history.history['mae']
    loss = history.history['loss']

    epochs = range(len(loss))  # Get number of epochs

    # ------------------------------------------------
    # Plot MAE and Loss
    # ------------------------------------------------
    plt.plot(epochs, mae, 'r')
    plt.plot(epochs, loss, 'b')
    plt.title('MAE and Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(["MAE", "Loss"])

    plt.figure()

    epochs_zoom = epochs[200:]
    mae_zoom = mae[200:]
    loss_zoom = loss[200:]

    # ------------------------------------------------
    # Plot Zoomed MAE and Loss
    # ------------------------------------------------
    plt.plot(epochs_zoom, mae_zoom, 'r')
    plt.plot(epochs_zoom, loss_zoom, 'b')
    plt.title('MAE and Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(["MAE", "Loss"])

    plt.figure()
    plt.show()


if __name__ == '__main__':
    main()
