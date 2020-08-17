import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras

# Some parameters for the serie
BASELINE = 10
AMPLITUDE = 40
SLOPE = 0.05
PERIOD = 365
NOISE = 5
SEED = 42
WINDOW_SIZE = 50


def plot_series(time, series, format='-', start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.grid(True)


def trend(time, slope=0):
    return time * slope


def seasonal_pattern(season_time):
    return np.where(season_time < 0.4,
                    np.cos(season_time * 2 * np.pi),
                    1 / np.exp(3 * season_time))


def seasonality(time, period, amplitude=1, phase=0):
    # % is rest of the division, so:
    # time % period = 1 if we're at t=1
    # time % period = 2 if we're at t=2
    # ...
    # this will linearly increase till
    # time % period = 0 if we're at t=period
    #
    # You can plot it using:
    # plot_series(time, season_time)
    #
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time=season_time)


def noise(time, noise_level=1, seed=None):
    # Returns an array with random number per time unit
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level


# Create the series
time = np.arange(4 * PERIOD + 1, dtype="float32")
series = BASELINE + \
         trend(time, SLOPE) + \
         seasonality(time, period=PERIOD, amplitude=AMPLITUDE) + \
         noise(time, NOISE, seed=SEED)

plt.figure(figsize=(10, 6))
plot_series(time, series)
plt.show()

# Fixed partitioning for start
split_time = 1000

time_train = time[:split_time]
value_train = series[:split_time]

plt.figure(figsize=(10, 6))
plot_series(time_train, value_train)
plt.show()

time_valid = time[split_time:]
value_valid = series[split_time:]

plt.figure(figsize=(10, 6))
plot_series(time_valid, value_valid)
plt.show()

# Naive forecast - lags 1 step behind the time series - orange line in our case
naive_forecast = series[split_time - 1:-1]

plt.figure(figsize=(10, 6))
plot_series(time_valid, value_valid)
plot_series(time_valid, naive_forecast)
plt.show()

plt.figure(figsize=(10, 6))
plot_series(time_valid, value_valid, start=0, end=150)
plot_series(time_valid, naive_forecast, start=1, end=151)
plt.show()

print(f'MSE for Naive Forecast {keras.metrics.mean_squared_error(value_valid, naive_forecast).numpy()}')
print(f'MAE for Naive Forecast {keras.metrics.mean_absolute_error(value_valid, naive_forecast).numpy()}')


# Moving average - mean of next window_size points - worst than naive forecast indeed
def moving_average_forecast(series, window_size):
    """Forecasts the mean of the last few values.
     If window_size=1, then this is equivalent to naive forecast"""
    forecast = []
    for time in range(len(series) - window_size):
        forecast.append(series[time:time + window_size].mean())
    return np.array(forecast)


moving_avg = moving_average_forecast(series, WINDOW_SIZE)[split_time - WINDOW_SIZE:]

plt.figure(figsize=(10, 6))
plot_series(time_valid, value_valid)
plot_series(time_valid, moving_avg)
plt.show()

print(f'MSE for Moving Average {keras.metrics.mean_squared_error(value_valid, moving_avg).numpy()}')
print(f'MAE for Moving Average {keras.metrics.mean_absolute_error(value_valid, moving_avg).numpy()}')

# Differencing
diff_series = (series[365:] - series[:-365])
diff_time = time[365:]

plt.figure(figsize=(10, 6))
plot_series(diff_time, diff_series)
plt.show()

diff_moving_avg = moving_average_forecast(diff_series, WINDOW_SIZE)[split_time - PERIOD - WINDOW_SIZE:]

plt.figure(figsize=(10, 6))
plot_series(time_valid, diff_series[split_time - PERIOD:])
plot_series(time_valid, diff_moving_avg)
plt.show()

diff_moving_avg_plus_past = series[split_time - PERIOD:-PERIOD] + diff_moving_avg

plt.figure(figsize=(10, 6))
plot_series(time_valid, value_valid)
plot_series(time_valid, diff_moving_avg_plus_past)
plt.show()

print(f'MSE for Differencing Moving Average {keras.metrics.mean_squared_error(value_valid, diff_moving_avg_plus_past).numpy()}')
print(f'MAE for Differencing Moving Average {keras.metrics.mean_absolute_error(value_valid, diff_moving_avg_plus_past).numpy()}')