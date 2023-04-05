import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers
from scipy import signal
import datetime
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
os.environ['LD_LIBRARY_PATH'] = '$LD_LIBRARY_PATH:/opt/rocm-5.3.0/lib'

weather_data = pd.read_csv('london_weather.csv')
weather_data = weather_data[['date', 'mean_temp', 'sunshine', 'global_radiation', 'max_temp', 'min_temp']]
weather_data = weather_data.set_index('date', drop=True)
weather_data.index = pd.to_datetime(weather_data.index, format="%Y%m%d")
weather_data = weather_data.interpolate(method='time')
weather_data = weather_data[:-359]
values = weather_data.values
training_data_len = math.ceil(len(values) * 0.8)

b, a = signal.iirfilter(4, 0.03, btype="low", ftype="butter")
values_filt = [None] * 5
for i in range(5):
    values_filt[i] = signal.filtfilt(b, a, values[:, i])
values_filt = np.array(values_filt).transpose()

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(values_filt)

# train_data = scaled_data[0: training_data_len, :]
# values = values.reshape(-1,1)
train_data = scaled_data[0: training_data_len, :]
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i - 60:i])
    y_train.append(train_data[i])

x_train, y_train = np.array(x_train), np.array(y_train)

test_data = scaled_data[training_data_len - 60:, :]
x_test = []
y_test = scaled_data[training_data_len:]

for i in range(60, len(test_data)):
    x_test.append(test_data[i - 60:i])

x_test = np.array(x_test)

model = keras.models.load_model("lstm_model")
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)


# generate the multi-step forecasts
def forecast(date):
    current_date = datetime.datetime.strptime(date, "%Y-%m-%d").date()  # Convert date string to date object
    last_date = datetime.date(2019, 12, 31)  # Last observed date in the dataset
    delta = current_date - last_date
    n_future = delta.days  # Difference of days in dates

    y_future = []
    x_pred = x_test[-1:, :, :]  # last observed input sequence
    y_pred = y_test[-1]  # last observed target value

    print("Predicting...")

    for i in range(n_future):
        # feed the last forecast back to the model as an input
        x_pred = np.append(x_pred[:, 1:, :], y_pred.reshape(1, 1, 5), axis=1)

        # generate the next forecast
        y_pred = model.predict(x_pred)

        # save the forecast
        y_future.append(y_pred)

    # transform the forecasts back to the original scale
    y_future = np.array(y_future).reshape(-1, 5)
    y_future = scaler.inverse_transform(y_future)
    df_future = pd.DataFrame(y_future,
                             columns=['Pred_mean_temp', 'Pred_sunshine', 'Pred_global_radiation', 'Pred_max_temp',
                                      'Pred_min_temp'])
    df_future['Date'] = pd.date_range(start='2019-12-31', periods=n_future)
    df_future = df_future.set_index('Date')

    # # For testing purposes only (Uncomment to test)
    # import matplotlib.pyplot as plt
    #
    # plt.figure(figsize=(16, 8))
    # plt.title('Forecasted mean temperature (\N{DEGREE SIGN}C)')
    # plt.xlabel('Date')
    # plt.ylabel('Mean temperature (\N{DEGREE SIGN}C)')
    # plt.plot(df_future['Pred_mean_temp'])
    # plt.legend(['Forecast', 'Predictions'], loc='lower right')
    # plt.show()

    return "The predicted mean temperature on " + str(date) + " is %.2f" % df_future['Pred_mean_temp'].iloc[-1] \
        + "\N{DEGREE SIGN}C with a variation of \u00B1%.2f" % (
                    df_future['Pred_max_temp'].iloc[-1] - df_future['Pred_min_temp'].iloc[-1]) \
        + "\N{DEGREE SIGN}C. Would you like to continue?"


# # For testing purposes only (Uncomment to test)
# forecast('2023-01-15')