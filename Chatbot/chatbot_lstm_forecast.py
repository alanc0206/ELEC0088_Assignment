## This script allows the Oracle to forecast London temperatures using a pre-trained LSTM model. ##
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from scipy import signal
import datetime
import os

# Set environment variables for AMD ROCM (comment if this is unnecessary)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
os.environ['LD_LIBRARY_PATH'] = '$LD_LIBRARY_PATH:/opt/rocm-5.3.0/lib'

# Load dataset
weather_data = pd.read_csv('london_weather.csv')
weather_data = weather_data[['date', 'mean_temp', 'sunshine', 'global_radiation', 'max_temp', 'min_temp']]
weather_data = weather_data.set_index('date', drop=True)
weather_data.index = pd.to_datetime(weather_data.index, format="%Y%m%d")
# Interpolate NaN values in dataset
weather_data = weather_data.interpolate(method='time')
values = weather_data.values
# Define train-test split
training_data_len = math.ceil(len(values) * 0.8)

# Apply low-pass 4th order Butterworth filter at period of 1/0.03 days
b, a = signal.iirfilter(4, 0.03, btype="low", ftype="butter")
values_filt = [None] * 5
for i in range(5):
    values_filt[i] = signal.filtfilt(b, a, values[:, i])
values_filt = np.array(values_filt).transpose()

# Scale and normalise data to a range between 0 to 1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(values_filt)

# Generate testing feature vector
test_data = scaled_data[training_data_len - 60:, :]
x_test = []
y_test = scaled_data[training_data_len:]

for i in range(60, len(test_data)):
    x_test.append(test_data[i - 60:i])

x_test = np.array(x_test)

# Load the pre-trained LSTM model and forecast
model = keras.models.load_model("lstm_model")
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)


# Generate the multi-step forecasts
def forecast(date):
    current_date = datetime.datetime.strptime(date, "%Y-%m-%d").date()  # Convert date string to date object
    last_date = datetime.date(2020, 12, 31)  # Last observed date in the dataset
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
    df_future['Date'] = pd.date_range(start='2020-12-31', periods=n_future)
    df_future = df_future.set_index('Date')

    # # For testing purposes only (Uncomment to test)
    # # Plot forecast
    # import matplotlib.pyplot as plt
    #
    # plt.figure(figsize=(16, 8))
    # plt.title('Forecasted mean temperature (\N{DEGREE SIGN}C)')
    # plt.xlabel('Date')
    # plt.ylabel('Mean temperature (\N{DEGREE SIGN}C)')
    # plt.plot(df_future['Pred_mean_temp'])
    # plt.legend(['Forecast', 'Predictions'], loc='lower right')
    # plt.show()

    # Return information to Oracle chatbot
    return "The predicted mean temperature on " + str(date) + " (YYYY-MM-DD) is %.2f" % df_future['Pred_mean_temp'].iloc[-1] \
        + "\N{DEGREE SIGN}C with a maximum temperature of %.2f" % df_future['Pred_max_temp'].iloc[-1] + \
        "\N{DEGREE SIGN}C and a minimum temperature of %.2f" % df_future['Pred_min_temp'].iloc[-1] \
        + "\N{DEGREE SIGN}C. Would you like to continue?"


# # For testing purposes only (Uncomment to test)
# # Calls forecast function
# forecast('2023-01-15')
