# %% raw
import math
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
os.environ['LD_LIBRARY_PATH'] = '$LD_LIBRARY_PATH:/opt/rocm-5.3.0/lib'

weather_data = pd.read_csv('london_weather.csv')
weather_data = weather_data[['date', 'max_temp', 'mean_temp', 'min_temp']]
weather_data = weather_data.set_index('date', drop = True)
weather_data.index = pd.to_datetime(weather_data.index,format="%Y%m%d")
weather_data = weather_data.fillna(method='ffill')
weather_data.iloc[60]

# %% raw
values = weather_data.values
training_data_len = math.ceil(len(values) * 0.8)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(values)

# train_data = scaled_data[0: training_data_len, :]
# values = values.reshape(-1,1)
train_data = scaled_data[0: training_data_len, :]
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i - 60:i])
    y_train.append(train_data[i])

x_train, y_train = np.array(x_train), np.array(y_train)

# x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
# %% raw
test_data = scaled_data[training_data_len - 60:, :]
x_test = []
y_test = scaled_data[training_data_len:]

for i in range(60, len(test_data)):
    x_test.append(test_data[i - 60:i])

x_test = np.array(x_test)
# x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))
# %% raw
model = keras.Sequential()
model.add(layers.LSTM(100, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(layers.LSTM(100, return_sequences=False))
model.add(layers.Dense(25))
model.add(layers.Dense(3))
model.summary()
# %% raw
model.compile(optimizer='adam', loss='mean_squared_error')
#early_stopper = keras.callbacks.EarlyStopping(monitor='loss', min_delta=0, patience=3, verbose=1, mode='min')
model.fit(x_train, y_train, batch_size=20, epochs=10)
# %% raw
predictions = model.predict(x_test)
# predictions = scaler.inverse_transform(predictions)
rmse = np.sqrt(np.mean(predictions - y_test) ** 2)
rmse
# %% raw
predictions = scaler.inverse_transform(predictions)

# %% raw
data = weather_data
train = data[:training_data_len]
validation = data[training_data_len:]
df = pd.DataFrame(predictions, columns=['Pred_max_temp', 'Pred_mean_temp', 'Pred_min_temp'])
df.index = validation.index
validation = pd.concat([validation, df], axis=1)
plt.figure(figsize=(16, 8))
plt.title('Model')
plt.xlabel('Date')
plt.ylabel('mean_temp')
# plt.plot(train)
plt.plot(validation[['mean_temp', 'Pred_mean_temp']])
plt.legend(['Val', 'Predictions'], loc='lower right')
plt.show()
# %% raw
# generate the multi-step forecasts
n_future = 1825
y_future = []

x_pred = x_test[-1:, :, :]  # last observed input sequence
y_pred = y_test[-1]  # last observed target value

for i in range(n_future):
    # feed the last forecast back to the model as an input
    x_pred = np.append(x_pred[:, 1:, :], y_pred.reshape(1, 1, 3), axis=1)

    # generate the next forecast
    y_pred = model.predict(x_pred)

    # save the forecast
    y_future.append(y_pred)

# transform the forecasts back to the original scale
y_future = np.array(y_future).reshape(-1, 3)
y_future = scaler.inverse_transform(y_future)
# %% raw
# df_future['Forecast'] = y_future.flatten()
df_future = pd.DataFrame(y_future,
                         columns=['Pred_max_temp',
                                  'Pred_mean_temp', 'Pred_min_temp'])
df_future['Date'] = pd.date_range(start='2020-12-31', periods=n_future)
df_future = df_future.set_index('Date')

plt.figure(figsize=(16, 8))
plt.title('Model')
plt.xlabel('Date')
plt.ylabel('mean_temp')
plt.plot(df_future['Pred_mean_temp'])
plt.plot(validation[['mean_temp', 'Pred_mean_temp']])
plt.legend(['Forecast', 'Val', 'Predictions'], loc='lower right')
plt.show()
# %% raw
model.save("lstm_model" , save_format="tf")
# %% raw
