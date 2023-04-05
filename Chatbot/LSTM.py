import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from scipy import signal
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
os.environ['LD_LIBRARY_PATH'] = '$LD_LIBRARY_PATH:/opt/rocm-5.3.0/lib'

weather_data = pd.read_csv('london_weather.csv')
weather_data = weather_data[['date', 'mean_temp', 'sunshine','global_radiation','max_temp','min_temp']]
weather_data = weather_data.set_index('date', drop = True)
weather_data.index = pd.to_datetime(weather_data.index,format="%Y%m%d")
weather_data = weather_data.interpolate(method='time')
weather_data = weather_data[:-359]

values = weather_data.values
training_data_len = math.ceil(len(values) * 0.8)

b, a = signal.iirfilter(4, 0.03, btype="low", ftype="butter")
print("Butterworth filter: \n")
print(b, a, sep="\n")
values_filt = [None] * 5
for i in range(5):
    values_filt[i] = signal.filtfilt(b, a , values[:,i])
values_filt = np.array(values_filt).transpose()

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(values_filt)

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

model = keras.Sequential()
model.add(layers.LSTM(64, return_sequences=False, input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(5))
model.summary()

model.compile(optimizer='adam', loss='mse')
model.fit(x_train, y_train, batch_size=128, epochs=6)

predictions = model.predict(x_test)
rmse = np.sqrt(np.mean(predictions - y_test) ** 2)

predictions = scaler.inverse_transform(predictions)


validation = weather_data[training_data_len:]
df = pd.DataFrame(predictions, columns = ['Pred_mean_temp', 'Pred_sunshine','Pred_global_radiation','Pred_max_temp','Pred_min_temp'])
df.index = validation.index
validation = pd.concat([validation, df], axis=1)
plt.figure(figsize=(16, 8))
plt.title('Model')
plt.xlabel('Date')
plt.ylabel('mean_temp')

plt.plot(validation[['mean_temp', 'Pred_mean_temp']])
plt.legend(['Mean Temperature', 'Predictions'], loc='lower right')
plt.show()

# generate the multi-step forecasts
n_future = 365*2
y_future = []

x_pred = x_test[-1:, :, :]  # last observed input sequence
y_pred = y_test[-1]  # last observed target value

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
# %% raw
# df_future['Forecast'] = y_future.flatten()

df_future = pd.DataFrame(y_future, columns = ['Pred_mean_temp','Pred_sunshine','Pred_global_radiation','Pred_max_temp','Pred_min_temp'])
df_future['Date'] = pd.date_range(start='2019-12-31', periods=n_future)
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
print("Model saved.")
# %% raw