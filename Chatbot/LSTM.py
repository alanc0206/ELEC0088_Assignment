import math
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

weather_data = pd.read_csv('london_weather.csv')
weather_data = weather_data[['date', 'mean_temp']]
weather_data = weather_data.set_index('date', drop = True)
weather_data.index = pd.to_datetime(weather_data.index,format="%Y%m%d")
weather_data = weather_data.fillna(method='ffill')
weather_data.head()

plt.figure(figsize=(15, 8))
plt.title('Mean Temp History')
plt.plot(weather_data['mean_temp'])
plt.xlabel('Date')
plt.ylabel('Temp')

# close_prices = stock_data['Price']
values = weather_data.values
training_data_len = math.ceil(len(values) * 0.8)

# scaler = MinMaxScaler(feature_range=(0,1))
# scaled_data = scaler.fit_transform(values.reshape(-1,1))

# train_data = scaled_data[0: training_data_len, :]
values = values.reshape(-1, 1)
train_data = values[0: training_data_len, :]
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i - 60:i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

test_data = values[training_data_len-60: , : ]
x_test = []
y_test = values[training_data_len:]

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
#%%
model = keras.Sequential()
model.add(layers.LSTM(100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(layers.LSTM(100, return_sequences=False))
model.add(layers.Dense(25))
model.add(layers.Dense(1))
model.summary()

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, batch_size= 10, epochs=20)