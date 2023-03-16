import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from statsmodels.tools.eval_measures import rmse
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from sklearn import preprocessing

df = pd.read_csv('01_01_18_to_13_03_23.csv')
df = df[['Date', 'Price']]
df = df.set_index('Date', drop = True)
df = df.iloc[::-1]
df.index = pd.to_datetime(df.index, format="%d/%m/%Y")

# Inference
test_data = df.tail(365)
df = df.iloc[:-365]

plt.figure(figsize=(16, 7))
plt.plot(df.loc['2018-01-02':'2023-03-13','Price'],
        label='Predicted', linestyle='-',  c='b')


# Prepare data for LSTM model by incoorporating timesteps of 60
def prepare_data(data):
 x_train = []
 y_train = []
 for i in range(60, len(data)):
  x_train.append(data[i - 60:i, 0])
  y_train.append(data[i, 0])
 x_train = np.array(x_train)
 y_train = np.array(y_train)
 x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

 return x_train, y_train


# Build and train LSTM model
def train_model(x_train, y_train):
 model = Sequential()
 n_neurons = x_train.shape[1] * x_train.shape[2]
 model.add(LSTM(n_neurons, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
 model.add(LSTM(n_neurons, return_sequences=False))
 model.add(Dense(5))
 model.add(Dense(1))
 model.compile(optimizer='adam', loss='mean_squared_error')
 model.fit(x_train, y_train, epochs=500, batch_size=10, validation_split=0.2)

 return model


# Make prediction
def make_prediction(data, x_test_data):
 x_train, y_train = prepare_data(data)
 LSTM_model = train_model(x_train, y_train)
 forecast = LSTM_model.predict(x_test_data)
 # forecast = scaler.inverse_transform(forecast)
 return forecast


# Plot predictions and actual closing price
def plot_predictions(total_data, test_data, forecast_data, title):
 test_data['Prediction'] = forecast_data
 total_data.index = pd.to_datetime(total_data.index)
 test_data.index = pd.to_datetime(test_data.index)
 plt.figure(figsize=(16, 7))
 plt.plot(total_data.loc['2021-09-29':'2023-03-13', 'Price'],
          label='Actual', linestyle='-', c='r')
 plt.plot(test_data.loc['2021-09-29':'2023-03-13', 'Prediction'],
          label='Predicted', linestyle='-', c='b')

 plt.xlabel('Date', fontsize='18')
 plt.ylabel('Stock Price', fontsize='18')
 plt.title('Stock Prediction ' + title, fontsize='20')

 plt.grid()
 plt.legend()
 plt.savefig('Stock Prediction ' + title + '.png',
             bbox_inches="tight",
             pad_inches=0.5,
             transparent=True,
             facecolor="w",
             edgecolor='w',
             orientation='landscape')

 # Prepare test data
 hist_data = pd.DataFrame(df['Price'])
 # hist_data.index = hist_data.index.strftime('%Y-%m-%d')
 dataset_total = pd.concat((hist_data, test_data), axis=0)
 inputs = dataset_total.iloc[len(dataset_total) - len(test_data) - 60:].values
 inputs = inputs.reshape(-1, 1)
 # scaler = preprocessing.MinMaxScaler()
 # inputs = scaler.fit_transform(inputs)
 x_test, y_test = prepare_data(inputs)

 # LSTM Model
 train = df.values
 forecast = make_prediction(train, x_test)
 plot_predictions(dataset_total, test_data, forecast, 'Prediction')