#!/usr/bin/env python
# coding: utf-8

# In[2]:


import math
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler 
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

stock_data = pd.read_csv('01_01_18_to_13_03_23.csv')
stock_data = stock_data[['Date', 'Price']]
stock_data = stock_data.set_index('Date', drop = True)
stock_data = stock_data.iloc[::-1]
stock_data.index = pd.to_datetime(stock_data.index, format="%d/%m/%Y")
stock_data.head()


# In[3]:


plt.figure(figsize=(15, 8))
plt.title('Stock Prices History')
plt.plot(stock_data['Price'])
plt.xlabel('Date')
plt.ylabel('Prices ($)')


# In[4]:


close_prices = stock_data['Price']
values = close_prices.values
training_data_len = math.ceil(len(values)*0.8 )

#scaler = MinMaxScaler(feature_range=(0,1))
#scaled_data = scaler.fit_transform(values.reshape(-1,1))

#train_data = scaled_data[0: training_data_len, :]
values = values.reshape(-1,1)
train_data = values[0: training_data_len, :]
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
    
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


# In[5]:


test_data = values[training_data_len-60: , : ]
x_test = []
y_test = values[training_data_len:]

for i in range(60, len(test_data)):
  x_test.append(test_data[i-60:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


# In[ ]:


model = keras.Sequential()
model.add(layers.LSTM(100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(layers.LSTM(100, return_sequences=False))
model.add(layers.Dense(25))
model.add(layers.Dense(1))
model.summary()


# In[ ]:


model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, batch_size= 10, epochs=20)


# In[ ]:


predictions = model.predict(x_test)
#predictions = scaler.inverse_transform(predictions)
rmse = np.sqrt(np.mean(predictions - y_test)**2)
rmse


# In[ ]:


data = stock_data.filter(['Price'])
train = data[:training_data_len]
validation = data[training_data_len:]
validation['Predictions'] = predictions
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date')
plt.ylabel('Close Price USD ($)')
#plt.plot(train)
plt.plot(validation[['Price', 'Predictions']])
plt.legend(['Val', 'Predictions'], loc='lower right')
plt.show()


# In[ ]:


x_test
plt.plot(x_test)


# In[ ]:


x_test.shape


# In[ ]:


predictions


# In[ ]:




