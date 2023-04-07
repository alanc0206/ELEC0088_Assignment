# Oracle weather forecasting chatbot 

Repository for the 2022/23 ELEC0088 assignment. We have trained, benchmarked, and validated ARIMA, LSTM, and FBProphet models and integrated our chosen model (LSTM) into a chatbot based on Google's Dialogflow framework.

## Environment and dependent packages

This project was developed on Python 3.10.9 using the following packages their specified versions:
- Tensorflow 2.11.0
- Pandas 1.5.2
- Scikit-learn 1.2.0
- Scipy 1.8.1
- google-api-core 1.34.0
- google-cloud-dialogflow 2.21.0
- matplotlib 3.6.2
- pmdarima 1.8.5
- statsmodels 0.13.5
- numpy 1.22.4
- prophet 1.1.2

Using other versions of these packages may result in incompatibilities or errors.

## Datasets
Our models are built for the 'London Weather Data' dataset assembled by Emmanuel F. Werr (https://www.kaggle.com/datasets/emmanuelfwerr/london-weather-data) from data provided by the European Climate Assessment (ECA) (https://www.ecad.eu/dailydata/index.php). Measurements for this dataset were collected by the Heathrow weather station in London.

## Structure of this project

### Chatbot directory

Directory to run the chatbot server and client. To run the service, open the /Chatbot directory, then have chatbot_server.py running on a server machine and connect to the server using chatbot_client.py on the client machine. Ensure that the london_weather.csv dataset and the lstm_model folder containing the saved LSTM model is in the same directory. The /Chatbot directory contains all the required files to run the server, as well as the saved LSTM model in tensorflow format. dialogflow_api.py and chatbot_lstm_forecast.py are custom modules used by the server to connect to Google Cloud's API as well as load and forecast custom trained LSTM models.

### LSTM directory (Model used by the chatbot)

Contains scripts to train, validate, and forecast using using our custom built LSTM neural network chosen to be used in our chatbot service. The directory also contains scripts and Jupyter Notebook demos to train and validate/benchmark our LSTM mdoel. LSTM_Training.py is used to train and save a new multivariate LSTM model using the london_weather.csv dataset. LSTM_Forecasting.ipynb uses the trained model to predict and plot multi-step forecasts to a desired date. Filtering.ipynb demonstrates the butterworth filter and shows the results on our dataset. LSTM_univariate.ipynb demonstrates a univariate LSTM model. 

### ARIMA directory

Directory containing files to train and validate ARIMA models. ARIMA_multi.ipynb trains a multivariate seasonal ARIMA model and show the results of a forecast by the model. ARIMA_single.ipynb and ARIMA_single_auto.ipynb trains and show the results of univariate ARIMA models.

### Prophet directory

Directory to train and show the results of FBProphet models. Besides our weather dataset, we also tested the prophet model on property prices and stock prices. 

