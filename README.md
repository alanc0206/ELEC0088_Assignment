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

Using other versions of the same packages may result in incompatibilities or errors.

## Datasets
Our models are built for the 'London Weather Data' dataset assembled by Emmanuel F. Werr (https://www.kaggle.com/datasets/emmanuelfwerr/london-weather-data) from data provided by the European Climate Assessment (ECA) (https://www.ecad.eu/dailydata/index.php). Measurements for this dataset were collected by the Heathrow weather station in London.

## Structure of this project

To run this project, open the /Chatbot directory, then have chatbot_server.py running on a server machine and connect to the server using chatbot_client.py on the client machine. Ensure that the london_weather.csv dataset is in the same directory. The /Chatbot directory contains all the required files to run the server, as well as files required to load the chosen LSTM prediction model to provide the service with weather forecasting functionalities. The /Chatbot directory also contains scripts and Jupyter Notebook demos to train and validate/benchmark our LSTM mdoel. The /ARIMA directory contains scripts to train, benchmark, and validate our ARIMA model. The /Prophet directory contains scripts to train, benchmark, and validate our FBProphet model.

