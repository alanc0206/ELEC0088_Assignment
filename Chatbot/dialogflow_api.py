from google.cloud import dialogflow_v2 as dialogflow
from google.api_core.exceptions import InvalidArgument
import pandas as pd
import numpy as np
from tensorflow import keras
import os

# Define API credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'elec0088-chatbot-frqp-e1e8b6b5eff0.json'

# Define agent parameters
DIALOGFLOW_PROJECT_ID = 'elec0088-chatbot-frqp'
DIALOGFLOW_AGENT_LANGUAGE = 'en'


class BotApi:

    # Constructor that starts a new session from uuid
    def __init__(self, session_id):
        self.session_client = dialogflow.SessionsClient()
        self.session = self.session_client.session_path(DIALOGFLOW_PROJECT_ID, session_id)

    # Fulfillment method
    def fulfill(self, text_to_bot):
        # Get text from client and rest api server
        text_input = dialogflow.TextInput(text=text_to_bot, language_code=DIALOGFLOW_AGENT_LANGUAGE)
        query = dialogflow.QueryInput(text=text_input)
        try:
            # Get response from dialogflow agent
            response = self.session_client.detect_intent(session=self.session, query_input=query)
            # Begin weather prediction when asked to do so
            if response.query_result.fulfillment_text.startswith('Find information'):
                # Get date
                date = response.query_result.fulfillment_text.split('date')[1]
                try:
                    # Call prediction method
                    return prophet_predict(date)
                except KeyError:
                    # Catch exception if date is invalid
                    return "That date is invalid or too far away to predict. Continue?"
        except InvalidArgument:
            return 'Invalid argument'

        # Return fulfillment text
        return response.query_result.fulfillment_text


def prophet_predict(date):
    df = pd.read_csv('prophet weather forecast.csv')
    df2 = df.drop(df.columns[0], axis=1)
    df2.index = df.ds
    df2 = df2.drop(df2.columns[0], axis=1)
    temps = df2.loc[date]
    return "The predicted mean temperature on " + str(date) + " is %.2f" % temps.iloc[0] \
        + "\N{DEGREE SIGN}C with a variation of \u00B1%.2f" % (temps.iloc[2] - temps.iloc[1]) \
        + "\N{DEGREE SIGN}C. Would you like to continue?"

def lstm_predict(date):
    model = keras.models.load_model("lstm_model")
    n_future = date - y_test[-1]
    for i in range(n_future):
        # feed the last forecast back to the model as an input
        x_pred = np.append(x_pred[:, 1:, :], y_pred.reshape(1, 1, 9), axis=1)

        # generate the next forecast
        y_pred = model.predict(x_pred)

        # save the forecast
        y_future.append(y_pred)
    return "The mean temperature on " + str(date) + " is %.2f" % y_future[-1] + "\N{DEGREE SIGN}C."

def arima_predict(date):
    pass

