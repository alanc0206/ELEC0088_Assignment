from google.cloud import dialogflow
from google.api_core.exceptions import InvalidArgument
import pandas as pd
import os

# Define API credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'elec0088-chatbot-frqp-e1e8b6b5eff0.json'

# Define agent parameters
DIALOGFLOW_PROJECT_ID = 'elec0088-chatbot-frqp'
DIALOGFLOW_LANGUAGE_CODE = 'en'


class BotApi:

    # Constructor that starts a new session from uuid
    def __init__(self, session_id):
        self.session_client = dialogflow.SessionsClient()
        self.session = self.session_client.session_path(DIALOGFLOW_PROJECT_ID, session_id)

    # Fulfillment method
    def fulfill(self, text_to_bot):
        # Get text from client and rest api server
        text_input = dialogflow.TextInput(text=text_to_bot, language_code=DIALOGFLOW_LANGUAGE_CODE)
        query_input = dialogflow.QueryInput(text=text_input)
        try:
            # Get response from dialogflow agent
            response = self.session_client.detect_intent(session=self.session, query_input=query_input)
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
    pass

def arima_predict(date):
    pass

