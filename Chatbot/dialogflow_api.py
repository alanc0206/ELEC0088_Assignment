## This application module acts as a Google cloud API client for the Dialogflow agent
## and calls prediction functions using relevant forecasting models.
# Import modules
from google.cloud import dialogflow_v2 as dialogflow
from google.api_core.exceptions import InvalidArgument
import pandas as pd
import chatbot_lstm_forecast as lstm
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
                    # Call prediction method (lstm or prophet)
                    return lstm.forecast(date)
                except KeyError:
                    # Catch exception if date is invalid
                    return "That date is invalid or too far away to predict. Continue?"
                except ValueError:
                    # Catch exception if date is invalid
                    return "That that is invalid. Please enter a date after 2020-12-31. Continue?"
        except InvalidArgument:
            return 'Invalid argument'

        # Return fulfillment text
        return response.query_result.fulfillment_text


# Function for prophet prediction
def prophet_predict(date):
    # Load predicted prophet values from csv file
    df = pd.read_csv('prophet weather forecast.csv')
    df2 = df.drop(df.columns[0], axis=1)
    df2.index = df.ds
    df2 = df2.drop(df2.columns[0], axis=1)
    temps = df2.loc[date]
    # Return predicted values from csv file
    return "The predicted mean temperature on " + str(date) + " (YYYY-MM-DD) is %.2f" % temps.iloc[0] \
        + "\N{DEGREE SIGN}C with a maximum temperature of %.2f" % temps.iloc[2] + \
        "\N{DEGREE SIGN}C and a minimum temperature of %.2f" % temps.iloc[1] \
        + "\N{DEGREE SIGN}C. Would you like to continue?"



