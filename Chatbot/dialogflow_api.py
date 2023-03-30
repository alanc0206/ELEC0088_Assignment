from google.cloud import dialogflow
from google.api_core.exceptions import InvalidArgument
import pandas as pd
import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'elec0088-chatbot-frqp-69ca19fc6944.json'

DIALOGFLOW_PROJECT_ID = 'elec0088-chatbot-frqp'
DIALOGFLOW_LANGUAGE_CODE = 'en'

df = pd.read_csv('prophet weather forecast.csv')
df2 = df.drop(df.columns[0], axis=1)
df2.index = df.ds
df2 = df2.drop(df2.columns[0], axis=1)


class BotApi:

    def __init__(self, session_id):
        self.session_client = dialogflow.SessionsClient()
        self.session = self.session_client.session_path(DIALOGFLOW_PROJECT_ID, session_id)

    def fulfill(self, text_to_bot):
        text_input = dialogflow.TextInput(text=text_to_bot, language_code=DIALOGFLOW_LANGUAGE_CODE)
        query_input = dialogflow.QueryInput(text=text_input)
        try:
            response = self.session_client.detect_intent(session=self.session, query_input=query_input)
            if response.query_result.fulfillment_text.startswith('Find information'):
                date = response.query_result.fulfillment_text.split('date')[1]
                try:
                    temps = df2.loc[date]
                    return "The predicted mean temperature on " + str(date) + " is %.2f" % temps.iloc[0] \
                        + "\N{DEGREE SIGN}C with a variation of \u00B1%.2f" % (temps.iloc[2] - temps.iloc[1]) \
                        + "\N{DEGREE SIGN}C. Would you like to continue?"
                except KeyError:
                    return "That date is invalid or too far away to predict. Continue?"
        except InvalidArgument:
            return 'Invalid argument'

        return response.query_result.fulfillment_text
