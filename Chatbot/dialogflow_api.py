from google.cloud import dialogflow
from google.api_core.exceptions import InvalidArgument
import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'elec0088-chatbot-frqp-69ca19fc6944.json'

DIALOGFLOW_PROJECT_ID = 'elec0088-chatbot-frqp'
DIALOGFLOW_LANGUAGE_CODE = 'en'


class BotApi:

    def __init__(self, session_id):
        self.session_client = dialogflow.SessionsClient()
        self.session = self.session_client.session_path(DIALOGFLOW_PROJECT_ID, session_id)

    def fulfill(self, text_to_bot):
        text_input = dialogflow.TextInput(text=text_to_bot, language_code=DIALOGFLOW_LANGUAGE_CODE)
        query_input = dialogflow.QueryInput(text=text_input)
        try:
            response = self.session_client.detect_intent(session=self.session, query_input=query_input)
        except InvalidArgument:
            return 'Invalid argument'

        return response.query_result.fulfillment_text




