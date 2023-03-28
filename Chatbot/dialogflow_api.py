import os
from google.cloud import dialogflow
from google.api_core.exceptions import InvalidArgument

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'elec0088-chatbot-frqp-69ca19fc6944.json'

DIALOGFLOW_PROJECT_ID = 'elec0088-chatbot-frqp'
DIALOGFLOW_LANGUAGE_CODE = 'en'
SESSION_ID = 'me'


def initiate():
    text_to_be_analyzed = "Hi"

    session_client = dialogflow.SessionsClient()
    session = session_client.session_path(DIALOGFLOW_PROJECT_ID, SESSION_ID)
    text_input = dialogflow.TextInput(text=text_to_be_analyzed, language_code=DIALOGFLOW_LANGUAGE_CODE)
    query_input = dialogflow.QueryInput(text=text_input)
    try:
        response = session_client.detect_intent(session=session, query_input=query_input)
    except InvalidArgument:
        raise

    fulfillment = response.query_result.fulfillment_text

    return fulfillment
