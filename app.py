from flask import Flask, request, make_response
import json
from pyngrok import ngrok
import pickle


forecast_app = Flask(__name__)

# Load the trained pickled model here
#model = pickle.load(open('', 'rb'))

@forecast_app.route('/')
def hello():
    return 'Hello world'

# Getting and sending responses to dialogflow
@forecast_app.route('/webhook', methods=['POST'])
def webhook():

    req = request.get_json(silent=True, force=True)
    res = processRequest(req)
    res = json.dumps(res)

    r = make_response(res)
    r.headers['Content-Type'] = 'application/json'
    return r  # Final Response sent to DialogFlow


def processRequest(req):  # This method processes the incoming request

    result = req.get("queryResult")
    parameters = result.get("parameters")
    year = parameters.get("yearinput")

    intent = result.get("intent").get('displayName')

    # if (intent == 'DataYes'):
    #     prediction = model.predict([[year]])
    #     output = round(prediction[0], 2)
    #
    #     fulfillmentText = "The Per capita income for this year is:  {} !".format(output)
    #
    #     return {
    #         "fulfillmentText": fulfillmentText
    #     }

if __name__ == '__main__':
    forecast_app.run()