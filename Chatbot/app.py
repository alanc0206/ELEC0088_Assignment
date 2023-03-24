from flask import Flask, request, make_response
import json
import pandas as pd
from pyngrok import ngrok


forecast_app = Flask(__name__)

# Load the trained pickled model here
#model = pickle.load(open('', 'rb'))

# Open ngrok tunnel
#http_tunnel = ngrok.connect(5000)

df = pd.read_csv('prophet weather forecast.csv')
df2 = df.drop(df.columns[0], axis=1)
df2.index = df.ds
df2 = df2.drop(df2.columns[0], axis=1)

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
    date = parameters.get("Date")
    date = str(date)
    date = date.split('T')[0]

    intent = result.get("intent").get('displayName')

    if (intent == 'Predict'):

        temps = df2.loc[date]
        fulfillmentText = "The predicted mean temperature on " + str(date) + " is %.2f" % temps.iloc[0] \
                          + "\N{DEGREE SIGN}C with a variation of \u00B1%.2f" % (temps.iloc[2] - temps.iloc[1]) \
                          + "\N{DEGREE SIGN}C. Would you like to continue?"

        return {
            "fulfillmentText": fulfillmentText
        }

if __name__ == '__main__':
    forecast_app.run()
