from flask import Flask, request, jsonify
from flask_cors import CORS
from chargingOptimization import getChargingOptimizationResults, assignChargingRequest
from combinedOptimization import getCombinedOptimizationResults, assignValuesCombined
import json

app = Flask(__name__)
CORS(app)

@app.route('/route-based', methods=['GET', 'POST'])
def route_based():
    # print(getCombinedOptimizationResults())
    assignChargingRequest(json.loads(request.data))
    return request.data
    # return getCombinedOptimizationResults()

@app.route('/charging-based', methods=['GET', 'POST'])
def charging_based():
    print(getChargingOptimizationResults())
    assignValuesCombined(json.loads(request.data))
    return request.data
    # return getChargingOptimizationResults()

if __name__ == '__main__':
    app.run()