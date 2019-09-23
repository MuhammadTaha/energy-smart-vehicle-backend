from flask import Flask, request, jsonify
from flask_cors import CORS
from chargingOptimization import getChargingOptimizationResults
from combinedOptimization import getCombinedOptimizationResults

app = Flask(__name__)
CORS(app)

@app.route('/route-based', methods=['GET', 'POST'])
def route_based():
    # print(getCombinedOptimizationResults())
    return getCombinedOptimizationResults()

@app.route('/charging-based', methods=['GET', 'POST'])
def charging_based():
    # print(getChargingOptimizationResults())
    return getChargingOptimizationResults()

if __name__ == '__main__':
    app.run()