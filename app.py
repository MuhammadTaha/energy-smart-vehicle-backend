from flask import Flask, request, jsonify
from flask_cors import CORS
from chargingOptimization import ChargingOptimizations
# from combinedOptimization import getCombinedOptimizationResults, assignValuesCombined, routeOptimization
import json

app = Flask(__name__)
CORS(app)

chargeOpt = ChargingOptimizations()

chargeOpt.loadModel()

@app.route('/route-based', methods=['GET', 'POST'])
def route_based():
    # print(getCombinedOptimizationResults())
    #assignRouteRequest(json.loads(request.data))
    # jsonData = json.loads(request.data)
    # arrivalTime,drivingWeight,waitingWeight,energyWeight = assignValuesCombined(jsonData)
    # routeOptimization(arrivalTime,drivingWeight,waitingWeight,energyWeight)
    # print(getCombinedOptimizationResults())
    return request.data
    # return getCombinedOptimizationResults()

@app.route('/charging-based', methods=['GET', 'POST'])
def charging_based():
    jsonData = json.loads(request.data)
    arrivalTime,departureTime,charRate,SOC_per_beg = chargeOpt.assignChargingRequest(jsonData)
    chargeOpt.charOptimization(arrivalTime,departureTime,charRate,SOC_per_beg)
    return chargeOpt.getChargingOptimizationResults()
#    return request.data
    # return getChargingOptimizationResults()

if __name__ == '__main__':
    app.run()