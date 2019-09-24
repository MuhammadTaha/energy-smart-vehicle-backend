import collections
import json
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gurobipy import *
from keras.models import load_model, Sequential



class ChargingOptimizations:

    def assignChargingRequest(self,data):
        arrivalTime = data['arrivalTime']
        departureTime = data['departureTime']
        charRate = float(data['powerSelection'])
        SOC_per_beg = float(data['chargingStatus'])/100
    #    self.SOC_per_end = data['']
        return arrivalTime,departureTime,charRate,SOC_per_beg

    # def printSolution():
    #     if mDynamic.status == GRB.Status.OPTIMAL:
    #         print('\n----------------------------------------------------------------\n')
    #         print('\nTotal energy transferred: %g kW' % sumTransferedEnergy)
    #         print('\nMinimal Charging Cost: %g ct' % mDynamic.objVal)
    #         print('\nTotal used charging time: %g h' % realChargingDuration)
    #         print('\nEnergy transferred during each hour:')
    #         chargex = mDynamic.getAttr('x', chargingRate)
    #         for t in np.arange(8):
    #             print('\n%s: %g kW - %f ct/kW' % (t, chargex[t], chargingPrices[t][0]))
    #         print('\n----------------------------------------------------------------\n')
    #         print('\nCharging costs using retail prices: %s ct' % (retailChargingCosts))
    #     else:
    #         print('No solution')


    mDynamic = Model()
    chargingRate = list()
    sumTransferedEnergy = 0
    realChargingDuration = 0.0
    retailChargingCosts = 0.0
    chargingPrices = []
    dynamicObjVal = 0

    def loadModel(self):
        self.model = load_model('.\Forecasting Models\energyModel.h5')
        self.model._make_predict_function()

    def charOptimization(self,arrivalTime,departureTime,charRate,SoC_per_beg):

        arrivalDT = datetime.strptime(arrivalTime, '%Y-%m-%dT%H:%M')
        departureDT = datetime.strptime(departureTime, '%Y-%m-%dT%H:%M')
        month = arrivalDT.month
        day = arrivalDT.day
        delta = departureDT - arrivalDT
        chargingDuration = int(delta.total_seconds()/3600)

        valueArray = []

        for i in range(arrivalDT.hour,departureDT.hour):
            array = [month,day,i,0,0,0,0,0,0,0]
            if (arrivalDT.isoweekday()<7):
                array[arrivalDT.isoweekday()+4] = 1
            elif (arrivalDT.isoweekday() == 7):
                array[4] = 1
            valueArray.append(array)


        valueDF = pd.DataFrame(valueArray)
        request = np.array(valueDF)
        request = np.reshape(request, (request.shape[0], 1, request.shape[1]))
        varPrices = self.model.predict(request)
        varPrices = varPrices/10



        # Car specs: BMW I3
        battCap = 22

        #Scenario values
        #startTime = 8
        #endTime = 16
        #SoC_per_beg = 0.1
        SoC_per_end = 0.8
        #maxCharRateStation = 11

        #charRate = min(maxCharRateCar,maxCharRateStation)

        ### Built Model

        ### Energy needed
        SoC_beg = battCap * SoC_per_beg
        SoC_end = battCap * SoC_per_end
        self.sumTransferedEnergy = float(SoC_end) - float(SoC_beg)

        '''
        Dynamic Price data taken from https://www.awattar.de/ (09.07.2019)
        Price consists of fixed price (19.61 ct/kWh) + energy stock market price
        Basic montly fee: 13.33€
        
        '''

        fixedPrice = 19.61
        #varPrices = [3.76, 3.59, 3.55, 3.5, 3.56, 3.63,
        #          4.3, 5.23, 5.24, 4.89, 4.54, 4.53,
        #          4.41, 4.18, 4.14, 4.07, 4.13, 4.37,
        #          4.8, 5.59, 5.7, 5.47, 5.22, 4.36]
        prices = [(v + fixedPrice).tolist() for v in varPrices]


        '''
        Retail price taken from https://www.vattenfall.de/stromtarife/strom-natur24 (Access: 14.07.2019)
        Only sustainable power supply for better comparison
        Price per kWh: 27.74 ct/kWh
        Basic montly fee: 10.30 €
        '''
        retailPrice = 27.74

        self.retailChargingCosts = self.sumTransferedEnergy * retailPrice

        ### Charging time & prices
        #chargingDuration = endTime - startTime
        chargingInterval = np.arange(chargingDuration)
        self.realChargingDuration = self.sumTransferedEnergy / float(charRate)
        #chargingPrices = prices[startTime:endTime]
        self.chargingPrices = np.round(prices,2)


        ### Variables
        for t in chargingInterval:
            self.chargingRate.append(self.mDynamic.addVar(vtype='C', lb=0.0, ub=float(charRate)))

        ### Constraints
        self.mDynamic.addConstr((quicksum(self.chargingRate) == self.sumTransferedEnergy))

        self.mDynamic.update()

        ### Objective Function
        self.mDynamic.setObjective(quicksum(self.chargingPrices[t][0] * self.chargingRate[t] for t in chargingInterval), GRB.MINIMIZE)

        self.mDynamic.optimize()
    #     printSolution()

        # chargingActivity = mDynamic.getAttr('x', chargingRate)

        ### time_slots = ['0-1','1-2','2-3','3-4','4-5','5-6','6-7','7-8']
        # timeSlots = [str(chargingPrices[x][0]) for x in range(chargingDuration)]
        # plt.bar(timeSlots,chargingActivity)
        # plt.ylabel('TRANSFERRED ENERGY (kW)')
        # plt.xlabel('ENERGY PRICE (ct/kW)')
        # plt.title('CHARGING ACTIVITY')

        # plt.show()



    def getChargingOptimizationResults(self):
        results = {}
        resultsChargingPrice = collections.defaultdict()
        results["sumTransferedEnergy"] = self.sumTransferedEnergy
        results["minimumChargingCost"] = round(self.mDynamic.ObjVal,2)
        results["UsedChargingTime"] = round(self.realChargingDuration,2)
        results["RetailChargingCost"] = self.retailChargingCosts
        chargex = self.mDynamic.getAttr('x', self.chargingRate)
        results["ChargingPrices"] = {}
        for t in np.arange(8):
            resultsChargingPrice.setdefault(round(chargex[t],2), []).append(self.chargingPrices[t][0])

        results["ChargingPrices"] = resultsChargingPrice
        return json.dumps(results)