from gurobipy import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import load_model
from datetime import datetime
from datetime import timedelta
from operator import itemgetter 
import math
import time
import collections, json

class RouteOptimizations:
    def assignValuesCombined(self,data):
        arrivalTime = data['arrivalTime']
        drivingWeight = data['minDrivingValue']
        waitingWeight = data['minWaitingValue']
        energyWeight = data['minEnergyValue']
        return arrivalTime,drivingWeight,waitingWeight,energyWeight

    def loadModel(self):
        self.temperatureModel = load_model('.\Forecasting Models\\temperatureModel.h5')
        self.precipitationModel = load_model('.\Forecasting Models\precipitationModel.h5')
        self.cityModel = load_model('.\Forecasting Models\cityModel.h5')
        self.highwayModel = load_model('.\Forecasting Models\highwayModel.h5')
        self.temperatureModel._make_predict_function()
        self.precipitationModel._make_predict_function()
        self.cityModel._make_predict_function()
        self.highwayModel._make_predict_function()
    
    recommendedRoute = pd.Series()
    minute = 0
    hour = 0
    month = 0
    day = 0
    
    def routeOptimization(self,arrivalTime,drivingWeight,waitingWeight,energyWeight):
        """
        Values to get from forecasting:
        Temperature for the relevant timeframe
        Energyprices
        Energy consumption of the windshieldwipers
        Energy consumption of the lights
        """
        arrivalDT = datetime.strptime(arrivalTime, '%Y-%m-%dT%H:%M')
        departureDT = arrivalDT - timedelta(hours=2)
        self.month = arrivalDT.month
        self.day = arrivalDT.day 
        self.hour = arrivalDT.hour
        self.minute = arrivalDT.minute
        timeArray = []
        for i in range(departureDT.hour,arrivalDT.hour+1):
            array = [self.month,self.day,i,0,0,0,0,0,0,0]
            if (arrivalDT.isoweekday()<7):
                array[arrivalDT.isoweekday()+4] = 1
            elif (arrivalDT.isoweekday() == 7):
                array[4] = 1
            timeArray.append(array)    
        timeDF = pd.DataFrame(timeArray)
        timeRequest = np.array(timeDF)
        timeRequest = np.reshape(timeRequest, (timeRequest.shape[0], 1, timeRequest.shape[1]))
        cityTime = self.cityModel.predict(timeRequest)
        cityTime = (cityTime/60).round(decimals=0)
        highwayTime = self.highwayModel.predict(timeRequest)
        highwayTime = (highwayTime/60).round(decimals=0)
        departureTimes = []
        for i in range(24):
            m = i*5
            departureTimes.append(departureDT + timedelta(minutes = m))
        
        arrivalTimesCity = []
        for i in range(24):
            dep = departureTimes[i]
            if dep.hour < self.hour:
                pred = cityTime[0][0]
                arr = dep + timedelta(minutes=int(pred))
                wait = (arrivalDT - arr).total_seconds() / 60
                if arr.hour < self.hour:
                    arrivalTimesCity.append(['City',dep.hour,dep.minute,arr.hour,arr.minute,pred,wait])
                elif (arr.hour == self.hour) & (arr.minute <=self. minute):
                    arrivalTimesCity.append(['City',dep.hour,dep.minute,arr.hour,arr.minute,pred,wait])
                else:
                    continue
            else:
                pred = cityTime[2][0]
                arr = dep + timedelta(minutes=int(pred))
                wait = (arrivalDT - arr).total_seconds() / 60
                if arr.hour < arrivalDT.hour:
                    arrivalTimesCity.append(['City',dep.hour,dep.minute,arr.hour,arr.minute,pred,wait])
                elif (arr.hour == self.hour) & (arr.minute <= self.minute):
                    arrivalTimesCity.append(['City',dep.hour,dep.minute,arr.hour,arr.minute,pred,wait])
                else:
                    continue
        arrCityDF = pd.DataFrame(arrivalTimesCity,columns=['route','dep_H','dep_M','arr_H','arr_M','dri_T','wai_T'])
        arrivalTimesHighway = []
        for i in range(24):
            dep = departureTimes[i]
            if dep.hour < self.hour:
                pred = highwayTime[0][0]
                arr = dep + timedelta(minutes=int(pred))
                wait = (arrivalDT - arr).total_seconds() / 60
                if arr.hour < self.hour:
                    arrivalTimesHighway.append(['Highway',dep.hour,dep.minute,arr.hour,arr.minute,pred,wait])
                elif (arr.hour == self.hour) & (arr.minute <= self.minute):
                    arrivalTimesHighway.append(['Highway',dep.hour,dep.minute,arr.hour,arr.minute,pred,wait])
                else:
                    continue
            else:
                pred = highwayTime[2][0]
                arr = dep + timedelta(minutes=int(pred))
                wait = (arrivalDT - arr).total_seconds() / 60
                if arr.hour < self.hour:
                    arrivalTimesHighway.append(['Highway',dep.hour,dep.minute,arr.hour,arr.minute,pred,wait])
                elif (arr.hour == self.hour) & (arr.minute <= self.minute):
                    arrivalTimesHighway.append(['Highway',dep.hour,dep.minute,arr.hour,arr.minute,pred,wait])
                else:
                    continue        
        arrHighwayDF = pd.DataFrame(arrivalTimesHighway,columns=['route','dep_H','dep_M','arr_H','arr_M','dri_T','wai_T'])
        arrDF = arrCityDF.append(arrHighwayDF)
        arrDF.columns = ['route','dep_H','dep_M','arr_H','arr_M','dri_T','wai_T'] 
        arrDF.index = range(len(arrDF))
    
        energyDF = timeDF.iloc[:,:3]
        energyRequest = np.array(energyDF)
        energyRequest = np.reshape(energyRequest, (energyRequest.shape[0], 1, energyRequest.shape[1]))
        precipitation = self.precipitationModel.predict(energyRequest)
        precipitation = precipitation > 0.5
        temperature = self.temperatureModel.predict(energyRequest)
        whiperCons = self.getWhiperConsumption(precipitation,arrDF)
        lightCons = self.getLightConsumption(arrDF,self.hour)
        acCons = self.getAcConsumption(temperature,arrDF)
        energyCons = []
        energyCost = []
        for i in range(len(whiperCons)):
            unitPrice = 0.2774
            if (arrDF.route[i] == 'City'):
                EUC = whiperCons[i] + lightCons[i]
                EL = 1458
                total = EUC + acCons[i] + EL
                cost = total * unitPrice / 1000
                energyCons.append(total)
                energyCost.append(cost)
            else:
                EUC = whiperCons[i] + lightCons[i]
                EL = 3025
                total = EUC + acCons[i] + EL
                cost = total * unitPrice / 1000
                energyCons.append(total)
                energyCost.append(cost)
    
        pricePerHourDriving = 7.48
        pricePerMinuteDriving = pricePerHourDriving/60
        pricePerHourWaiting = 1.5 * pricePerHourDriving
        pricePerMinuteWaiting = pricePerHourWaiting/60
        
        waitingCost = []
        drivingCost = []
        for i in range(len(arrDF)):
            drivingCost.append(arrDF.dri_T[i] * pricePerMinuteDriving)
            waitingCost.append(arrDF.wai_T[i] * pricePerMinuteWaiting)
        
        costMatrix = np.matrix((energyCons,drivingCost,waitingCost,energyCost)).T
        costDF = pd.DataFrame(costMatrix, columns = ['eneCons','driCost','waiCost','eneCost'])
        costDF['eneCost'] = costDF.eneCost.astype(float)
        costDF['eneCons'] = costDF.eneCons.astype(float)

        DF = pd.merge(arrDF,costDF,on=arrDF.index)

        a = DF.driCost.values * int(drivingWeight)
        b = DF.waiCost.values * int(waitingWeight)
        c = DF.eneCost.values * int(energyWeight)

        scenarioScores = a + b + c

        # scenarioScores = DF.driCost * drivingWeight + DF.waiCost * waitingWeight + DF.eneCost * energyWeight
        scores = list(scenarioScores)
        best = scores.index(min(scores))
        self.recommendedRoute = DF.iloc[best]
        
        m = Model()
        
        choice = list()
        for x in range(len(DF)):
            choice.append(m.addVar(vtype=GRB.BINARY))
            
        temp = scores
        
        m.update()
        
        m.addConstr(quicksum(choice) == 1)
        
        m.setObjective(quicksum(temp[x] * choice[x] for x in range(len(temp))), GRB.MINIMIZE)
        
        m.optimize()
        
        # printSolution(self.hour,self.minute,drivingWeight,waitingWeight,energyWeight,recommendedRoute)
        
    def getWhiperConsumption(self,precipitation, arrDF):
        whiperConsumptionPerH = 0.06
        
        whiperConsumption = []
        for i in range(len(arrDF)):
            if sum(precipitation) == 0:
                whiperConsumption.append(0)
            elif arrDF.iloc[i,1] == arrDF.iloc[i,3]:
                if precipitation[0] == False:
                    whiperConsumption.append(0)
                else:
                    cons = arrDF.iloc[i,5]/60 * whiperConsumptionPerH
                    whiperConsumption.append(cons)
            else:
                if precipitation[1] == False:
                    whiperConsumption.append(0)
                else:
                    cons1 = (60-arrDF.iloc[i,2])/60 * whiperConsumptionPerH
                    cons2 = (arrDF.iloc[i,4]/60) * whiperConsumptionPerH
                    whiperConsumption.append(cons1+cons2)
        
        return whiperConsumption
    
    def getLightConsumption(self,arrDF, hour):
        lightConsumption = []
        lightConsumptionHighwayDay = 20.89
        lightConsumptionHighwayNight = 53.35
        lightConsumptionCityDay = 13.67
        lightConsumptionCityNight = 34.92
        
        sunData = pd.read_csv('sun.csv',index_col=[0])
        sunDataRel = sunData[(sunData.month == self.month) & (sunData.day == self.day)]
        
        srH = sunDataRel.srH.item()
        ssH = sunDataRel.ssH.item()
        night = []
        for i in range(hour - 2, hour + 1):
            if (i < srH) | (i > ssH):
                night.append(1)
            else:
                night.append(0)
        
        dH = np.array(arrDF.dep_H) 
        aH = np.array(arrDF.arr_H)
        hs = np.unique(np.concatenate((dH,aH)))
        for i in range(len(arrDF)):
            if (night[np.where(hs == arrDF.dep_H[i])[0][0]] == 1) & (arrDF.route[i] == 'Highway'):
                lightConsumption.append(lightConsumptionHighwayNight)
            elif (night[np.where(hs == arrDF.dep_H[i])[0][0]] == 0) & (arrDF.route[i] == 'Highway'):
                lightConsumption.append(lightConsumptionHighwayDay)
            elif (night[np.where(hs == arrDF.dep_H[i])[0][0]] == 1) & (arrDF.route[i] == 'City'):
                lightConsumption.append(lightConsumptionCityNight)
            else:
                lightConsumption.append(lightConsumptionCityDay)
                
        return lightConsumption
                
    def getAcConsumption(self,temperature,arrDF):
        acConsumption = []
        
        dH = np.array(arrDF.dep_H) 
        aH = np.array(arrDF.arr_H)
        hs = np.unique(np.concatenate((dH,aH)))
        
        for i in range(len(arrDF)):
            t = temperature[np.where(hs == arrDF.dep_H[i])[0][0]]
            EC = -0.000002*math.pow(t,5)+0.0003*math.pow(t,4)-0.0214*math.pow(t,3)+0.5865*math.pow(t,2)-9.3772*t+290.39
            acConsumption.append(EC)
        return acConsumption        
    
    def getCombinedOptimizationResults(self):
        results = collections.defaultdict()
        results["desiredArrivalTime"] = str(self.hour) + ":" + str(self.minute)
        results["recommendedRoute"] = self.recommendedRoute.route
        results["departureTIme"] = str(self.recommendedRoute.dep_H) +":"+ str(self.recommendedRoute.dep_M)
        results["arrivalTime"] = str(self.recommendedRoute.arr_H) +":"+ str(self.recommendedRoute.arr_M)
        results["drivingTime"] = self.recommendedRoute.dri_T
        results["waitingTime"] = self.recommendedRoute.wai_T
        results["energyPrice"] = round(self.recommendedRoute.eneCost,2)
        results["energyConsumed"] = round(self.recommendedRoute.eneCons/1000 ,2)
        results["waitingCost"] = round(self.recommendedRoute.waiCost ,2)
        results["drivingCost"] = round(self.recommendedRoute.driCost ,2)

        return json.dumps(results)