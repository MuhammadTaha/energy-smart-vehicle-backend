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

def assignValuesCombined(data):
    arrivalTime = '2019-09-24T16:00'
    drivingWeight = 8
    waitingWeight = 8
    energyWeight = 8
    return arrivalTime,drivingWeight,waitingWeight,energyWeight

### Solution
def printSolution(hour,minute,drivingWeight,waitingWeight,energyWeight,recommendedRoute):
    print('\n------------------------------------------------------------')
    print('\nDesired arrival time: %s : %s' % (hour,minute))
    print("\nWeight driving time: %s  Weight waiting time: %s  Weight energy consumption: %s" % (drivingWeight,waitingWeight,energyWeight))
    print('\n------------------------------------------------------------')
    print('\nRecommended Route: %s' % (recommendedRoute.route))
    print('\nDeparture Time: %s : %s' % (recommendedRoute.dep_H, recommendedRoute.dep_M))
    print('\nArrival Time: %g : %g' % (recommendedRoute.arr_H, recommendedRoute.arr_M))
    print('\nDriving Time: %g minutes' % (recommendedRoute.dri_T))
#     print('\nShortest possible driving time: %g minutes' % (m.ObjVal))
    print('\nWaiting Time: %g minutes' % (recommendedRoute.wai_T))
    print('\nEnergy Price: %g €' % (recommendedRoute.eneCost))

def routeOptimization(arrivalTime,drivingWeight,waitingWeight,energyWeight):
    """
    Values to get from forecasting:
    Temperature for the relevant timeframe
    Energyprices
    Energy consumption of the windshieldwipers
    Energy consumption of the lights
    """
#     temperatureModel = load_model('./Forecasting Models/temperatureModel.h5')
#     precipitationModel = load_model('./Forecasting Models/precipitationModel.h5')
#     cityModel = load_model('./Forecasting Models/cityModel.h5')
#     highwayModel = load_model('./Forecasting Models/highwayModel.h5')
    
    arrivalDT = datetime.strptime(arrivalTime, '%Y-%m-%dT%H:%M')
    departureDT = arrivalDT - timedelta(hours=2)
    month = arrivalDT.month
    day = arrivalDT.day 
    hour = arrivalDT.hour
    minute = arrivalDT.minute
    timeArray = []
    
    for i in range(departureDT.hour,arrivalDT.hour+1):
        array = [month,day,i,0,0,0,0,0,0,0]
        if (arrivalDT.isoweekday()<7):
            array[arrivalDT.isoweekday()+4] = 1
        elif (arrivalDT.isoweekday() == 7):
            array[4] = 1
        timeArray.append(array)
        
    timeDF = pd.DataFrame(timeArray)
    timeRequest = np.array(timeDF)
    timeRequest = np.reshape(timeRequest, (timeRequest.shape[0], 1, timeRequest.shape[1]))
    cityTime = cityModel.predict(timeRequest)
    cityTime = (cityTime/60).round(decimals=0)
    highwayTime = highwayModel.predict(timeRequest)
    highwayTime = (highwayTime/60).round(decimals=0)

    departureTimes = []
    for i in range(24):
        m = i*5
        departureTimes.append(departureDT + timedelta(minutes = m))
    
    
    arrivalTimesCity = []
    for i in range(24):
        dep = departureTimes[i]
        if dep.hour < hour:
            pred = cityTime[0][0]
            arr = dep + timedelta(minutes=int(pred))
            wait = (arrivalDT - arr).total_seconds() / 60
            if arr.hour < hour:
                arrivalTimesCity.append(['City',dep.hour,dep.minute,arr.hour,arr.minute,pred,wait])
            elif (arr.hour == hour) & (arr.minute <= minute):
                arrivalTimesCity.append(['City',dep.hour,dep.minute,arr.hour,arr.minute,pred,wait])
            else:
                continue
        else:
            pred = cityTime[2][0]
            arr = dep + timedelta(minutes=int(pred))
            wait = (arrivalDT - arr).total_seconds() / 60
            if arr.hour < arrivalDT.hour:
                arrivalTimesCity.append(['City',dep.hour,dep.minute,arr.hour,arr.minute,pred,wait])
            elif (arr.hour == hour) & (arr.minute <= minute):
                arrivalTimesCity.append(['City',dep.hour,dep.minute,arr.hour,arr.minute,pred,wait])
            else:
                continue
    arrCityDF = pd.DataFrame(arrivalTimesCity,columns=['route','dep_H','dep_M','arr_H','arr_M','dri_T','wai_T'])
    
    arrivalTimesHighway = []
    for i in range(24):
        dep = departureTimes[i]
        if dep.hour < hour:
            pred = highwayTime[0][0]
            arr = dep + timedelta(minutes=int(pred))
            wait = (arrivalDT - arr).total_seconds() / 60
            if arr.hour < hour:
                arrivalTimesHighway.append(['Highway',dep.hour,dep.minute,arr.hour,arr.minute,pred,wait])
            elif (arr.hour == hour) & (arr.minute <= minute):
                arrivalTimesHighway.append(['Highway',dep.hour,dep.minute,arr.hour,arr.minute,pred,wait])
            else:
                continue
        else:
            pred = highwayTime[2][0]
            arr = dep + timedelta(minutes=int(pred))
            wait = (arrivalDT - arr).total_seconds() / 60
            if arr.hour < arrivalDT.hour:
                arrivalTimesHighway.append(['Highway',dep.hour,dep.minute,arr.hour,arr.minute,pred,wait])
            elif (arr.hour == hour) & (arr.minute <= minute):
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
    precipitation = precipitationModel.predict(energyRequest)
    precipitation = precipitation > 0.5
    temperature = temperatureModel.predict(energyRequest)
    
    whiperCons = getWhiperConsumption(precipitation,arrDF)
    lightCons = getLightConsumption(arrDF,hour)
    acCons = getAcConsumption(temperature,arrDF)
    
    energyCons = []
    energyCost = []
    for i in range(len(whiperCons)):
        unitPrice = 0.2774
        
        EUC = 4551 + whiperCons[i] + lightCons[i]
        EL = 4181
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
    
    costMatrix = np.matrix((drivingCost,waitingCost,energyCost)).T
    costDF = pd.DataFrame(costMatrix, columns = ['driCost','waiCost','eneCost'])
    costDF['eneCost'] = costDF.eneCost.astype(float)
    
    DF = pd.merge(arrDF,costDF,on=arrDF.index)    
    scenarioScores = DF.driCost * drivingWeight + DF.waiCost * waitingWeight + DF.eneCost * energyWeight
    scores = list(scenarioScores)
    best = scores.index(min(scores))
    recommendedRoute = DF.iloc[best]
    print(recommendedRoute)
    
    m = Model()
    
    choice = list()
    for x in range(len(DF)):
        choice.append(m.addVar(vtype=GRB.BINARY))
        
    temp = scores
    
    m.update()
    
    m.addConstr(quicksum(choice) == 1)
    
    m.setObjective(quicksum(temp[x] * choice[x] for x in range(len(temp))), GRB.MINIMIZE)
    
    m.optimize()
    
    printSolution(hour,minute,drivingWeight,waitingWeight,energyWeight,recommendedRoute)
    
def getWhiperConsumption(precipitation, arrDF):
    whiperConsumptionPerH = 0.06
    lightConsumptionPerH = 0.054
    
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

def getLightConsumption(arrDF, hour):
    lightConsumption = []
    lightConsumptionHighwayDay = 20.89
    lightConsumptionHighwayNight = 53.35
    lightConsumptionCityDay = 13.67
    lightConsumptionCityNight = 34.92
    
    sunData = pd.read_csv('sun.csv',index_col=[0])
    sunDataRel = sunData[(sunData.month == month) & (sunData.day == day)]
    
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
            
def getAcConsumption(temperature,arrDF):
    acConsumption = []
    
    dH = np.array(arrDF.dep_H) 
    aH = np.array(arrDF.arr_H)
    hs = np.unique(np.concatenate((dH,aH)))
    
    for i in range(len(arrDF)):
        t = temperature[np.where(hs == arrDF.dep_H[i])[0][0]]
        EC = -0.000002*math.pow(t,5)+0.0003*math.pow(t,4)-0.0214*math.pow(t,3)+0.5865*math.pow(t,2)-9.3772*t+290.39
        acConsumption.append(EC)
    return acConsumption        

def getCombinedOptimizationResults():
    results = collections.defaultdict();
    results["desiredArrivalTime"] = desiredArrivalTimeHour +":"+ desiredArrivalTimeMinute
    results["recommendedRoute"] = recommendedRoute.Route
    results["departureTIme"] = recommendedRoute.departureTimeHour +":"+ recommendedRoute.departureTimeMinute
    results["arrivalTime"] = recommendedRoute.arrivalTimeHour +":"+recommendedRoute.arrivalTimeMinute
    results["drivingTime:"] = recommendedRoute.travelTime
    results["waitingTime:"] = recommendedRoute.waitingTime
    results["energyPrice:"] = recommendedRoute.energyPrice

    return json.dumps(results)