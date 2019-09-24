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
def printSolution():
    print('\n------------------------------------------------------------')
    print('\nRecommendation for %s (%s.%s.%s):' % (recommendedRoute.Weekday,recommendedRoute.Day,recommendedRoute.Month,recommendedRoute.Year))
    print('\nDesired arrival time: %s : %s' % (desiredArrivalTimeHour,desiredArrivalTimeMinute))
    print("\nWeight driving time: %s  Weight waiting time: %s  Weight energy consumption: %s" % (drivingWeight,waitingWeight,energyWeight))
    print('\n------------------------------------------------------------')
    print('\nRecommended Route: %s' % (recommendedRoute.Route))
    print('\nDeparture Time: %s : %s' % (recommendedRoute.departureTimeHour, recommendedRoute.departureTimeMinute))
    print('\nArrival Time: %g : %g' % (recommendedRoute.arrivalTimeHour, recommendedRoute.arrivalTimeMinute))
    print('\nDriving Time: %g minutes' % (recommendedRoute.travelTime))
#     print('\nShortest possible driving time: %g minutes' % (m.ObjVal))
    print('\nWaiting Time: %g minutes' % (recommendedRoute.waitingTime))
    print('\nEnergy Price: %g €' % (recommendedRoute.energyPrice))

def routeOptimization(arrivalTime,drivingWeight,waitingWeight,energyWeight):
    """
    Values to get from forecasting:
    Temperature for the relevant timeframe
    Energyprices
    Energy consumption of the windshieldwipers
    Energy consumption of the lights
    """
    temperatureModel = load_model('temperatureModel.h5')
    precipitationModel = load_model('precipitationModel.h5')
    cityModel = load_model('cityModel.h5')
    highwayModel = load_model('highwayModel.h5')
    
    arrivalDT = datetime.strptime(arrivalTime, '%Y-%m-%dT%H:%M')
    month = arrivalDT.month
    day = arrivalDT.day 
    hour = arrivalDT.hour
    minute = arrivalDT.minute
    
    timeArray = []
    
    for i in range(arrivalDT.hour - 2,arrivalDT.hour+1):
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
        departureTimes.append(arrivalDT + timedelta(minutes = m))
    
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
    
    arrCityDF = pd.DataFrame(arrivalTimesCity,columns=['Route','depH','depM','arrH','arrM','drivingTime','waitingTime])
    
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
            
    arrHighwayDF = pd.DataFrame(arrivalTimesHighway,columns=['Route','depH','depM','arrH','arrM','drivingTime','waitingTime])
    arrDF = arrCityDF.append(arrHighwayDF)   
    arrDF.columns = ['route','dep_H','dep_M','arr_H','arr_M','dri_T','wai_T']     
    
    energyDF = timeDF.iloc[:,:3]
    energyRequest = np.array(energyDF)
    energyRequest = np.reshape(energyRequest, (energyRequest.shape[0], 1, energyRequest.shape[1]))
    precipitation = precipitationModel.predict(energyRequest)
    precipitation = precipitation > 0.5
    temperature = temperatureModel.predict(energyRequest)
    
    whiperCons = getWhiperConsumption(precipitation,arrDF)
    lightCons = getLightConsumption(precipitation,arrDF,hour)
    acCons = getAcConsumption(temperature,arrDF)
    
    EUC = []
    for i in range(len(whiperCons)):
        EUC.append()
    EUC = 4551
    EL = 4181
    unitPrice = 0.2774
            
    
    #data = pd.read_csv('Dataset_goal_3_time_csv.csv',sep=';')
    
    ### Calcuate average travel time from max and min travel time
    #city = list()
    #highway = list()
    #for row in np.arange(data.shape[0]):
    #    #print('Scenario I average: ',sum(list(data.iloc[row,3:5].values))/2)
    #    city.append(sum(list(data.iloc[row,3:5].values))/2)
    #    
    #    #print('Scenario II average: ',sum(list(data.iloc[row,6:].values))/2)
    #    highway.append(sum(list(data.iloc[row,6:].values))/2)
    #    
    #data['averageTimeCity'] = city
    #data['averageTimeHighway'] = highway
    #
    #### Split Columns
    #
    #daySplitted = data.Day.str.split(' - ', n = 1, expand=True)
    #
    #data['Weekday'] = daySplitted[0]
    #data['Date'] = daySplitted[1]
    #
    #dateSplitted = data.Date.str.split('/',n=2, expand = True)
    #
    #data['Month'] = dateSplitted[0]
    #data['Day'] = dateSplitted[1]
    #data['Year'] = dateSplitted[2]
    #
    #timeSplitted = data['Departure time'].str.split(':',n=1, expand=True)
    #
    #data['Hour'] = timeSplitted[0]
    #data['Minute'] = timeSplitted[1]
    #
    #temperature = pd.read_csv('temp.csv', sep='\t')
    #temperature.columns = ['day','time','temperature']
    #temperature = temperature.apply(lambda x: x.str.replace(',','.'))
    #time = pd.DataFrame(temperature.time.str.split(':').tolist(),columns=['hour','minute','second'])
    #temperature = temperature.merge(time,on=temperature.index)
    #temperature.drop(columns = ['key_0','time'],inplace=True)
    
    ### User specified values
    """
    User selects a day and a desired arrival time
    Possible days to select: Monday, Wednesday, Friday, Sunday
    Driving data for 06:00 - 10:00
    """
    desiredArrivalTimeHour = arrivalDT.hour
    desiredArrivalTimeMinute = arrivalDT.minute
    """
    Weight for the importance of driving and waiting time
    Needs to be a value between 0 and 1
    If weight is 0: Only waiting time is considered
    If weight is 1: Only driving time is considered
    """
    #drivingWeight = 10
    #waitingWeight = 1
    #energyWeight = 3
    
    """
    Prices for Driving Time and Waiting Time
    """
    
    pricePerHour = 20
    pricePerMinute = pricePerHour/60
    
        
    days = list(data.Weekday.drop_duplicates())
        
    tupledict_input = list()
    N=10
    for d in days:
        tempData = data[data.Weekday == d]
        for st in range(len(tempData)):
            startingTime = tempData['Departure time'].values[st]
            averageTravelTimeCity = tempData['averageTimeCity'].values[st]
            averageTravelTimeHighway = tempData['averageTimeHighway'].values[st]
            tupledict_input.append([('city', d, tempData.iloc[st,:].Year,tempData.iloc[st,:].Month,
                                    tempData.iloc[st,:].Day,tempData.iloc[st,:].Hour,tempData.iloc[st,:].Minute), 
                                    averageTravelTimeCity])
            tupledict_input.append([('highway', d, tempData.iloc[st,:].Year,tempData.iloc[st,:].Month,
                                    tempData.iloc[st,:].Day,tempData.iloc[st,:].Hour,tempData.iloc[st,:].Minute), 
                                    averageTravelTimeHighway])
    
    times = tupledict(tupledict_input) 
    relevantTimes = times.subset('*', selectedDay)
    
    ### calculate arrival time for each trip with min time
    arrivalTime = list()
    
    for t in relevantTimes.iteritems():
    #     arrivalTime.append(datetime(int(t[0][2]),int(t[0][3]),int(t[0][4]),
    #                              int(t[0][5]),int(t[0][6])) + timedelta(minutes = min_value))
        arrivalTime.append(datetime(int(t[0][2]),int(t[0][3]),int(t[0][4]),
                                 int(t[0][5]),int(t[0][6])) + timedelta(minutes = t[1]))    
    
    desiredArrivalTime = desiredArrivalTimeHour * 60 + desiredArrivalTimeMinute
    arrivalTimeMinutes = [arrivalTime[x].hour * 60 + arrivalTime[x].minute for x in range(len(arrivalTime))]
    
    m = Model()
    
    ### Create DataFrame with all possible routes and departure times
    pddf = pd.DataFrame.from_dict(relevantTimes.items())
    
    pddfSplitted = pd.DataFrame(pddf[0].tolist(), index=pddf.index)
    pddfSplitted.columns=['Route','Weekday','Year','Month','Day','departureTimeHour','departureTimeMinute']
    pddfSplitted['travelTime']=pddf[1]
    
    ### Add arrival time
    arrivalHour = [arrivalTime[x].hour for x in range(len(arrivalTime))]
    arrivalMinute = [arrivalTime[x].minute for x in range(len(arrivalTime))]
    
    pddfSplitted['arrivalTimeHour'] = arrivalHour
    pddfSplitted['arrivalTimeMinute'] = arrivalMinute
    
    pddfSplitted['temperature'] = 0
    for i in range(len(pddfSplitted)):
        for j in range(len(temperature)):
            if (pddfSplitted.departureTimeHour[i] == temperature.hour[j]) & (pddfSplitted.departureTimeMinute[i] == temperature.minute[j]):
                pddfSplitted.temperature[i] = temperature.temperature[j]
    
    ### Add waiting time
    waitingTime = [desiredArrivalTime - arrivalTimeMinutes[x] for x in range(len(arrivalTimeMinutes))]
    pddfSplitted['waitingTime'] = waitingTime
    
    ### Keep only scenarios where arrival is before goal time
    possibleRecommendations = pddfSplitted[pddfSplitted.waitingTime >= 0]
    possibleRecommendations.index = range(len(possibleRecommendations))
    
    ### energy min
    

    
    possibleRecommendations['energyConsumption'] = 0
    possibleRecommendations['energyPrice'] = 0
    for i in range(len(possibleRecommendations)):
        t = float(possibleRecommendations.temperature[i])
    #     EC = 1.974*math.pow(t,2) - 44.889*t +4409.5
        EC = -0.000002*math.pow(t,5)+0.0003*math.pow(t,4)-0.0214*math.pow(t,3)+0.5865*math.pow(t,2)-9.3772*t+290.39
        possibleRecommendations.energyConsumption[i] = EC+EUC+EL
        possibleRecommendations['energyPrice'][i] = possibleRecommendations.energyConsumption[i] * unitPrice / 1000
        
    possibleRecommendations['drivingPrice'] = possibleRecommendations.travelTime * pricePerMinute
    possibleRecommendations['waitingPrice'] = possibleRecommendations.waitingTime * pricePerMinute 
    
    ### Scenario scores
    # scenarioScores = possibleRecommendations.travelTime * weightDrivingTime + possibleRecommendations.waitingTime * weightWaitingTime
    # recommendedRoute = possibleRecommendations.iloc[scenarioScores.idxmin()]
    
    scenarioScores = possibleRecommendations.drivingPrice * drivingWeight + possibleRecommendations.waitingPrice * waitingWeight + possibleRecommendations.energyPrice * energyWeight
    recommendedRoute = possibleRecommendations.iloc[scenarioScores.idxmin()]
    
    
    ### variable to choose only the minimum driving time
    
    choice = list()
    for x in range(len(relevantTimes)):
        choice.append(m.addVar(vtype=GRB.BINARY))
        
    temp = relevantTimes.select()
    
    m.update()
    
    m.addConstr(quicksum(choice) == 1)
    
    m.setObjective(quicksum(temp[x] * choice[x] for x in range(len(temp))), GRB.MINIMIZE)
    
    m.optimize()
    
    printSolution()
    
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
            light.append(1)
        else:
            light.append(0)
    
    dH = np.array(arrDF.dep_H) 
    aH = np.array(arrDF.arr_H)
    hs = np.unique(np.concatenate((dH,aH)))
    for i in range(len(arrDF)):
        if day[np.where(hs == arrDF.dep_H[i])[0][0]] == 1 & arrDF.route == 'Highway':
            lightConsumption.append(lightConsumptionHighwayNight)
        elif day[np.where(hs == arrDF.dep_H[i])[0][0]] == 0 & arrDF.route == 'Highway':
            lightConsumption.append(lightConsumptionHighwayDay)
        elif day[np.where(hs == arrDF.dep_H[i])[0][0]] == 1 & arrDF.route == 'City':
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
        t = temperature[np.where(hd == arrDF.dep_H[i])[0][0]]
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