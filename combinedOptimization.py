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

"""
Values to get from forecasting:
Temperature for the relevant timeframe
Energyprices
Energy consumption of the windshieldwipers
Energy consumption of the lights
"""
data = pd.read_csv('Dataset_goal_3_time_csv.csv',sep=';')

### Calcuate average travel time from max and min travel time
city = list()
highway = list()
for row in np.arange(data.shape[0]):
    #print('Scenario I average: ',sum(list(data.iloc[row,3:5].values))/2)
    city.append(sum(list(data.iloc[row,3:5].values))/2)
    
    #print('Scenario II average: ',sum(list(data.iloc[row,6:].values))/2)
    highway.append(sum(list(data.iloc[row,6:].values))/2)
    
data['averageTimeCity'] = city
data['averageTimeHighway'] = highway

### Split Columns

daySplitted = data.Day.str.split(' - ', n = 1, expand=True)

data['Weekday'] = daySplitted[0]
data['Date'] = daySplitted[1]

dateSplitted = data.Date.str.split('/',n=2, expand = True)

data['Month'] = dateSplitted[0]
data['Day'] = dateSplitted[1]
data['Year'] = dateSplitted[2]

timeSplitted = data['Departure time'].str.split(':',n=1, expand=True)

data['Hour'] = timeSplitted[0]
data['Minute'] = timeSplitted[1]

temperature = pd.read_csv('temp.csv', sep='\t')
temperature.columns = ['day','time','temperature']
temperature = temperature.apply(lambda x: x.str.replace(',','.'))
time = pd.DataFrame(temperature.time.str.split(':').tolist(),columns=['hour','minute','second'])
temperature = temperature.merge(time,on=temperature.index)
temperature.drop(columns = ['key_0','time'],inplace=True)

### User specified values
"""
User selects a day and a desired arrival time
Possible days to select: Monday, Wednesday, Friday, Sunday
Driving data for 06:00 - 10:00
"""
selectedDay = 'Monday'
desiredArrivalTimeHour = 10
desiredArrivalTimeMinute = 0
"""
Weight for the importance of driving and waiting time
Needs to be a value between 0 and 1
If weight is 0: Only waiting time is considered
If weight is 1: Only driving time is considered
"""
drivingWeight = 10
waitingWeight = 1
energyWeight = 3

"""
Prices for Driving Time and Waiting Time
"""
pricePerHour = 20
pricePerMinute = pricePerHour/60


def assignValuesCombined(data):
    print(data)



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

EUC = 4551
EL = 4181
unitPrice = 0.2477

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