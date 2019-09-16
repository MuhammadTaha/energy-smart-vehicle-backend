# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt2
# import math
# from keras.utils import to_categorical
# from keras.models import Sequential
# from keras.layers import LSTM, Dense,Activation, Dropout
#
# data_dim = 16
# timesteps = 8
# num_classes = 10
#
# def get_data(normalized=0):
#
#     # col_names = ['s_year', 'Day_No','NumberTripsperday','RelativeFrequency','1_or_0','PercentageTripperHour','HourofDay','Precipitation']
#     col_names = ['s_year','Day', 'Day_or_Night',"marketprice [Euro/MWh]", "CityRouteTime(sec) - distance 10.8km", "HighwayRouteTime(sec) - distance 16.5 km"]
#     stocks = pd.read_csv(r"Master_Dataset_Final.csv", encoding = "ISO-8859-1" , header=0)
#     stocks.drop(columns=col_names)
#     df = pd.DataFrame(stocks)
#     df.drop(col_names,inplace=True, axis='columns')
#
#     df['Temperature'] = df['Temperature'].astype(int);
#     # df['PercentageTripperHour'] = df['PercentageTripperHour']*100;
#     # df['HourofDay'] = pd.to_datetime(df['HourofDay'],format= '%H:%M:%S' ).dt.hour;
#
#     print(df.head(10))
#     print(list(df.columns))
#     return df
# print("in lstm")
# data = get_data()
# # labels = data.columns
# #
# # data.drop(['Date'],inplace=True, axis='columns')
# #
# #
# # data["1_or_0"] = to_categorical(data["1_or_0"])
# #
# train_size = int(len(data) * 0.8)
# train, test = data[0:train_size], data[train_size:len(data)]
# print('Observations: %d' % (len(data)))
# print('Training Observations: %d' % (len(train)))
# print('Testing Observations: %d' % (len(test)))
# #
# #
# # x_train = train.iloc[:,:]
# # print(x_train.columns)
# # y_train = train.iloc[:,-5:]
# # print(y_train.columns)
# #
# # x_test = test.iloc[:,:]
# # y_test = test.iloc[:,-5:]
# x_train = train.iloc[:,:-4]
# y_train = train.iloc[:,-4:]
#
# x_test = test.iloc[:,:-4]
# y_test = test.iloc[:,-4:]
#
# data_dim =  x_train.shape[1] #number of input features
#
# # # expected input data shape: (batch_size, timesteps, data_dim)
# model = Sequential()
# model.add(LSTM(32, return_sequences=True,
#                input_shape=(1, data_dim)))  # returns a sequence of vectors of dimension 32
# model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
# model.add(LSTM(32))  # return a single vector of dimension 32
# model.add(Dense(4))
# model.add(Activation("linear"))
#
# model.compile(loss='mean_squared_error',
#               optimizer='rmsprop',
#               metrics=['accuracy'])
# #
# trainX = np.array(x_train)
# testX = np.array(x_test)
# trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
# testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
#
# model.fit(trainX, y_train,
#           batch_size=64, epochs=3)
#
# trainScore = model.evaluate(trainX,y_train , verbose=0)
# print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore[0], math.sqrt(trainScore[0])))
# #
# testScore = model.evaluate(testX, y_test, verbose=0)
# print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore[0], math.sqrt(testScore[0])))
#
# p = model.predict(testX)
#
# plt2.subplot(2, 1, 1)
#
# plt2.axis((0,100,0,30))
# val = np.array(p)
# print(p)
# plt2.plot(np.array(p),color='red', label='prediction')
#
# plt2.plot(np.array(y_test),color='blue', label='test')
# plt2.xlabel('no of tuples')
# plt2.title('Modal evalution and prediction using Test Data')
# plt2.legend(loc='upper right')
#
# #
# #
# # # Power calculation
# #
# # both = 53.6 + 53.6 + 53 + 35.4 + 35.4
# # day  = 45.8 + both
# # night = 112.4 + 127.8 + 7.4 + 19.2+ 14.4 + 4.8 + both
# #
# # #
# # plt2.subplot(2, 1, 2)
# # for v in val:
# #     for i in range(len(v)):
# #         if(i == 2 ):
# #             if(v[i] >= 0.5):
# #                 pSum = (both+day)*(v[0]/100) * 11.1 * 0.22194
# #                 plt2.plot(math.ceil(v[1]), pSum,'.-')
# #             else:
# #                 pSum = (both+night)*(v[0]/100) * 11.1 * 0.22194
# #                 plt2.plot(math.ceil(v[1]),pSum, '.-')
# #
# #
# # # plt2.plot(arr, '.-')
# # plt2.xlabel('time (hours)')
# # plt2.ylabel('Power consumed(kwh)')
# # plt2.title("Lights")
# #
# # plt2.subplot(2, 2, 2)
# # for v in val:
# #     for i in range(len(v)):
# #         if(i == 3 ):
# #                 pSum = 40*(v[0]/100) * 11.1 * 0.22194 * v[i]
# #
# #                 plt2.plot(math.ceil(v[1]), pSum,'.-')
# #
# # # precipitation
# # plt2.xlabel('Time(hours)')
# # plt2.ylabel('Power consumed(kwh)')
# # plt2.title("Screen Wiper")
# #
# plt2.show()
# #




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt2
import math
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense,Activation, Dropout
from sklearn.model_selection import train_test_split
timesteps = 8
num_classes = 10

def get_data(normalized=0):
    col_names = ['s_year','Day', 'Day_or_Night',"Precipitation_Qty","marketprice [Euro/MWh]","CityRouteTime(sec) - distance 10.8km","HighwayRouteTime(sec) - distance 16.5 km"]
    stocks = pd.read_csv(r"Master_Dataset_Final.csv", encoding = "ISO-8859-1" , header=0)
    # print(stocks.columns)
    stocks.drop(columns=col_names)
    df = pd.DataFrame(stocks)
    df.drop(col_names,inplace=True, axis='columns')
    #
    # df['Temperature'] = df['Temperature'].astype(int)
    return df


print("in lstm")
data = get_data()
# labels = data.columns

# data.drop(['Date'],inplace=True, axis='columns')


# data["1_or_0"] = to_categorical(data["1_or_0"])

train_size = int(len(data) * 0.8)
# train, test = data[0:train_size], data[train_size:len(data)]
# rs = ShuffleSplit(n_splits=5, random_state=0, test_size=0.25, train_size=None)
# train, test = rs.get_n_splits(data)

# print('Observations: %d' % (len(data)))
# print('Training Observations: %d' % (len(train)))
# print('Testing Observations: %d' % (len(test)))


x_train = data.iloc[:,:]
print(x_train.columns)
y_train = data.iloc[:,-6:]
print(y_train.columns)

# x_test = data.iloc[:,:]
# y_test = data.iloc[:,-6:]

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state = 2) # 0.2 test_size means 20%

data_dim =  x_train.shape[1] #number of input features

# # expected input data shape: (batch_size, timesteps, data_dim)
model = Sequential()
model.add(LSTM(64, return_sequences=True,
               input_shape=(1, data_dim)))  # returns a sequence of vectors of dimension 32
model.add(LSTM(64, return_sequences=True))  # returns a sequence of vectors of dimension 32
model.add(LSTM(64))  # return a single vector of dimension 32
# model.add(Dropout(0.7))
model.add(Dense(6))

model.add(Activation("linear"))

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])

trainX = np.array(x_train)
testX = np.array(x_test)
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

model.fit(trainX, y_train,
          batch_size=32, epochs=35)

trainScore = model.evaluate(trainX,y_train , verbose=0)
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore[0], math.sqrt(trainScore[0])))
#
testScore = model.evaluate(testX, y_test, verbose=0)
print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore[0], math.sqrt(testScore[0])))

p = model.predict(testX)

model.save('minEnergyModel.h5')  # creates a HDF5 file 'my_model.h5'


val = np.array(p)
print(p)

plt2.plot(np.array(p),color='red', label='prediction')

plt2.plot(np.array(y_test),color='blue', label='test')
plt2.xlabel('no of tuples')
plt2.title('Modal evalution and prediction using Test Data')
plt2.legend(loc='upper right')

plt2.show()

