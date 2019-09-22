import numpy as np
import pandas as pd
import matplotlib.pyplot as plt2
import math
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense,Activation, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
timesteps = 8
num_classes = 10

def get_data(normalized=0):
    col_names = ['s_year','Day', 'Day_or_Night',"Precipitation_Qty","marketprice [Euro/MWh]","CityRouteTime(sec) - distance 10.8km","HighwayRouteTime(sec) - distance 16.5 km"]
    stocks = pd.read_csv(r"Master_Dataset_Final.csv", encoding = "ISO-8859-1" , header=0)
    stocks.drop(columns=col_names)
    df = pd.DataFrame(stocks)
    df.drop(col_names,inplace=True, axis='columns')
    return df


print("in lstm")
data = get_data()
data.drop(columns=['DayNum','Night','No_Precipitation', 'Temperature',
       'DayTime','EnergyConsumedWiper(Wh/km)'],inplace=True)

train_size = int(len(data) * 0.8)
train, test = data[0:train_size], data[train_size:len(data)]

x_train = train.iloc[:,:-1]
print(x_train.columns)
y_train = train.iloc[:,-1:]
print(y_train.columns)

x_test = data.iloc[:,:-1]
y_test = data.iloc[:,-1:]

data_dim =  x_train.shape[1] #number of input features

# # expected input data shape: (batch_size, timesteps, data_dim)
model = Sequential()
model.add(LSTM(64, return_sequences=True,
               input_shape=(1, data_dim)))  # returns a sequence of vectors of dimension 32
model.add(LSTM(64, return_sequences=True))  # returns a sequence of vectors of dimension 32
model.add(LSTM(64))  # return a single vector of dimension 32
model.add(Dropout(0.5))
model.add(Dense(1))

model.add(Activation("linear"))

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])

trainX = np.array(x_train)
testX = np.array(x_test)
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

callbacks   = [
      EarlyStopping(monitor='val_loss', patience=5, mode='min', min_delta=0.01),
      ModelCheckpoint('\lightModel.h5', monitor='val_loss', save_best_only=True, mode='min')
]
model.fit(trainX, y_train,
          batch_size=32, epochs=100,callbacks=callbacks)

trainScore = model.evaluate(trainX,y_train , verbose=0)
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore[0], math.sqrt(trainScore[0])))
#
testScore = model.evaluate(testX, y_test, verbose=0)
print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore[0], math.sqrt(testScore[0])))

p = model.predict(testX)

model.save('precipitationModel.h5')  # creates a HDF5 file 'my_model.h5'


val = np.array(p)
print(p)

plt2.plot(np.array(p),color='red', label='prediction')

plt2.plot(np.array(y_test),color='blue', label='test')
plt2.xlabel('no of tuples')
plt2.title('Modal evalution and prediction using Test Data')
plt2.legend(loc='upper right')

plt2.show()

