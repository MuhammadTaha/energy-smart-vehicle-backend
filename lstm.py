import numpy as np
import pandas as pd
import matplotlib.pyplot as plt2
import math
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense,Activation, Dropout

data_dim = 16
timesteps = 8
num_classes = 10

print("sdfsdfdfsdfsffffeeww")
def get_data(normalized=0):

    col_names = ['Date', 'Day_No','NumberTripsperday','RelativeFrequency','1_or_0','PercentageTripperHour','HourofDay','Precipitation']
    stocks = pd.read_csv(r"Dataset_Daylight_Rainfall_Berlin4.csv", header=0)

    df = pd.DataFrame(stocks)
    df['RelativeFrequency'] = df['RelativeFrequency']*100 ;
    df['PercentageTripperHour'] = df['PercentageTripperHour']*100;
    df['HourofDay'] = pd.to_datetime(df['HourofDay'],format= '%H:%M:%S' ).dt.hour;

    return df
print("in lstm")
data = get_data()
# labels = data.columns

data.drop(['Date'],inplace=True, axis='columns')


data["1_or_0"] = to_categorical(data["1_or_0"])

train_size = int(len(data) * 0.8)
train, test = data[0:train_size], data[train_size:len(data)]
print('Observations: %d' % (len(data)))
print('Training Observations: %d' % (len(train)))
print('Testing Observations: %d' % (len(test)))


x_train = train.iloc[:,:-4]
y_train = train.iloc[:,-4:]

x_test = test.iloc[:,:-4]
y_test = test.iloc[:,-4:]


data_dim =  x_train.shape[1] #number of input features

# # expected input data shape: (batch_size, timesteps, data_dim)
model = Sequential()
model.add(LSTM(32, return_sequences=True,
               input_shape=(1, data_dim)))  # returns a sequence of vectors of dimension 32
model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
model.add(LSTM(32))  # return a single vector of dimension 32
model.add(Dense(4))
model.add(Activation("linear"))

model.compile(loss='mean_squared_error',
              optimizer='rmsprop',
              metrics=['accuracy'])

trainX = np.array(x_train)
testX = np.array(x_test)
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

model.fit(trainX, y_train,
          batch_size=64, epochs=20)

trainScore = model.evaluate(trainX,y_train , verbose=0)
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore[0], math.sqrt(trainScore[0])))
#
testScore = model.evaluate(testX, y_test, verbose=0)
print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore[0], math.sqrt(testScore[0])))

p = model.predict(testX)

plt2.subplot(2, 1, 1)

plt2.axis((0,100,0,30))
val = np.array(p)
print(p)
plt2.plot(np.array(p),color='red', label='prediction')

plt2.plot(np.array(y_test),color='blue', label='test')
plt2.xlabel('no of tuples')
plt2.title('Modal evalution and prediction using Test Data')
plt2.legend(loc='upper right')



# Power calculation

both = 53.6 + 53.6 + 53 + 35.4 + 35.4
day  = 45.8 + both
night = 112.4 + 127.8 + 7.4 + 19.2+ 14.4 + 4.8 + both

#
plt2.subplot(2, 1, 2)
for v in val:
    for i in range(len(v)):
        if(i == 2 ):
            if(v[i] >= 0.5):
                pSum = (both+day)*(v[0]/100) * 11.1 * 0.22194
                plt2.plot(math.ceil(v[1]), pSum,'.-')
            else:
                pSum = (both+night)*(v[0]/100) * 11.1 * 0.22194
                plt2.plot(math.ceil(v[1]),pSum, '.-')


# plt2.plot(arr, '.-')
plt2.xlabel('time (hours)')
plt2.ylabel('Power consumed(kwh)')
plt2.title("Lights")

plt2.subplot(2, 2, 2)
for v in val:
    for i in range(len(v)):
        if(i == 3 ):
                pSum = 40*(v[0]/100) * 11.1 * 0.22194 * v[i]

                plt2.plot(math.ceil(v[1]), pSum,'.-')

# precipitation
plt2.xlabel('Time(hours)')
plt2.ylabel('Power consumed(kwh)')
plt2.title("Screen Wiper")

plt2.show()

