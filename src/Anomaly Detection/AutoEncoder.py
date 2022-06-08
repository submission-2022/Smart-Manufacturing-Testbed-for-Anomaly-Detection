import pandas
import sys
import matplotlib.pyplot as plt
import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt

## Enter File Name Here
file_name = 'Process_data_Outside Dewpoint.csv'

# convert an array of values into a dataset matrix (lookback: the number of previous time steps to use as input variables to predict the next time period. Here lookback=1)
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)


# fix random seed for reproducibility
numpy.random.seed(7)

# load the dataset
dataframe = pandas.read_csv(file_name, usecols=[1], engine='python')
dataset = dataframe.values
dataset = dataset.astype('float32')
#plt.plot(dataset)
#plt.show()

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# split into train and test sets
train_size = int(len(dataset) * 0.6)
test_size = (len(dataset) - train_size) * 0.1
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(len(train), len(test))

create_dataset(dataset, look_back=1)
	
# reshape into X=t and Y=t+1
look_back = 20
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
print(trainX.shape, len(trainX), trainY.shape)
print(testX.shape, len(testX), testY.shape)
	
# create and fit the LSTM network (4 LSTM blocks or neurons, and an output layer that makes a single value prediction)
model = Sequential(
    [
        layers.Input(shape=(trainX.shape[1], trainX.shape[2])),
        layers.Conv1D(
            filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
        ),
        layers.Dropout(rate=0.2),
        layers.Conv1D(
            filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
        ),
        layers.Conv1DTranspose(
            filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
        ),
        layers.Dropout(rate=0.2),
        layers.Conv1DTranspose(
            filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
        ),
        layers.Conv1DTranspose(filters=1, kernel_size=7, padding="same"),
    ]
)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
model.summary()
model.fit(trainX, trainY, epochs=5, batch_size=10, verbose=2)



# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
print(trainPredict[:,0,0].shape)
# # invert predictions (to return to original data scale)
# trainPredict = scaler.inverse_transform(trainPredict[0])
# trainY = scaler.inverse_transform([trainY])
# testPredict = scaler.inverse_transform(testPredict[0])
# testY = scaler.inverse_transform([testY])
# calculate root mean squared error
print(trainY.shape)
#print(trainPredict.shape)
trainScore = math.sqrt(mean_squared_error(trainY, trainPredict[:,0,0]))
print('Train Score: %.3f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY, testPredict[:,0,0]))
print('Test Score: %.3f RMSE' % (testScore))	

