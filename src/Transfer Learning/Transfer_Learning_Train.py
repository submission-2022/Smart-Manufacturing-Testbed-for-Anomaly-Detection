import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from keras.utils import to_categorical


# define baseline model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(8, input_dim=2, activation='relu'))
	model.add(Dense(8, activation='relu'))
	model.add(Dense(3, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

seed = 7
numpy.random.seed(seed)


## Input Parameters
Num_Samples = 2797
Dataset_File = "knoy_mpu_train_100.csv"


# load dataset
dataframe = pandas.read_csv(Dataset_File, header=0, index_col=0)
dataset = dataframe.values
X = dataset[1:Num_Samples,0:2].astype(float)
Y = dataset[1:Num_Samples,2]
#scalar = MinMaxScaler()
#scalar.fit(X)
#X = scalar.transform(X)
print(X)
print(Y)


# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)



estimator = KerasClassifier(build_fn=baseline_model, epochs=10, batch_size=500, verbose=0)


kfold = KFold(n_splits=10, shuffle=True, random_state=seed)


results = cross_val_score(estimator, X, Y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))



