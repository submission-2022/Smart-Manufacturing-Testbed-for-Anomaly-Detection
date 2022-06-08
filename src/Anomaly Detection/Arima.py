from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
#from pandas.tools.plotting import autocorrelation_plot
import time
 
def parser(x):
	 return datetime.strptime(x, '%m-%d')
	
## Enter File Name Here
file_name = 'Process_data_Outside Dewpoint.csv'
	
dev_id = 32 
series = read_csv(file_name, header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
X = series.values
size = int(len(X) * 0.6)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
model = ARIMA(history, order=(5,2,0))
model_fit = model.fit()

start_time = time.time()

for t in range(len(test)):
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	print('predicted=%f, expected=%f' % (yhat, obs))

test_run_time_seconds = time.time() - start_time
print('ARIMA Inference Time:' + str(test_run_time_seconds))
 
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)