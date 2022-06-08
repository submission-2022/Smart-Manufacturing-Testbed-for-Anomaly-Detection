# Smart-Manufacturing-Testbed-for-Anomaly-Detection
We have publicly released our source codes and benchmark data to enable others reproduce our work. In particular, we are publicly releasing, with this submission, our smart manufacturing database corpus of 4 datasets. This resource will encourage the community to standardize efforts at benchmarking anomaly detection in this important domain. We also encourage the community to expand this resource by contributing their new datasets and models.



# Code Instructions

Anomaly Detection Codes:
Under Anomaly detection folder, we have the following source codes:

1.	Arima.py: Training and testing ARIMA forecasting model 
2.	LSTM.py: Training and testing LSTM forecasting model 
3.	AutoEncoder.py: Training and testing AutoEncoder forecasting model 
4.	DNN.py: Training and testing DNN forecasting model 
5.	RNN.py: Training and testing RNN forecasting model 
6.	GluonTModels.py: That file contains the rest of the forecasting models along with the required functions and hyper-parameters. In particular, it contains the codes to train and examine performance for the following models:
a.	DeepAR
b.	DeepFactors
c.	Seasonal Naive
d.	Random Forest
e.	Auto-Regression

Transfer Learning Codes:
Under Transfer learning folder, we have the following source codes:
1.	Transfer_Learning_Pre_processing.py: This code prepares the CSV files used for training defect type classifiers and transfer learning. 

2.	Transfer_Learning_Train.py: This code trains and tests the defect type classifier, including feature encoding, defect classifier building, testing, and performance reporting. 


Running the Codes:
To run any code, we need just to run the command “python code_name.py” while changing the datafile name inside the code to the dataset of choice.
Prerequisites:
Our codes have the following libraries that need to be installed (which can be installed using apt-get install or conda): 
(i)   numpy
(ii)  scipy
(iii) pandas
(iv) GluonTS
(v)  sklearn
(vi) keras
(vii) statsmodels
(viii) matplotlib
(x)  simplejson
(xi) re
(xii) csv


# Datsets Instructions
(1)-(2) MEMs and  Piezoelectric datasets

Highlights of the datasets
To build these datasets, an experiment was conducted in the motor testbed to collect machine condition data for different health conditions. During the experiment, the acceleration signals were collected from both piezoelectric and MEMS sensors at the same time with the sampling rate of 3.2 kHz and 10 Hz, respectively, for X, Y, and Z axes. Different levels of machine health condition was induced by mounting a mass on the balancing disk, thus different levels of mechanical imbalance are used to trigger failures. Failure conditions were classified as one of three possible states - normal, near-failure, and failure. In this experiment, three levels of mechanical imbalance (i.e., normal, near-failure, failure) were considered acceleration data were collected at the ten rotational speeds (100, 200, 300, 320, 340, 360, 380, 400, 500, and 600 RPM) for each condition. While the motor is running, 50 samples were collected at a 10 second interval, for each of the ten rotational speeds.

Reading the dataset:
Both Piezoelectric and MEMs databases are in CSV format. For Anomaly detection, we have a single RPM (CSV file) while for transfer learning we have several rpms for each RPM (where all CSV files are compressed in .zip format). The CSV file for Piezoelectric has many more samples (due to higher sampling rate). Each data instance (row) contains the following columns: X, Y, Z for the several vibration sensors.

(3) Process data
Highlights of the dataset:
• Start date: 7/27/2021
• End date: 3/21/2022 
• Measurement Columns: Air Pressure 1, Air Pressure 2, Chiller 1  
  Supply Tmp, Chiller 2 Supply Tmp, Outside Air Temp, Outside 
  Humidity, Outside Dewpoint, etc.
• Measurement interval: 5 mins (1 data point per 5 min)
• Description: SPC data collected from internally mounted sensors in 
  Processes.

Reading the dataset:
The Process data is in CSV format. The CSV file has around 49K instances where each data instance (row) contains the following columns: Timestamp, Air Pressure 1, Air Pressure 2, Chiller 1 Supply Tmp, Chiller 2 Supply Tmp, Outside Air Temp, Outside Humidity, and Outside Dewpoint. 

Abnormal Dates:
We observed abnormal operations of the machine which occurred on February 1st 2022 and March 8th 2022.

(4) Pharmaceutical Packaging

Highlights of the dataset:
• Start date: 9/13/2021, some data loss between December and January
• End date: May 2022
• Measurement location: Air Compressor 2, Chiller 1, Chiller 2, Jomar 
  moulding machine
• Sample rate for each axis: 3.2 kHz
• Measurement interval: 30 mins
• Measurement duration: 1 seconds

Reading the dataset
• Each .txt file indicates vibration data for each day
• Vibration data for one measurement is written into 5 lines
•1st line: date and time that the measurement started
•2nd line: x-axis vibration data (3200 data points)
•3rd line: y-axis vibration data (3200 data points)
•4th line: z-axis vibration data (3200 data points)
•5th line: time difference between each data point






