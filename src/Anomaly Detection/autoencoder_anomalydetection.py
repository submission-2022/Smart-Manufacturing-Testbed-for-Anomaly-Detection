# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 10:40:24 2022

@author: jbg13
"""
from glob import glob
import os
import pandas as pd
import numpy as np
import yaml
import scipy
#os.getcwd()
import itertools
import matplotlib.pyplot as plt
#from statmodels
from sklearn.decomposition import PCA
import scipy.signal
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import datetime
from datetime import datetime
import os, time
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch
import pandas as pd
#import tensorflow as tf
#from tensorflow import keras
from scipy import stats, fft
from sklearn.model_selection import train_test_split
import csv

def load_dataset4(normal=True, name = "name"):
    name1 = "/spi/" + name
    PATH = os.getcwd() + name1
    EXT_fault = "*.txt"
    all_csv_files_fault = [file
                     for path, subdir, files in os.walk(PATH)
                     for file in glob(os.path.join(path, EXT_fault))]
    dd,xx,yy,zz,tt = [],[],[],[],[]
    for i in range(len(all_csv_files_fault)):
        file = open(all_csv_files_fault[i],'r')
        with open(all_csv_files_fault[i]) as f:
            content = f.readlines()
        try:
            content = [x.strip() for x in content]
            d = content[0::5]
            x = content[1::5]
            y = content[2::5]
            z = content[3::5]
            t = content[4::5]
            for ii in range(0,len(t)):
                d_t = d[ii]
                xx_t = np.array(x[ii][1:-1].split(', ')).astype(np.float)[:2400]#*7.8/1000)
                yy_t = np.array(y[ii][1:-1].split(', ')).astype(np.float)[:2400]#*7.8/1000)
                zz_t = np.array(z[ii][1:-1].split(', ')).astype(np.float)[:2400]#*7.8/1000)
                tt_t = np.array([np.array(t[ii][1:-1].split(', ')).astype(np.float)[-1]-np.array(t[ii][1:-1].split(', ')).astype(np.float)[0]])
                dd.append(d_t)
                xx.append(xx_t)
                yy.append(yy_t)
                zz.append(zz_t)
                tt.append(tt_t)
        except ValueError as e:
            continue
            
            f.close()
    #Fs = len(np.array(x[ii][1:-1].split(', ')).astype(np.float)) / np.median(ttt)
    
    #total = [dd,xx,yy,zz,tt]
    #for idx, item in enumerate(total):
    #    sorted_list = sorted(total, key=lambda t: datetime.strptime(dd[i], '%Y/%m/%d %H:%M:%S'))
    
    return dd,xx,yy,zz,tt

dd,xx,yy,zz,tt = load_dataset4(name = "2022")

dd_test = []
for i in range(len(dd)):
    try:
        dd_test.append(datetime.strptime(dd[i], "%a %b %d %H:%M:%S %Y"))
    except ValueError:
        try:
            dd_test.append(datetime.strptime(dd[i],"%a %d %b %Y %I:%M:%S %p EST"))
        except ValueError:
            dd_test.append(datetime.strptime(dd[i],"%a %d %b %Y %I:%M:%S %p EDT"))
        

xx_test, yy_test, zz_test = [], [] ,[]
xx_test = [x for y, x in sorted(zip(dd_test, xx), key=lambda pair: pair[0])]
yy_test = [x for y, x in sorted(zip(dd_test, yy), key=lambda pair: pair[0])]
zz_test = [x for y, x in sorted(zip(dd_test, zz), key=lambda pair: pair[0])]
dd_test = sorted(dd_test)

xx_test = pd.DataFrame(xx_test)
#xx_test = xx_test[5490:]        ## start from Sep 13
yy_test = pd.DataFrame(yy_test)
#yy_test = yy_test[5490:]
zz_test = pd.DataFrame(zz_test)
#zz_test = zz_test[5490:]
#dd_test = dd_test[5490:]
dd_test2 = []
for i in range(len(dd_test)):
    if i % 2 == 0:
        dd_test2.append(dd_test[i])
xx_test = xx_test.iloc[::2]
yy_test = yy_test.iloc[::2]
zz_test = zz_test.iloc[::2]
dd_test = dd_test2

xx_test = xx_test.reset_index(drop = True) #reset index
yy_test = yy_test.reset_index(drop = True)
zz_test = zz_test.reset_index(drop = True)


process_data = []
with open('../Process_data.csv', newline='') as csvfile:
    df = csv.reader(csvfile, delimiter=',')
    for row in df:
        header = []
        if i == 0:
            header = row
        else:
            process_data.append(row)
    process_data = pd.DataFrame(process_data, columns = ['Timestamp', 'Air Pressure1', 'Air Pressure2', 'Chiller 1 Supply Tmp',
       'Chiller 2 Supply Tmp', 'Outside Air Temp', 'Outside Humidity',
       'Outside Dewpoint'])
    process_time = process_data['Timestamp']
    process_data = process_data.apply(pd.to_numeric, errors='coerce')
    process_data['Timestamp'] = process_time
    process_data = process_data.iloc[1: , :]

process_data.columns = ['Timestamp', 'Air Pressure1', 'Air Pressure2', 'Chiller 1 Supply Tmp',
       'Chiller 2 Supply Tmp', 'Outside Air Temp', 'Outside Humidity',
       'Outside Dewpoint']

chiller1_SplyTmp = process_data['Chiller 1 Supply Tmp']
process_air_pressure2 = process_data['Air Pressure2']

#process_data = process_data[9931:] #Data from Sep 13th
#process_time = process_time[9931:] #timestamp from Sep 13th

process_data = process_data.reset_index(drop = True) #reset index

process_data['Timestamp'][:-1] = pd.to_datetime(process_data['Timestamp'][:-1]) # convert to timestamp type
process_data = process_data[:-1] #remove the last row (Grand total)
process_time = process_data['Timestamp']


labels = np.zeros(len(xx_test))
timestamp = []
measurement_period = pd.date_range("2022-01-30", periods=120) #setup the period that we want to look at
data = []
data_time = []
vibration_data = []
vibration_time = []
for date in measurement_period:
    
    for i in range(len(process_data)):
        if process_data['Timestamp'][i].date() == date.date():
            data.append(process_data['Chiller 1 Supply Tmp'][i])
            data_time.append(process_data['Timestamp'][i])
    for i in range(len(xx_test)):
        if dd_test[i].date() == date.date():
            vibration_data.append(xx_test.iloc[i])
            vibration_time.append(dd_test[i])
    
for idx, x in enumerate(vibration_data):
    count = 0
    current = []
    sum_data = 0
    for i in range(len(data)):
         if abs(data_time[i] - vibration_time[idx]) and (abs(data_time[i].minute - vibration_time[idx].minute) < 15):
             count = count + 1
             current.append(data[i])
             sum_data = sum_data + data[i]
    try:
        #label = sum_data/count
        label = np.max(current)
    except ZeroDivisionError:
        label = np.mean(labels[-3:])
                #label = 0
            
    labels[idx] = label

for i in range(len(labels)):
    count = 0
    current = []
    sum_data = 0
    for j in range(len(data)):
         if abs(data_time[j] - vibration_time[idx]) < datetime.timedelta(minutes = 10):
             count = count + 1
             current.append(data[i])
             sum_data = sum_data + data[i]
    try:
        label = sum_data/count
    except ZeroDivisionError:
        label = np.mean(labels[idx-3:idx])
        #label = 0
            
    labels[i] = label
    
for i in range(len(labels)):
    try:
        labels[i] = data[i]
    except IndexError:
        labels[i] = np.mean(labels[i-4:i-1])


def timeFeatures(data):
    feature = [] # initialize feature list
    for i in range(len(data)):
        mean = np.mean(data.iloc[i]) # mean
        std = np.std(data.iloc[i]) # standard deviation
        rms = np.sqrt(np.mean(data.iloc[i] ** 2)) # root mean squre
        peak = np.max(abs(data.iloc[i])) # peak
        cf = peak/rms # crest factor
        # number of feature of each measurement = 7
        feature.append(np.array([mean,std,rms,peak,cf], dtype=float))
    feature = np.array(feature)
    return feature # feature list, each element is numpy array with datatype float

# DFT magnitude data
def freqFeatures(data):
    feature = []
    for i in range(len(data)):
        N = len(data[i]) # number of data
        yf = 2/N*np.abs(fft.fft(data[i])[:N//2]) # yf is DFT signal magnitude
        feature.append(np.array(yf))
    feature = np.array(feature)
    return feature

x_time_data = timeFeatures(xx_test)
y_time_data = timeFeatures(yy_test)
z_time_data = timeFeatures(zz_test)
x_freq_data = freqFeatures(xx_test)
input_feature = x_time_data
#input_feature = xx_test
input_feature = np.asarray(input_feature)
input_feature = torch.Tensor(input_feature)
#labels = torch.tensor(labels)
#labels = np.asarray(labels)

def gen_labels(data):
    label = np.zeros(len(data))
    for i in range(len(data)):
        if x_time_data[i][3] > 3:
            label[i] = 1
        else:
            label[i] = 0
    return label

def gen_labels2(dd_test):
    label = np.zeros(len(dd_test))
    for i in range(len(dd_test)):
        if (dd_test[i].weekday() >= 0 and dd_test[i].weekday() <= 4):
            label[i] = 1
            if dd_test[i].weekday() == 4 and dd_test[i].hour >= 19:
                label[i] = 0
        elif dd_test[i].weekday() > 4:
            label[i] = 0
            if dd_test[i].weekday() == 6 and dd_test[i].hour > 22:
                label[i] = 1
        else:
            label[i] = 0
        
    return label

def gen_abnormal_labels(label,abnormal_date):
    for item in abnormal_date:
        for i in range(len(dd_test)):
            if (dd_test[i].date() == item.date()):
                label[i] = 2
    return label

abnormal_date = [datetime(2022,1,31), datetime(2022,2,1), datetime(2022,3,7), datetime(2022,3,8)]
label = gen_labels2(dd_test)
label = gen_abnormal_labels(label,abnormal_date)

label = gen_labels2(dd_test)
label = gen_labels(input_feature)
label = torch.Tensor(label)
#label = torch.Tensor(labels)


train_data, test_data, train_labels, test_labels = train_test_split(
    input_feature, label, test_size=0.5, random_state=20
)

train_test_split_ratio = 0.7
#input_feature = input_feature[::2]
#label = label[::2]
train_data, test_data = input_feature[:int(len(input_feature)*train_test_split_ratio)], input_feature[int(len(input_feature)*train_test_split_ratio):]
train_labels, test_labels = label[:int(len(label)*train_test_split_ratio)], label[int(len(label)*train_test_split_ratio):]
train_labels, test_labels = torch.Tensor(train_labels), torch.Tensor(test_labels)

class AE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder_hidden_layer = nn.Linear(
            in_features=kwargs["input_shape"], out_features=128
        )
        self.encoder_output_layer = nn.Linear(
            in_features=128, out_features=64
        )
        self.decoder_hidden_layer = nn.Linear(
            in_features=64, out_features=128
        )
        self.decoder_output_layer = nn.Linear(
            in_features=128, out_features=kwargs["input_shape"]
        )
        self.linear_output_layer = nn.Linear(in_features = 64, out_features = 3)

    def forward(self, features):
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        code = torch.relu(code)
        activation = self.decoder_hidden_layer(code)
        activation = torch.relu(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.relu(activation)
        class_out = self.linear_output_layer(code)
        return reconstructed, class_out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# create a model from `AE` autoencoder class
# load it to the specified device, either gpu or cpu

model = AE(input_shape=5).to(device)

# create an optimizer object
# Adam optimizer with learning rate 1e-3
optimizer = optim.Adam(model.parameters(), lr=1e-6)

# mean-squared error loss
classification_criterion = nn.MSELoss()
reconstruction_criterion = nn.L1Loss()

##Training

def train_model(model, train_losses, train_counter, epoch, wp, wr):
    for epoch in range(epochs):
        model = model.train()
        loss = 0
        for batch_idx, batch_features in enumerate(train_data):
        # load it to the active device
            batch_features = batch_features.to(device)
            targets = train_labels[batch_idx].to(device)
            # reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes
            optimizer.zero_grad()
            
            # compute reconstructions
            reconstruction, class_out = model(batch_features)
            
            # compute training reconstruction loss
            train_loss = wp*classification_criterion(class_out, targets) + wr*reconstruction_criterion(reconstruction, batch_features)
            
            # compute accumulated gradients
            train_loss.backward()
            
            # perform parameter update based on current gradients
            optimizer.step()
            
            # add the mini-batch training loss to epoch loss
            loss += train_loss.item()
            
            if batch_idx % 10 == 0: # We record our output every 10 batches
                train_losses.append(train_loss.item()) # item() is to get the value of the tensor directly
                train_counter.append((batch_idx*64) + ((epoch-1)*len(train_data)))
            if batch_idx % 100 == 0: # We visulize our output every 10 batches
                print(f'Epoch {epoch}: [{batch_idx*len(batch_features)}/{len(train_data)*5}] Loss: {train_loss.item()}')
## train model1
epochs = 20
outputs = []
train_losses = []
train_counter = []
wp=0.2
wr=0.8
train_model(model, train_losses, train_counter, epochs,wp, wr)

##Testing
def test_model(model, test_losses, test_counter, epochs):
    for epoch in range(epochs):
        model.eval()
        test_loss = 0
        correct = 0
        for batch_idx, batch_features in enumerate(test_data):
            batch_features = batch_features.to(device)
            targets = test_labels[batch_idx].to(device)
            
            # compute reconstructions
            reconstruction, class_out = model(batch_features)
            class_argmax = torch.argmax(class_out.data)
            #print(targets,class_out)
            test_loss += F.mse_loss(class_out, targets, reduction='sum').item()
            #test_classification_loss += F.mse_loss(class_out, targets, reduction='sum').item()
            #test_reconstruction_loss += F.mse_less(reconstruction, batch_features, reduction = 'sum').item()
            pred = class_argmax # we get the estimate of our result by look at the largest class value
            correct += pred.eq(targets.data.view_as(pred)).sum() # sum up the corrected samples
            pred_labels.append(pred.item())
  
        test_loss /= len(test_data)
        test_losses.append(test_loss)
        test_counter.append(len(train_data)*epoch)
        print(f'Test result on epoch {epoch}: Avg loss is {test_loss}, Accuracy: {100.*correct/len(test_data)}%')
epochs = 1
test_losses = []
test_counter = []
pred_labels = []
test_model(model,test_losses,test_counter,epochs)

#Confusion matrix
x_pred = test_labels.detach().cpu()
from sklearn.metrics import confusion_matrix
import seaborn as sn
pred_labels = torch.tensor(pred_labels)
matrix = confusion_matrix(x_pred, pred_labels)
matrix.diagonal()/matrix.sum(axis=1)
classes = ['stopped','running','abnormal']
df_cm = pd.DataFrame(matrix/np.sum(matrix)*10, index = [i for i in classes],
                     columns = [i for i in classes])
plt.figure(figsize = (12,7))
sn.heatmap(df_cm, annot=True)