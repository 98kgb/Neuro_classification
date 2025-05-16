#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code load calculation result and analysis.

@author: gibaek
"""
import os, sys, glob

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
from brian2 import *
__spyder__ = True
if '__spyder__' in globals():
    del device
prefs.codegen.target = 'cython'  # define prefer engine
path_c = os.path.dirname(os.path.abspath(__file__))
path_p = os.path.dirname(path_c)
path_g = os.path.dirname(path_p)
sys.path.append(path_c)
sys.path.append(path_p)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, accuracy_score
from utils import *
from tqdm import tqdm

#%%

chb_num = '01'
model = 'ML'

data = pd.read_csv(f'{path_g}/data/cadence/chbmit/chb{chb_num}/sim_result_{model}.csv')

caps = ['1e-12', '3e-12', '5e-12', '7e-12', '9e-12'] 

for cap in caps:
    measures = []
    times = []
    for ii in range(1,4):
        if model == 'LIF':
            t = data[f'/Vout_{ii} (C20.c={cap},C12.c={cap},C0.c={cap}) X'].values
            v = data[f'/Vout_{ii} (C20.c={cap},C12.c={cap},C0.c={cap}) Y'].values
        elif model == 'ML':
            t = data[f'/V_K_{ii} (C1.c={cap},C2.c={cap},C16.c={cap}) X'].values
            v = data[f'/V_K_{ii} (C1.c={cap},C2.c={cap},C16.c={cap}) Y'].values
        time = np.array([x for x in t if isinstance(x, (int, float))])
        voltage = np.array([x for x in v if isinstance(x, (int, float))])
            
        times.append(time)
        measures.append(voltage)
        
    color = ['r', 'g', 'b', 'y']
    fig, ax = plt.subplots(3,1, figsize = (15,8))
    
    for ii in range(len(measures)):
        
        ax[ii].plot(times[ii], measures[ii], label = f'Channel {ii}', c = color[ii])
        ax[ii].set_xlabel('time (sec)')
        ax[ii].set_ylabel('Vout (V)')
        ax[ii].set_xlim([2, max(times[ii])])
        
        ax[ii].axvspan(0, max(times[ii])/2, color='red', alpha=0.3, label = 'sezure')
        ax[ii].legend()    
    
    plt.suptitle(f'Patient {chb_num}, {model} circuit with {cap}pF cap')
    plt.tight_layout()
    plt.show()
    
    predicts = []
    actuals = []
    
    for jj in tqdm(range(3), desc = 'classifying'):
    
        actual = np.zeros(int(max(times[jj])/2))
        predict = np.zeros(int(max(times[jj])/2))
        
        # define actual value
        time_window = 2
        for ii in range(actual.shape[0]):
            if ii < actual.shape[0]/2:
                actual[ii] = 1
                
        # define prediction value
        threshold = -2
        signal = measures[jj]
        for ii in range(predict.shape[0]):
            start = ii * time_window
            end = (ii + 1) * time_window
            mask = (times[jj] >= start) & (times[jj] < end)
            window_signal = signal[mask]
            window_signal = pd.Series(window_signal)
            spike_count = ((window_signal.shift(1, fill_value=0) > threshold) & (window_signal <= threshold)).sum()
            predict[ii] = spike_count
        
        for ii in range(predict.shape[0]):
            if predict[ii]>0:
                predict[ii] = 1
            else:
                predict[ii] = 0
        
        actuals.append(actual)
        predicts.append(predict)
    
    fig, ax = plt.subplots(1, 3, figsize = (14,4.5))
    
    result = np.zeros([3,3])
    for ii in range(3):    
        # confusion matrix
        conf_matrix = confusion_matrix(actuals[ii], predicts[ii])
        
        # visualization
        sns.heatmap(conf_matrix, annot = True, fmt = 'd', cmap = 'Reds', ax = ax[ii])
        ax[ii].set_xlabel("Prediction")
        ax[ii].set_ylabel("Actual")
        ax[ii].set_title(f"Using channel {ii}")
        
        # 지표 계산
        precision = precision_score(actuals[ii], predicts[ii], zero_division=0)
        recall = recall_score(actuals[ii], predicts[ii], zero_division=0)  # = sensitivity
        f1 = f1_score(actuals[ii], predicts[ii], zero_division=0)
        
        result[ii,:] = [precision, recall, f1]
        print(f"🔎 Performance Metrics for measure: {ii+1}")
        print(f"Precision     : {precision:.3f}")
        print(f"Sensitivity   : {recall:.3f}")
        print(f"F1 Score      : {f1:.3f}\n")
    
    plt.suptitle(f'Patient 1, {model} circuit with {cap} capacitance')
    plt.tight_layout()
    plt.show()
    
    df = pd.DataFrame(result, columns=['Precision', 'Sensitivity', 'F1 Score'])
    save_dir = f'{path_g}/data/cadence/chbmit/chb{chb_num}/{model}/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    df.to_csv(f'{save_dir}/Balance_cap_{cap}_classification_result.csv')
    
#%% Hyperparameter comparison
models = ['LIF', 'ML']
chb_num = '02'
caps = ['1e-12', '3e-12', '5e-12', '7e-12', '9e-12'] 

fig, ax = plt.subplots(1,2, figsize = (14,6))
for jj  in range(2):
    score = np.zeros([3,len(caps)])
    save_dir = f'{path_g}/data/cadence/chbmit/chb{chb_num}/{models[jj]}/'
    for idx, cap in enumerate(caps):
        temp = pd.read_csv(f'{save_dir}/Balance_cap_{cap}_classification_result.csv')
        score[:,idx] = temp['F1 Score']
    
    caps_plot = np.linspace(1,9,5)
    color = ['r', 'g', 'b']
    

    for ii in range(3):
        ax[jj].plot(caps_plot, score[ii,:], label = f'channel {ii+1}', c = color[ii], alpha = 0.8, marker = 'o')
        ax[jj].set_xlabel('Capacitance (pF)')
        ax[jj].set_ylabel('F1 score')
        ax[jj].set_title(f'{models[jj]} model optimization', fontsize = 16)
        ax[jj].legend()
        ax[jj].grid()
plt.suptitle(f'Patient {chb_num} diagnosis', fontsize = 18)
plt.tight_layout()
#%% Classification with each optimal system

chb_num = '01'
models = ['LIF', 'ML']
chs = [2,3] # for each model
caps = ['3e-12', '5e-12']
actuals, predicts = [], []
times, voltages = [], []


for idx, model in enumerate(models):
    data = pd.read_csv(f'{path_g}/data/cadence/chbmit/chb{chb_num}/sim_result_{model}.csv')
    
    cap = caps[idx]
    ch = chs[idx]
    
    if model == 'LIF':
        t = data[f'/Vout_{ch} (C20.c={cap},C12.c={cap},C0.c={cap}) X'].values
        v = data[f'/Vout_{ch} (C20.c={cap},C12.c={cap},C0.c={cap}) Y'].values
    elif model == 'ML':
        t = data[f'/V_K_{ch} (C1.c={cap},C2.c={cap},C16.c={cap}) X'].values
        v = data[f'/V_K_{ch} (C1.c={cap},C2.c={cap},C16.c={cap}) Y'].values
    
    time = np.array([x for x in t if isinstance(x, (int, float))])
    voltage = np.array([x for x in v if isinstance(x, (int, float))])
    
    times.append(time)
    voltages.append(voltage)
    
    actual = np.zeros(int(max(time)/2))
    predict = np.zeros(int(max(time)/2))

    # define actual value
    time_window = 2
    for ii in range(actual.shape[0]):
        if ii < actual.shape[0]/2:
            actual[ii] = 1
            
    # define prediction value
    threshold = -2
    signal = voltage
    for ii in range(predict.shape[0]):
        start = ii * time_window
        end = (ii + 1) * time_window
        mask = (time >= start) & (time < end)
        window_signal = signal[mask]
        window_signal = pd.Series(window_signal)
        spike_count = ((window_signal.shift(1, fill_value=0) > threshold) & (window_signal <= threshold)).sum()
        predict[ii] = spike_count

    for ii in range(predict.shape[0]):
        if predict[ii]>0:
            predict[ii] = 1
        else:
            predict[ii] = 0
    
    actuals.append(actual)
    predicts.append(predict)
#%%

fig, ax = plt.subplots(2,1, figsize = (15,10))
for idx in range(2):
    ax[idx].plot(times[idx], voltages[idx], label = 'Output signal', c = 'b', alpha = 0.2)
    ax[idx].set_xlabel('time (sec)', fontsize = 16)
    ax[idx].set_ylabel('Vout (V)', fontsize = 16)
    ax[idx].set_xlim([2, max(time)])
    ax[idx].set_title([2, max(time)])
    ax[idx].set_title(f'{models[idx]} feature extraction result', fontsize = 18)
    
    ax[idx].axvspan(0, max(time)/2, color='red', alpha=0.3, label = 'Seizure period')
    ax[idx].legend()

plt.tight_layout()
plt.show()

#%% confusion matrix
fig, ax = plt.subplots(1,2, figsize = (12,6))
acc = np.zeros([2,3])
for ii in range(2):
    conf_matrix = confusion_matrix(actuals[ii], predicts[ii])
    
    # visualization
    sns.heatmap(conf_matrix, annot = True, fmt = 'd', cmap = 'Reds', ax = ax[ii])
    ax[ii].set_xlabel("Prediction")
    ax[ii].set_ylabel("Actual")
    ax[ii].set_title(f"Patient {chb_num}    detection with {models[ii]}")
    
    # 지표 계산
    precision = precision_score(actuals[ii], predicts[ii], zero_division=0)
    recall = recall_score(actuals[ii], predicts[ii], zero_division=0)  # = sensitivity
    f1 = f1_score(actuals[ii], predicts[ii], zero_division=0)
    acc[ii,:] = [precision, recall, f1]
    print(f"🔎 Performance Metrics for measure: {ii+1}")
    print(f"Precision     : {precision:.3f}")
    print(f"Sensitivity   : {recall:.3f}")
    print(f"F1 Score      : {f1:.3f}\n")
    
    
    df = pd.DataFrame(result, columns=['Precision', 'Sensitivity', 'F1 Score'])

plt.show()

met = ['Precision', 'Recall', 'F1']
plt.figure(1)
x = np.arange(len(met))
width = 0.3
fontsize = 12
plt.bar(x        , acc[0,:], width, label='LIF', color='green', alpha = 0.5)
plt.bar(x + width, acc[1,:], width, label='ML', color='red', alpha = 0.5)

plt.xticks(x, met, fontsize = fontsize)
plt.ylabel('Metric value', fontsize = fontsize)
plt.ylim([0,1])
plt.grid(True, axis='y')
plt.legend()




