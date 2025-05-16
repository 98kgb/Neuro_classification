#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code load calculation result and analysis.

@author: gibaek
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os, sys, glob
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

chb_num = '01'
folder_path = f'{path_g}/physionet.org/files/chbmit/1.0.0/chb{chb_num}/'
edf_files = sorted(glob.glob(os.path.join(folder_path, '*.edf')))
seizure_dict = parse_summary_file(f'{folder_path}/chb{chb_num}-summary.txt', verbose= False)

# Define feature extraction subject
timing = []
for file_num in range(4):
    file = edf_files[file_num]
    save_dir = f"/home/gibaek/data/cadance/chbmit/chb{chb_num}/File{file_num+1}"
    
    file_basename = os.path.basename(file)
    sz_timing = seizure_dict.get(file_basename)
    print(f'\nLoading... {file_basename}')
    timing.append(sz_timing)
    if sz_timing:
        print(f"\n{file_basename}: seizure interval: {sz_timing[0]}\n")
    else:
        print(f"\n{file_basename} has no seizure information.\n")

#%%
cap = 5
model = 'AH'
data = pd.read_csv(f'{path_g}/data/cadence/chbmit/chb{chb_num}/{model}_01_to_04_Cap_{cap}pF.csv')

if model == 'AH':
    t = data['/Vout0 X']
    
    meas_0 = data['/Vout0 Y']
    meas_1 = data['/Vout1 Y']
    meas_2 = data['/Vout2 Y']
    meas_3 = data['/Vout3 Y']
elif model == 'ML':
    t = data['/V_K0 X']
    
    meas_0 = data['/V_K0 Y']
    meas_1 = data['/V_K1 Y']
    meas_2 = data['/V_K2 Y']
    meas_3 = data['/V_K3 Y']

measures = [meas_0, meas_1, meas_2, meas_3]
color = ['r', 'g', 'b', 'y']
fig, ax = plt.subplots(4,1, figsize = (15,8))


for ii in range(len(measures)):
    ax[ii].plot(t, measures[ii], label = f'Measure {ii+1}', c = color[ii])
    ax[ii].set_xlabel('time (sec)')
    ax[ii].set_ylabel('Vout (V)')
    ax[ii].set_xlim([2, 3600])
    
    if timing[ii]:
        for start, end in timing[ii]:
            ax[ii].axvspan(start, end, color='red', alpha=0.3, label = 'sezure')
    ax[ii].legend()    

plt.suptitle(f'Channel: T8-P8, {model} circuit with {cap}pF cap')
plt.tight_layout()
plt.show()

predicts = []
actuals = []

for measure in tqdm(range(4), desc = 'classifying'):

    actual = np.zeros(1800)
    predict = np.zeros(1800)
    
    # define actual value
    time_window = 2
    if timing[measure]:
        for ii in range(actual.shape[0]):
            start = ii * time_window
            end = (ii + 1) * time_window
            for (sz_start, sz_end) in timing[measure]:
                if end > sz_start and start < sz_end:
                    actual[ii] = 1
                    break
    
    # define prediction value
    threshold = -2
    signal = measures[measure]
    for ii in range(predict.shape[0]):
        start = ii * time_window
        end = (ii + 1) * time_window
        mask = (t >= start) & (t < end)
        window_signal = signal[mask]
        spike_count = ((window_signal.shift(1, fill_value=0) > threshold) & (window_signal <= threshold)).sum()
        predict[ii] = spike_count
    
    for ii in range(predict.shape[0]):
        if predict[ii]>0:
            predict[ii] = 1
        else:
            predict[ii] = 0
    
    actuals.append(actual)
    predicts.append(predict)
    
fig, ax = plt.subplots(1,4, figsize = (16,4.5))

result = np.zeros([4,3])
for ii in range(4):    
    # confusion matrix
    conf_matrix = confusion_matrix(actuals[ii], predicts[ii])
    
    # visualization
    sns.heatmap(conf_matrix, annot = True, fmt = 'd', cmap = 'Reds', ax = ax[ii])
    ax[ii].set_xlabel("Predict number")
    ax[ii].set_ylabel("Actual number")
    ax[ii].set_title(f"measure {ii+1}")
    
    # 지표 계산
    precision = precision_score(actuals[ii], predicts[ii], zero_division=0)
    recall = recall_score(actuals[ii], predicts[ii], zero_division=0)  # = sensitivity
    f1 = f1_score(actuals[ii], predicts[ii], zero_division=0)
    
    result[ii,:] = [precision, recall, f1]
    print(f"🔎 Performance Metrics for measure: {ii+1}")
    print(f"Precision     : {precision:.3f}")
    print(f"Sensitivity   : {recall:.3f}")
    print(f"F1 Score      : {f1:.3f}\n")

plt.suptitle(f'Channel: T8-P8, {model} circuit with {cap} capacitance')
plt.tight_layout()
plt.show()

df = pd.DataFrame(result, columns=['Precision', 'Sensitivity', 'F1 Score'],
                  index = ['Measure1', 'Measure2', 'Measure3', 'Measure4'])

df.to_csv(f'{path_g}/data/cadence/chbmit/chb{chb_num}/{model}_01_to_04_Cap_{cap}pF_result.csv')
