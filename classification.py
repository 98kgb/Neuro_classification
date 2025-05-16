#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, glob
import pandas as pd
import numpy as np
import mne
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- Setup ---
path_c = os.path.dirname(os.path.abspath(__file__))
path_p = os.path.dirname(path_c)
path_g = os.path.dirname(path_p)
sys.path.append(path_c)
from utils import parse_summary_file, corr_cal

#%% Load data
chb_num = '01'
folder_path = f'{path_g}/physionet.org/files/chbmit/1.0.0/chb{chb_num}/'
edf_files = sorted(glob.glob(os.path.join(folder_path, '*.edf')))
seizure_dict = parse_summary_file(f'{folder_path}/chb{chb_num}-summary.txt', verbose=False)

file = edf_files[3]
file_basename = os.path.basename(file)
raw = mne.io.read_raw_edf(file, verbose='error')

ch_names = ['T8-P8-1', 'FT10-T8', 'FP2-F4']
ch_name = ch_names[0]
norm_current = 5e-9
selected_channels = raw.ch_names
tau_list = [10,20,40,60,100]
c_m_list = [10,50,100,200,400]
aa = 0
bb = 2
norm = True

# ch_names = ['T8-P8-1', 'FT10-T8', 'FP2-F4', 'F8-T8']
ch_names = ['FT10-T8']

model_name = 'LIF' # LIF or ML

result = []
for ch_name in ch_names:
    file_name = f'spike_train_chb{chb_num}_{file_basename.replace(".edf", "")}_ch_name_{ch_name}_norm_I_{norm_current}'

    result.append(np.load(f'./spike_data/{file_name}_{model_name}_{tau_list[aa]}.npy', allow_pickle=True))
    

prediction = []
threshold = 0
for ii in range(result[0].shape[0]):
    spike_num = 0
    for jj in range(len(ch_names)):
        spike_num += len(result[jj][ii])
    if spike_num > threshold:
        prediction.append(1)
    else:
        prediction.append(0)
prediction = np.array(prediction)

Label = np.load(f'./spike_data/spike_train_chb{chb_num}_{file_basename.replace(".edf", "")}_label.npy')



from sklearn.metrics import confusion_matrix, accuracy_score

# confusion matrix
conf_matrix = confusion_matrix(Label, prediction)
accuracy = accuracy_score(Label, prediction)

# visualization

plt.figure(2)
font = {'size': 15}
plt.rc('font', **font)
sns.heatmap(conf_matrix, annot = True, fmt = 'd', cmap = 'Reds')
plt.xlabel("Predict number")
plt.ylabel("Actual number")
plt.title(f"{model_name} model   Accuracy: {accuracy:.4f}")
print("\nAccuracy: ", accuracy)


