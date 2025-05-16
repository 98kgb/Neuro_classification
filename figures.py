#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 10:13:18 2025

@author: gibaek
"""

import os, sys, glob
import pandas as pd
import numpy as np
import mne
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
# from scipy.stats import pointbiserialr, ttest_ind

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

file = edf_files[4]
file_basename = os.path.basename(file)
raw = mne.io.read_raw_edf(file, verbose='error')

# visulize raw EEG signal

# segment = raw.copy().crop(tmin = 2996, tmax = 3036)
ch_name = raw.ch_names[0]
EEG, times = raw[ch_name, :]
EEG *= 1e6
scaling = len(EEG[0])
idx = int(len(EEG[0])/3600*2)

fs = 256
upsample_factor = 100
norm_current = 1e-5
eeg_centered = EEG - np.mean(EEG) # center align

nyq = fs/2 # nyquist frequency
b, a = butter(4, [1/nyq, 40/nyq], btype='band') # only consider frequency between 1 and 40 Hz
eeg_filtered = filtfilt(b, a, eeg_centered) # filter the raw EEG

eeg_upsampled = np.repeat(eeg_filtered, upsample_factor) # increase time resolution
dt_original = 1 / fs
dt_upsampled = dt_original / upsample_factor
total_samples = len(eeg_upsampled)
times_upsampled = np.arange(total_samples) * dt_upsampled

I_stim = eeg_upsampled * norm_current  # unit: A
#%%
fig, ax = plt.subplots(1,3, figsize = (15,5))

ax[0].plot(times, EEG[0], linewidth = .1, c = 'b', alpha = 0.9)
ax[0].set_xlabel('Time (s)')
ax[0].set_ylabel('Amplitude (μV)')
ax[0].set_title(f'Channel {ch_name}')

ax[1].plot(times[:idx], EEG[0][:idx], linewidth = 1, c = 'b', alpha = 0.9)
ax[1].set_xlabel('Time (s)')
ax[1].set_ylabel('Amplitude (μV)')
ax[1].set_title(f'Channel {ch_name} Epoch 1')

ax[2].plot(times_upsampled[:idx*upsample_factor], I_stim[:idx*upsample_factor],
           linewidth = 1, c = 'b', alpha = 0.9)
ax[2].set_xlabel('Time (s)')
ax[2].set_ylabel('Amplitude (nA)')
ax[2].set_title(f'Channel {ch_name} Epoch 1')

plt.tight_layout()
