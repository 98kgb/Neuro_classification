#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code extract features from raw EEG signals

@author: gibaek
"""

import os, sys, glob, re
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path_c = os.path.dirname(os.path.abspath(__file__))
path_p = os.path.dirname(path_c)
path_g = os.path.dirname(path_p)
sys.path.append(path_c)

from utils import *
from tqdm import tqdm
from brian2 import *
__spyder__ = True
if '__spyder__' in globals():
    del device
    
prefs.codegen.target = 'cython'  # define prefer engine
#%%
def feature_extract(chb_num, verbose = False):
    ## Load raw EEG data
    chb_num = chb_num
    folder_path = f'{path_g}/physionet.org/files/chbmit/1.0.0/chb{chb_num}/'
    edf_files = sorted(glob.glob(os.path.join(folder_path, '*.edf')))
    seizure_dict = parse_summary_file(f'{folder_path}/chb{chb_num}-summary.txt', verbose= False)
    
    # Extraction starts
    for file in edf_files:
        file_basename = os.path.basename(file)
        sz_timing = seizure_dict.get(file_basename)
        print(f'\nLoading... {file_basename}')
        
        # Checking if there is seizure data
        if sz_timing:
            print(f"\n{file_basename}: seizure interval: {sz_timing[0]}\n")
            fs = 256
            raw = mne.io.read_raw_edf(file, verbose = 'error')
            ch_names = raw.ch_names
            if 'ECG' in ch_names:
                ch_names.remove('ECG')
                
            for ch_name in tqdm(ch_names, desc = 'Feature extraction for raw EEG signal'):
                raw_signal, times = raw[ch_name,:]
                raw_signal = raw_signal[0]
                if verbose:
                    print(f'\nMin: {min(raw_signal)}, Max: {max(raw_signal)}, Mean: {np.mean(raw_signal)}')
                
                time_window = 2
                epoch_len = fs * time_window
                num_epoch = int(raw_signal.shape[0]/epoch_len)
                sz_labels = get_epoch_labels(sz_timing, times, num_epoch, epoch_len)
                
                features_raw = extract_eeg_features(raw_signal = raw_signal,
                                                    times = times, num_epochs = num_epoch,
                                                    epoch_len = epoch_len, sz_labels = sz_labels, threshold=0.5e-4)
                df = pd.DataFrame(features_raw)
                if not os.path.exists(f'{path_c}/raw_data/chb{chb_num}'):
                    os.makedirs(f'{path_c}/raw_data/chb{chb_num}')
                df.to_csv(f'{path_c}/raw_data/chb{chb_num}/{file_basename.replace(".edf", "")}_ch_{ch_name}.csv', index=False)
                
                if verbose:
                    print("\n Feature Extraction Completed")
                    print(f"Total epoch: {len(df)}")
                    print(f"Seizure epoch: {np.sum(df['label'] == 1)}")
                    print(f"Non-seizure epoch: {np.sum(df['label'] == 0)}")
    
        else:
            print(f"\n{file_basename} has no seizure information.\n")
    
    
    #%% combine w.r.t channel name
    path = f'{path_c}/raw_data/chb{chb_num}/'
    file_list = os.listdir(path)
    numbers = set()
    for filename in file_list:
        match = re.search(fr'chb{chb_num}_(\d+)_', filename)
        if match:
            numbers.add(match.group(1))
    numbers = sorted(numbers)
    file = edf_files[0]
    raw = mne.io.read_raw_edf(file, verbose = 'error')
    ch_names = raw.ch_names
    if 'ECG' in ch_names:
        ch_names.remove('ECG')
    
    for ch_name in ch_names:
        for ii in range(len(numbers)):
            if f'chb{chb_num}_{numbers[ii]}_ch_{ch_name}.csv' in os.listdir(f'{path_c}/raw_data/chb{chb_num}/'): # check if the file exist
                if ii == 0:
                    tot = pd.read_csv(f'{path_c}/raw_data/chb{chb_num}/chb{chb_num}_{numbers[ii]}_ch_{ch_name}.csv')
                else:
                    temp = pd.read_csv(f'{path_c}/raw_data/chb{chb_num}/chb{chb_num}_{numbers[ii]}_ch_{ch_name}.csv')
                    tot = pd.concat([tot, temp], ignore_index= True)
        tot.to_csv(f'{path_c}/raw_data/chb{chb_num}/ch_{ch_name}.csv', index=False)
    
    
    #%% delete old files
    
    folder_path = f'{path_c}/raw_data/chb{chb_num}/'
    file_list = os.listdir(folder_path)
    for file in file_list:
        if file.startswith(f"chb{chb_num}"):
            full_path = os.path.join(folder_path, file)
            os.remove(full_path)
            print(f"Deleted: {file}")
    
    print(f"EEG feature extraction for patient {chb_num} completed")