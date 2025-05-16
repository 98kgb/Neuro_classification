#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 12:34:42 2025

@author: gibaek
"""

import os, sys
import numpy as np
import mne
import matplotlib.pyplot as plt
import scipy.signal as signal
from brian2 import *
__spyder__ = True

if '__spyder__' in globals():
    del device

from Neurons import *
from tqdm import tqdm

prefs.codegen.target = 'cython'  # define prefer engine
path_c = os.path.dirname(os.path.abspath(__file__))
path_p = os.path.dirname(path_c)
path_g = os.path.dirname(path_p)
sys.path.append(path_c)

from utils import *


def generate_dataset(chb_num, ch_names, verbose):
    folder_path = f'{path_g}/physionet.org/files/chbmit/1.0.0/chb{chb_num}/'
    edf_files = sorted(glob.glob(os.path.join(folder_path, '*.edf')))
    seizure_dict = parse_summary_file(f'{folder_path}/chb{chb_num}-summary.txt', verbose= False)
    plt.figure(1)
    for ch_name in ch_names:
        aa, bb = 0, 0
        sz_signal = []
        nsz_signal = []
    
        for file_num in tqdm(range(len(edf_files)), desc = f'Transforming channel {ch_name}'):
            
            file = edf_files[file_num]
            
            save_dir = f"/home/gibaek/data/cadence/chbmit/chb{chb_num}"
            
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                
            file_basename = os.path.basename(file)
            sz_timing = seizure_dict.get(file_basename)
            
            fs = 256
            upsample_factor = 10
            norm_current =  1e-9 # 1nA
            
            raw = mne.io.read_raw_edf(file, verbose = 'error') # load EEG dataset
            verbose = False
            
            file_name = f'ch_name_{ch_name}_balance'
            
            if f'{file_name}' in os.listdir(save_dir):
                print('File already exsit')
                continue
            
            times_upsampled, I_stim, valid = EEG_to_Current(raw = raw, channel_name = ch_name, upsample_factor = upsample_factor,
                                                        fs = fs, norm_current = norm_current, verbose = False)
            if valid:
                time_window = 2 # 5 sec
                epoch_len = fs * upsample_factor * time_window
                num_epoch = int(I_stim.shape[0]/epoch_len)
                
                if sz_timing: # Check if there is seizure and channel info
                    for ii in range(num_epoch):
                        I_cal = I_stim[fs * upsample_factor * time_window * ii:fs * upsample_factor * time_window * (ii+1)] 
                        t_cal = times_upsampled[fs * upsample_factor * time_window * ii:fs * upsample_factor * time_window * (ii+1)] # t_cal should be same for simulataneous simulation in cadance..
                        if t_cal[-1] > sz_timing[0][0] and t_cal[0] < sz_timing[0][1]:
                            aa += 1
                            sz_signal.append(I_cal)
                        else:
                            bb += 1
                            nsz_signal.append(I_cal)
            
        print(f"Seizure: {aa} Non-seizure: {bb}")
        
        length = int(fs*len(sz_signal)*time_window*upsample_factor)
        
        final_signal = np.zeros([length*2, 2])
        for ii in range(len(sz_signal)):
            st_idx = ii*len(sz_signal[0])
            fin_idx = (ii+1)*len(sz_signal[0])
            final_signal[st_idx:fin_idx, 0] = np.linspace(ii * 2, (ii+1) * 2 ,2 * fs *10 + 1)[:2 * fs *10]
            final_signal[st_idx:fin_idx, 1] = sz_signal[ii]
        
        for ii in range(len(sz_signal), len(sz_signal) * 2):
            st_idx = ii*len(nsz_signal[0])
            fin_idx = (ii+1)*len(nsz_signal[0])
            final_signal[st_idx:fin_idx, 0] = np.linspace(ii * 2, (ii+1) * 2 ,2 * fs *10 + 1)[:2 * fs *10]
            final_signal[st_idx:fin_idx, 1] = nsz_signal[np.random.randint(len(nsz_signal))]
        
        plt.plot(final_signal[:,0], final_signal[:,1], alpha = 0.5)
        
        np.savetxt(f'{save_dir}/{file_name}.txt', delimiter = ' ', X = final_signal)
    
    plt.show()
        
