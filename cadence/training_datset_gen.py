#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file convert raw EEG signals by upscaling, and spliting.

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

chb_num = '01'
folder_path = f'{path_g}/physionet.org/files/chbmit/1.0.0/chb{chb_num}/'
edf_files = sorted(glob.glob(os.path.join(folder_path, '*.edf')))
seizure_dict = parse_summary_file(f'{folder_path}/chb{chb_num}-summary.txt', verbose= False)

for file_num in range(5):
    file = edf_files[file_num]
    
    if file_num<10:
        save_dir = f"/home/gibaek/data/cadence/chbmit/chb{chb_num}/File0{file_num+1}"
    else:
        save_dir = f"/home/gibaek/data/cadence/chbmit/chb{chb_num}/File{file_num+1}"
    
    file_basename = os.path.basename(file)
    sz_timing = seizure_dict.get(file_basename)
    print(f'\nLoading... {file_basename}')
    if sz_timing:
        print(f"\n{file_basename}: seizure interval: {sz_timing[0]}\n")
    else:
        print(f"\n{file_basename} has no seizure information.\n")
        
    # dividing
    
    fs = 256
    upsample_factor = 100
    norm_current =  1e-9 # 1nA
    
    raw = mne.io.read_raw_edf(file, verbose = 'error') # load EEG dataset
    verbose = False
    ch_names = ['T8-P8-1', 'FT10-T8', 'FP2-F4']
    # ch_names = ['T8-P8-1']
    for ch_name in ch_names:
    
        file_name = f'chb{chb_num}_{file_basename.replace(".edf", "")}_ch_name_{ch_name}_norm_I_{norm_current}'
        
        if f'{file_name}_0sec_to_3600sec.txt' in os.listdir(save_dir):
            print('File already exsit')
            continue
        
        times_upsampled, I_stim, _ = EEG_to_Current(raw = raw, channel_name = ch_name, upsample_factor = upsample_factor,
                                                    fs = fs, norm_current = norm_current, verbose = False)
        
        time_window = 3600 # 5 sec
        epoch_len = fs * upsample_factor * time_window
        num_epoch = int(I_stim.shape[0]/epoch_len)
        signals = np.zeros([epoch_len,2])
            
        for ii in range(num_epoch):
            
            I_cal = I_stim[fs * upsample_factor * time_window * ii:fs * upsample_factor * time_window * (ii+1)] 
            t_cal = times_upsampled[fs * upsample_factor * time_window * 0:fs * upsample_factor * time_window * (1)] # t_cal should be same for simulataneous simulation in cadance..
            signals[:, 0] = t_cal
            signals[:, 1] = I_cal
        
            if os.path.isdir(save_dir):
                np.savetxt(f'{save_dir}/{file_name}_{ii*time_window}sec_to_{(ii+1)*time_window}sec.txt', delimiter = ' ', X = signals)
            else:
                os.makedirs(save_dir)
                np.savetxt(f'{save_dir}/{file_name}_{ii*time_window}sec_to_{(ii+1)*time_window}sec.txt', delimiter = ' ', X = signals)
