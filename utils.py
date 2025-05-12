#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code load CHB-MIT dataset (EEG;Electroencephalography) signal and analysis it.

@author: gibaek
"""

import os, sys
import glob
import mne
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

path_c = os.path.dirname(os.path.abspath(__file__))
path_p = os.path.dirname(path_c)
path_g = os.path.dirname(path_p)
sys.path.append(path_c)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.stats import pointbiserialr, ttest_ind
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import mutual_info_classif

def EEG_to_Current(raw, channel_name = 'FP1-F7', fs = 256, norm_current=1e-9, upsample_factor = 100, verbose = True):
    
    if channel_name in raw.ch_names:
        eeg_signal, times = raw[channel_name, :]
    
        if verbose:
            print(f"min: {np.min(eeg_signal)*1e6} μV, max: {np.max(eeg_signal)*1e6} μV")
    
        eeg_signal *= 1e3 # convert μV to mV
        eeg_centered = eeg_signal - np.mean(eeg_signal) # center align
        
        nyq = fs/2 # nyquist frequency
        b, a = butter(4, [1/nyq, 40/nyq], btype='band') # only consider frequency between 1 and 40 Hz
        eeg_filtered = filtfilt(b, a, eeg_centered) # filter the raw EEG
        
        # Upsampling time and intensity
        eeg_upsampled = np.repeat(eeg_filtered, upsample_factor) # increase time resolution
        dt_original = 1 / fs
        dt_upsampled = dt_original / upsample_factor
        total_samples = len(eeg_upsampled)
        times_upsampled = np.arange(total_samples) * dt_upsampled
        
        I_stim = eeg_upsampled * norm_current  # unit: A
        
        return times_upsampled, I_stim, True
    
    else:
        print(f'No channel name \'{channel_name}\' in {raw}')
        
        return False, False, False
        
def parse_summary_file(summary_path, verbose = True):
    """
    Parsing summary file and convert seizure interval to dictionary
    Parameters
    ----------
    summary_path : str
        'chbxx-summary.txt' file path

    Returns
    -------
    seizure_dict : dict
        {'chb01_03.edf': [(start1, end1), ...], ...}
    """
    seizure_dict = {}
    with open(summary_path, 'r') as f:
        lines = f.readlines()

    current_file = None
    seizure_list = []
    
    for line in lines:
        line = line.strip()
        if line.startswith('File Name:'):
            if current_file and seizure_list:
                seizure_dict[current_file] = seizure_list
            current_file = line.split(':')[-1].strip()
            seizure_list = []
        elif line.startswith('Seizure'):
            parts = line.split(':')
            time_val = float(parts[-1].strip().split()[0])
            if 'Start' in line:
                start_time = time_val
            elif 'End' in line:
                end_time = time_val
                seizure_list.append((start_time, end_time))

    if current_file and seizure_list:
        seizure_dict[current_file] = seizure_list
    
    if verbose:
        print("EDF file(s) including seizures:")
        for fname in sorted(seizure_dict.keys()):
            print(f"  - {fname} → {len(seizure_dict[fname])} seizure(s)")

    return seizure_dict

def get_epoch_labels(seizure_times, total_duration, num_epoch, epoch_len):
    """
    seizure_times : list of (start, end) tuples
    total_duration : total signal duration (sec)
    num_epoch : total number of window
    epoch_len : epoch length (sec)

    return : [0, 0, 1, 0, 1, ...] format epoch label numpy array
    """

    labels = []
    if seizure_times:
        for ii in range(num_epoch):
            start = total_duration[ii * epoch_len]
            end = total_duration[(ii + 1) * epoch_len-1]
            label = 0
            for (sz_start, sz_end) in seizure_times:
                if end > sz_start and start < sz_end:
                    label = 1
                    break
            labels.append(label)
        
        labels = np.array(labels)
        
    else:
        labels = np.zeros(num_epoch)
        print('No seizure existed.')
        
    return labels

def extract_eeg_features(raw_signal, times, num_epochs, epoch_len, sz_labels, threshold=0.2):
    """
    Extract features from raw EEG signal that mirror those computed from the Morris-Lecar neuron model.

    Parameters
    ----------
    raw_signal : np.ndarray
        1D array of EEG voltage data (single channel), in mV.
    times : np.ndarray
        Time vector corresponding to the EEG signal, in seconds.
    fs : float
        Sampling rate of the EEG signal (Hz).
    threshold : float
        Voltage threshold (in mV) used to count "spike-like" events.
    epoch_len_sec : float
        Length of each epoch (window) in seconds.

    Returns
    -------
    features : list of dict
        A list of dictionaries, one per epoch, each containing:
        - spike_count_raw
        - isi_mean_raw
        - isi_std_raw
        - v_var_raw
        - dvdt_max_raw
    """
    
    features = []
    
    for ii in range(num_epochs):
        start = ii * epoch_len
        end = (ii + 1) * epoch_len
        sig = raw_signal[start:end]
        t = times[start:end]
        
        # Detect upward threshold crossings (spike-like events)
        spike_inds = np.where((sig[1:] > threshold) & (sig[:-1] <= threshold))[0]
        spike_times = t[spike_inds]
        spike_count = len(spike_inds)
        
        # Compute ISI features
        if spike_count >= 2:
            isi = np.diff(spike_times)
            isi_mean = np.mean(isi)
            isi_std = np.std(isi)
        else:
            isi_mean = 0
            isi_std = 0
        
        # Voltage variance
        v_var = np.var(sig)
        
        # Maximum derivative (dV/dt)
        dvdt = np.diff(sig) / np.diff(t)
        dvdt_max = np.max(np.abs(dvdt))
        
        features.append({
            'epoch': ii,
            'spike_count': spike_count,
            'isi_mean': isi_mean,
            'isi_std': isi_std,
            'v_var': v_var,
            'dvdt_max': dvdt_max,
            'label': sz_labels[ii]
        })
        
    return features

def corr_cal(dataframe):
    """
    Calculate correlation between extracted features and label.

    Parameters
    ----------
    dataframe : pandas.dataframe
        Contains features and label of each window (epoch)
    Returns
    -------
    dataframe: pandas.dataframe
        Contains r, p_r, t, p_t, AUC, MI
    """
    results = []
    
    for column in dataframe.columns:
        if column not in ['epoch', 'label']:
            try:
                x = dataframe[column]
                y = dataframe['label']
    
                # Correlation
                r, p_r = pointbiserialr(x, y)
    
                # t-test
                t, p_t = ttest_ind(x[y == 1], x[y == 0], equal_var=False)
                
                # AUC
                auc = roc_auc_score(y, x)
    
                # results.append({
                #     'feature': column,
                #     'r': r, 'p_r': p_r,
                #     't': t, 'p_t': p_t,
                #     'auc': auc})
                results.append({
                    'feature': column,
                    'r': r, 't': t,
                    'auc': auc})
                
            except Exception as e:
                print(f"Error on feature {column}: {e}")
    
    # Mutual Information
    feature_names = [col for col in dataframe.columns if col not in ['epoch', 'label']]
    
    X = dataframe[feature_names]
    y = dataframe['label']
    
    mi = mutual_info_classif(X, y, discrete_features=False, random_state=42)
    
    # Add MI to results
    for i, feat in enumerate(feature_names):
        results[i]['mi'] = mi[i]
        
    # --- Save as DataFrame ---
    df = pd.DataFrame(results)
    df = df.set_index('feature')
    
    return df

#%% visulization and Analysis
if __name__ == '__main__':
    chb_num = '01'
    folder_path = f'{path_g}/physionet.org/files/chbmit/1.0.0/chb{chb_num}/'
    edf_files = sorted(glob.glob(os.path.join(folder_path, '*.edf')))
    # Load EEG info for chb_num.
    seizure_dict = parse_summary_file(f'{folder_path}/chb{chb_num}-summary.txt', verbose= False)

    # Define feature extraction subject
    file = edf_files[2]
    file_basename = os.path.basename(file)
    sz_timing = seizure_dict.get(file_basename)
    print(f'\nLoading... {file_basename}')
    if sz_timing:
        print(f"\n{file_basename}: seizure interval: {sz_timing[0]}\n")
    else:
        print(f"\n{file_basename} has no seizure information.\n")
        
    # Convert it to current signal
    fs = 256
    upsample_factor = 100
    raw = mne.io.read_raw_edf(file, verbose = 'error') # load EEG dataset
    verbose = True
    ch_names = raw.ch_names

    times_upsampled, I_stim, EEG = EEG_to_Current(raw = raw, channel_name= ch_names[0],
                                                             upsample_factor = upsample_factor,
                                                             norm_current=1e-9)  
    eeg_signal, eeg_centered, eeg_filtered, times = EEG[0], EEG[1], EEG[2], EEG[3]

    
    t_step, inter_num = 256, 200
    
    fig, ax = plt.subplots(1,3, figsize = (15,6))
    fontsize = 15
    
    ax[0].plot(times[t_step*inter_num:t_step*(inter_num+1)],
             eeg_signal[0,t_step*inter_num:t_step*(inter_num+1)],
             '--',alpha = 1, label = 'raw')
    ax[0].plot(times[t_step*inter_num:t_step*(inter_num+1)],
             eeg_centered[0,t_step*inter_num:t_step*(inter_num+1)],
             alpha = 0.5, label = 'centered')
    ax[0].plot(times[t_step*inter_num:t_step*(inter_num+1)],
             eeg_filtered[0,t_step*inter_num:t_step*(inter_num+1)],
             alpha = 0.5, label = 'filtered')
    ax[0].set_xlabel('$time (sec)$', fontsize = fontsize)
    ax[0].set_ylabel('$EEG\ signal\ (mV)$', fontsize = fontsize)
    ax[0].grid()
    ax[0].legend()
    
    
    ax[1].plot(times_upsampled[t_step*upsample_factor*inter_num:t_step*upsample_factor*(inter_num+1)],
             I_stim[t_step*upsample_factor*inter_num:t_step*upsample_factor*(inter_num+1)]*1e9,
             c = 'b', alpha = 0.7)
    ax[2].plot(times_upsampled[t_step*upsample_factor*inter_num:t_step*upsample_factor*(inter_num)+7000],
             I_stim[t_step*upsample_factor*inter_num:t_step*upsample_factor*(inter_num)+7000]*1e9,
             c = 'b', alpha = 0.7)
    
    for ii in range(1,3):
        ax[ii].set_title("Converted Current Stimulus", fontsize = fontsize)
        ax[ii].set_xlabel('$time (sec)$', fontsize = fontsize)
        ax[ii].set_ylabel("Current (nA)", fontsize = fontsize)
        ax[ii].grid()
        
    plt.tight_layout()
    

    
    
