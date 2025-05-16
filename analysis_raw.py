import os, sys, glob, re
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

def rank(chb_num, ch_names, model_name, base, mean = True):
    # Init min/max trackers
    t_min, t_max = np.ones(5) * 10, np.zeros(5)
    mi_min, mi_max = np.ones(5), np.zeros(5)
    r_min, r_max = np.ones(5), np.zeros(1)
    print("Computing min/max per feature...")
    for ch_name in tqdm(ch_names):
        df = pd.read_csv(f'{path_c}/raw_data/chb{chb_num}/ch_{ch_name}.csv')
        df = corr_cal(df)
        
        if df[['r', 't', 'auc', 'mi']].isnull().values.any():
            continue
        
        r = np.abs(df['r'].values)
        t = np.abs(df['t'].values)
        mi = df['mi'].values
        
        r_max = np.maximum(r_max, r)
        r_min = np.minimum(r_min, r)
        t_max = np.maximum(t_max, t)
        t_min = np.minimum(t_min, t)
        
        mi_max = np.maximum(mi_max, mi)
        mi_min = np.minimum(mi_min, mi)
    
    # Initialize metric matrix
    metrics = np.zeros([len(ch_names), 4])
    
    print("Computing normalized average metrics per channel...")
    
    for idx, ch_name in enumerate(tqdm(ch_names)):
        df = pd.read_csv(f'{path_c}/raw_data/chb{chb_num}/ch_{ch_name}.csv')
        df = corr_cal(df)
        
        r = np.mean(np.abs(df['r'].values))
        t = np.mean(np.abs(df['t'].values))
        auc = np.mean(df['auc'].values)  # AUC is already in 01
        mi = np.mean(df['mi'].values)
                 
        metrics[idx, :] = [r, t, auc, mi]
    
    return metrics


def analysis_EEG(chb_num, plot = True):
    # Load data
    folder_path = f'{path_g}/physionet.org/files/chbmit/1.0.0/chb{chb_num}/'
    edf_files = sorted(glob.glob(os.path.join(folder_path, '*.edf')))
    
    file = edf_files[0]
    raw = mne.io.read_raw_edf(file, verbose='error')
    
    # visulize raw EEG signal
    segment = raw.copy().crop(tmin = 2996, tmax = 3036)
    segment.plot(n_channels = 5, scalings = 1e-4)
    
    # Channel importance
    selected_channels = raw.ch_names
    metric_raw = rank(chb_num = chb_num, ch_names= selected_channels, model_name= 'raw', base = 1e-4)
    metric_names = ['r', 't', 'AUC', 'MI']
    df_raw = pd.DataFrame(metric_raw, columns=metric_names, index=selected_channels)

    fig, ax = plt.subplots(1, 3, figsize =   (10,10))
    df_r = df_raw.sort_values('r', ascending=False)[:10]
    df_AUC = df_raw.sort_values('AUC', ascending=False)[:10]
    df_MI = df_raw.sort_values('MI', ascending=False)[:10]
    
    sns.heatmap(df_r[['r']], annot=True, annot_kws={"size": 12}, fmt=".2f",
                cmap="YlGnBu", cbar=False, linewidths=0.5, linecolor='gray', ax = ax[0])
    sns.heatmap(df_AUC[['AUC']], annot=True, annot_kws={"size": 12}, fmt=".2f",
                cmap="YlGnBu", cbar=False, linewidths=0.5, linecolor='gray', ax = ax[1])
    sns.heatmap(df_MI[['MI']], annot=True, annot_kws={"size": 12}, fmt=".2f",
                cmap="YlGnBu", cbar=False, linewidths=0.5, linecolor='gray', ax = ax[2])
    for ii in range(3):
        ax[ii].tick_params(axis='both', labelsize=13)
    fig.suptitle('Top 10 correlated channels of each metrics', fontsize = 15)
    plt.tight_layout()
    
    r_top = df_r.index[:5]
    AUC_top = df_AUC.index[:5]
    MI_top = df_MI.index[:5]
    
    count = np.zeros(len(raw.ch_names))
    for idx, ch_name in enumerate(raw.ch_names):
        if ch_name in r_top:
            count[idx] += 1
        if ch_name in AUC_top: 
            count[idx] += 1
        if ch_name in MI_top: 
            count[idx] += 1
    
    plt.figure(figsize = (10,5))
    plt.bar(raw.ch_names, count, alpha = 0.5)
    plt.xticks(fontsize = 12)
    plt.yticks([0,1,2,3], fontsize = 12)
    plt.ylabel('Appearance in Top 6 list', fontsize = 15)
    plt.tick_params(rotation= 45)
    plt.show()
    top_indices = np.argsort(count)[-3:][::-1]
    top_channels = [raw.ch_names[i] for i in top_indices]
    
    print("Top 3 channels:", top_channels)
    
    return top_channels
