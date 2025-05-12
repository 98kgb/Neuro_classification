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

file = edf_files[2]
file_basename = os.path.basename(file)
raw = mne.io.read_raw_edf(file, verbose='error')

# visulize raw EEG signal

segment = raw.copy().crop(tmin = 2996, tmax = 3036)
segment.plot(n_channels = 5, scalings = 1e-4)

#%% Find the best parameters
ch_names = ['T8-P8-1', 'FT10-T8', 'FP2-F4']
ch_name = ch_names[1]
norm_current = 5e-9
tau_list = [10,20,40,60,100]
c_m_list = [10,50,100,200,400]
met_list = ['r', 'auc', 'mi']
c = ['r', 'g', 'b']

fig, ax = plt.subplots(2,3, figsize = (15,9))
for idx, ch_name in enumerate(ch_names):
    LIF_mets = []
    ML_mets = []
    for met in met_list:
        temp = np.zeros([2,5])
        for kk in range(len(tau_list)):
            file_name = f'spike_train_chb{chb_num}_{file_basename.replace(".edf", "")}_ch_name_{ch_name}_norm_I_{norm_current}'
            
            LIF = np.load(f'./spike_data/{file_name}_LIF_{tau_list[kk]}.npy', allow_pickle=True)
            ML = np.load(f'./spike_data/{file_name}_ML_{c_m_list[kk]}.npy', allow_pickle=True)
            Label = np.load(f'./spike_data/spike_train_chb{chb_num}_{file_basename.replace(".edf", "")}_label.npy')
            
            
            def compute_isi_features(spike_times):
                if len(spike_times) < 2:
                    return 0, 0
                isi = np.diff(spike_times)
                return np.mean(isi), np.std(isi)
            
            isi_LIF = [compute_isi_features(sp) for sp in LIF]
            isi_ML = [compute_isi_features(sp) for sp in ML]
            
            # Unpack
            isi_LIF_mean, isi_LIF_std = zip(*isi_LIF)
            isi_ML_mean, isi_ML_std = zip(*isi_ML)
            
            # Convert it to dataframe
            df_LIF = pd.DataFrame({
                'epoch': np.arange(len(LIF)),
                'SpikeCount': [len(sp) for sp in LIF],
                'ISI_Mean': isi_LIF_mean,
                'ISI_Std': isi_LIF_std,
                'label' : Label
            })
            
            df_ML = pd.DataFrame({
                'epoch': np.arange(len(ML)),
                'SpikeCount': [len(sp) for sp in ML],
                'ISI_Mean': isi_ML_mean,
                'ISI_Std': isi_ML_std,
                'label' : Label
            })
            
            # Calculate correlation
            co_LIF = corr_cal(df_LIF)
            co_ML = corr_cal(df_ML)
            info_list = co_LIF.index
            aa = 1
            temp[0, kk] = co_LIF.loc[info_list[aa], met]
            temp[1, kk] = co_ML.loc[info_list[aa], met]
        LIF_mets.append(temp[0,:])
        ML_mets.append(temp[1,:])
    
    for ii in range(len(met_list)):
        ax[0][ii].plot(tau_list, LIF_mets[ii], marker = 'o', c = c[idx], alpha = 0.5, label = ch_name)
        ax[0][ii].set_xlabel('LIF time constant')
        ax[1][ii].plot(c_m_list, ML_mets[ii], marker = 'o', c = c[idx], alpha = 0.5, label = ch_name)
        ax[1][ii].set_xlabel('ML Membrane Capacitance')
        for jj in range(2):
            ax[jj][ii].set_ylabel(f'{met_list[ii]}')
            ax[jj][ii].grid()

fig.legend(ch_names, loc='lower center', ncol=3, fontsize=12,
           bbox_to_anchor=(0.5, -0.05))
plt.suptitle(f'{info_list[aa]} correlation with Seizure Label')
plt.tight_layout()

#%% comparing two models

ch_names = ['T8-P8-1', 'FT10-T8', 'FP2-F4']
ch_name = ch_names[1]
norm_current = 5e-9
selected_channels = raw.ch_names
tau_list = [10,20,40,60,100]
c_m_list = [10,50,100,200,400]
aa = 0
bb = 2
norm = True
file_name = f'spike_train_chb{chb_num}_{file_basename.replace(".edf", "")}_ch_name_{ch_name}_norm_I_{norm_current}'

LIF = np.load(f'./spike_data/{file_name}_LIF_{tau_list[aa]}.npy', allow_pickle=True)
ML = np.load(f'./spike_data/{file_name}_ML_{c_m_list[bb]}.npy', allow_pickle=True)
Label = np.load(f'./spike_data/spike_train_chb{chb_num}_{file_basename.replace(".edf", "")}_label.npy')


def compute_isi_features(spike_times):
    if len(spike_times) < 2:
        return 0, 0
    isi = np.diff(spike_times)
    return np.mean(isi), np.std(isi)

isi_LIF = [compute_isi_features(sp) for sp in LIF]
isi_ML = [compute_isi_features(sp) for sp in ML]

# Unpack
isi_LIF_mean, isi_LIF_std = zip(*isi_LIF)
isi_ML_mean, isi_ML_std = zip(*isi_ML)

# Convert it to dataframe
df_LIF = pd.DataFrame({
    'epoch': np.arange(len(LIF)),
    'SpikeCount': [len(sp) for sp in LIF],
    'ISI_Mean': isi_LIF_mean,
    'ISI_Std': isi_LIF_std,
    'label' : Label
})

df_ML = pd.DataFrame({
    'epoch': np.arange(len(ML)),
    'SpikeCount': [len(sp) for sp in ML],
    'ISI_Mean': isi_ML_mean,
    'ISI_Std': isi_ML_std,
    'label' : Label
})

# Calculate correlation
co_LIF = corr_cal(df_LIF)
co_ML = corr_cal(df_ML)

# features = ['SpikeCount', 'ISI_Mean']
features = ['SpikeCount']
x_label = ['$r_{norm}$', '$AUC_{norm}$', '$MI_{norm}$']
metrics = ['r', 'auc', 'mi']
keys = [f'{m}' for m in metrics]
df_tot = [co_LIF, co_ML]
fig, ax = plt.subplots(1, len(features), figsize=(6*len(features), 6))
fontsize = 12

LIF_values = df_tot[0].loc[features[0], keys].values
ML_values = df_tot[1].loc[features[0], keys].values

if norm:    
    max_EM = np.zeros(len(metrics))
    
    for ii in range(len(metrics)):
        max_EM[ii] = max(abs(ML_values[ii]), abs(LIF_values[ii]))

    ML_values = ML_values/max_EM
    LIF_values = LIF_values/max_EM


x = np.arange(len(metrics))
width = 0.3
plt.bar(x        , LIF_values, width, label='LIF', color='green', alpha = 0.5)
plt.bar(x + width, ML_values, width, label='ML', color='red', alpha = 0.5)

plt.xticks(x, x_label, fontsize = fontsize)
plt.title(f"Performance of feature '{features[0]}'", fontsize = fontsize)
plt.ylabel('Metric value', fontsize = fontsize)
plt.grid(True, axis='y')
plt.legend()
#%%
for idx, feature in enumerate(features):
    
    col = idx
    
    # base metric names
    keys = [f'{m}' for m in metrics]
    LIF_values = df_tot[0].loc[feature, keys].values
    ML_values = df_tot[1].loc[feature, keys].values
    
    if norm:    
        max_EM = np.zeros(len(metrics))
        
        for ii in range(len(metrics)):
            max_EM[ii] = max(abs(ML_values[ii]), abs(LIF_values[ii]))
    
        ML_values = ML_values/max_EM
        LIF_values = LIF_values/max_EM
    
    
    x = np.arange(len(metrics))
    width = 0.3
    
    ax[col].bar(x        , LIF_values, width, label='LIF', color='green', alpha = 0.5)
    ax[col].bar(x + width, ML_values, width, label='ML', color='red', alpha = 0.5)
    
    ax[col].set_xticks(x, x_label, fontsize = fontsize)
    ax[col].set_title(f"Performance of feature '{feature}'", fontsize = fontsize)
    ax[col].set_ylabel('Metric value', fontsize = fontsize)
    ax[col].grid(True, axis='y')

fig.legend(['LIF', 'Morris Lecar'], loc='lower center', ncol=3, fontsize=fontsize,
           bbox_to_anchor=(0.5, -0.05))
plt.suptitle(f'Correlation of extracted features for EEG channel {ch_name}')
plt.tight_layout()
plt.show()
