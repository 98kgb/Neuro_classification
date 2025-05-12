
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from brian2 import *
from Neurons import *
from tqdm import tqdm
__spyder__ = True
if '__spyder__' in globals():
    del device
prefs.codegen.target = 'cython'  # define prefer engine
path_c = os.path.dirname(os.path.abspath(__file__))
path_p = os.path.dirname(path_c)
path_g = os.path.dirname(path_p)
sys.path.append(path_c)

from utils import *


def get_spike_train(I_cal, dt, threshold, model='LIF', param = 10):
    '''
    Parameters
    ----------
    I_cal : TYPE
        DESCRIPTION.
    dt : TYPE
        DESCRIPTION.
    threshold : threshold voltage that decide the spike existance (unit: mV)
        DESCRIPTION.
    model : TYPE, optional
        DESCRIPTION. The default is 'LIF'.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.
    '''
    
    if model == 'LIF':
        M = LIF(I_cal, dt = dt, Vt = 50, t_con = param) # LIF
        V = M.V[0] / mV
        spike_times = np.where((V[1:] > threshold) & (V[:-1] < threshold))[0]
        return spike_times * dt, V
    elif model == 'M-L':
        M = M_L_con(I_cal, dt, C_m = param)
        V = M.V[0] / mV
        spike_times = np.where((V[1:] > threshold) & (V[:-1] < threshold))[0]
        return spike_times * dt, V
    else:
        raise ValueError("Model must be 'LIF', or 'M-L'")


#%% Load raw EEG data
chb_num = '01'
folder_path = f'{path_g}/physionet.org/files/chbmit/1.0.0/chb{chb_num}/'
edf_files = sorted(glob.glob(os.path.join(folder_path, '*.edf')))
seizure_dict = parse_summary_file(f'{folder_path}/chb{chb_num}-summary.txt', verbose= False)

# Define feature extraction subject
file = edf_files[3]
file_basename = os.path.basename(file)
sz_timing = seizure_dict.get(file_basename)
print(f'\nLoading... {file_basename}')
if sz_timing:
    print(f"\n{file_basename}: seizure interval: {sz_timing[0]}\n")
else:
    print(f"\n{file_basename} has no seizure information.\n")
#%%
fs = 256
upsample_factor = 100
raw = mne.io.read_raw_edf(file, verbose = 'error') # load EEG dataset
verbose = False
models = ['M-L', 'LIF']
# norm_currents = [5e-9]
ch_names = ['T8-P8-1', 'FT10-T8', 'FP2-F4']

tau_list = [10,20,40,60,100]
c_m_list = [10,50,100,200,400]
norm_current = 5e-9
# tau = 100
# c_m = 400


for ch_name in ch_names:
    for kk in range(len(tau_list)):
        file_name = f'spike_train_chb{chb_num}_{file_basename.replace(".edf", "")}_ch_name_{ch_name}_norm_I_{norm_current}'
        
        if f'{file_name}_LIF_{tau_list[kk]}.npy' in os.listdir(path_c+'/spike_data/'):
            print('File already exsit')
            continue
        
        
        times_upsampled, I_stim, _ = EEG_to_Current(raw = raw, channel_name = ch_name, upsample_factor = upsample_factor,
                                                    fs = fs, norm_current = norm_current, verbose = False)
        
        time_window = 2 # 2 sec
        epoch_len = fs * upsample_factor * time_window
        num_epoch = int(I_stim.shape[0]/epoch_len)
        
        if f'spike_train_chb{chb_num}_{file_basename.replace(".edf", "")}_label.npy' in os.listdir(path_c+'/spike_data/'):
            print('Label file already exists')
        else:
            sz_labels = get_epoch_labels(sz_timing, times_upsampled, num_epoch, epoch_len)
            np.save(f'./spike_data/spike_train_chb{chb_num}_{file_basename.replace(".edf", "")}_label.npy', sz_labels)
        
        
        # Compute features for each window with M-L model
        features_LIF = []
        features_ML = []
        
        visualize = False
        verbose = False
        
        for ii in tqdm(range(num_epoch), desc = f'Feature extracting for {file_basename.replace(".edf", "")} channel: {ch_name}'):
            I_cal = I_stim[fs * upsample_factor * time_window * ii:fs * upsample_factor * time_window * (ii+1)] 
            t_cal = times_upsampled[fs * upsample_factor * time_window * ii:fs * upsample_factor * time_window * (ii+1)]
            neuron_dt = t_cal[1] - t_cal[0]
            
            # Compute neuron behavior
            train_LIF, _ = get_spike_train(I_cal, neuron_dt, threshold = 0, model = 'LIF', param = tau_list[kk])
            train_M_L, _ = get_spike_train(I_cal, neuron_dt, threshold = 0, model = 'M-L', param = c_m_list[kk])
            
            features_LIF.append(train_LIF)
            features_ML.append(train_M_L)
            
        LIF_data = np.array(features_LIF, dtype=object)
        ML_data = np.array(features_ML, dtype=object)
        np.save(f'./spike_data/{file_name}_LIF_{tau_list[kk]}.npy', LIF_data)
        np.save(f'./spike_data/{file_name}_ML_{c_m_list[kk]}.npy', ML_data)
    
#%%
checking = False
if checking:
    chb_num = '01'
    folder_path = f'{path_g}/physionet.org/files/chbmit/1.0.0/chb{chb_num}/'
    edf_files = sorted(glob.glob(os.path.join(folder_path, '*.edf')))
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
        
    fs = 256
    upsample_factor = 100
    raw = mne.io.read_raw_edf(file, verbose = 'error') # load EEG dataset
    verbose = False
    models = ['M-L', 'LIF']
    norm_currents = [1e-9, 2e-9, 3e-9]
    ch_names = ['T8-P8-1', 'FT10-T8', 'FP2-F4']
    # ch_names = raw.ch_names

    norm_current = 5e-9
    ch_name = 'T8-P8-1' # 'FT10-T8' 
    
    file_name = f'spike_train_chb{chb_num}_{file_basename.replace(".edf", "")}_ch_name_{ch_name}_norm_I_{norm_current}'
    
    times_upsampled, I_stim, _ = EEG_to_Current(raw = raw, channel_name = ch_name, upsample_factor = upsample_factor,
                                                fs = fs, norm_current = norm_current, verbose = False)
    
    time_window = 2 # 2 sec
    epoch_len = fs * upsample_factor * time_window
    num_epoch = int(I_stim.shape[0]/epoch_len)
    ii = 1500
    I_cal = I_stim[fs * upsample_factor * time_window * ii:fs * upsample_factor * time_window * (ii+1)] 
    t_cal = times_upsampled[fs * upsample_factor * time_window * ii:fs * upsample_factor * time_window * (ii+1)]
    neuron_dt = t_cal[1] - t_cal[0]
    
    # Compute neuron behavior
    tau = 10
    c_m = 10
    train_LIF, V_LIF = get_spike_train(I_cal, neuron_dt, threshold = 0, model = 'LIF', param = tau)
    train_M_L, V_ML = get_spike_train(I_cal, neuron_dt, threshold = 0, model = 'M-L', param = c_m)

    fig, ax = plt.subplots(2,2, figsize = (10,8))
    ax[0][0].plot(t_cal, V_LIF, alpha = 0.8)
    ax[0][0].set_title(f'LIF with $tau$ {tau} $ms$')
    
    ax[0][1].plot(t_cal, V_ML, alpha = 0.8)
    ax[0][1].set_title(f'ML with $C_m$ {c_m} $uF/cm^2$')
    
    for ii in range(2):
        ax[0][ii].tick_params(axis = 'x',rotation = 45)
        ax[0][ii].set_xlabel('$Time (sec)$')
        ax[0][ii].set_ylabel('$V_m (mV)$')
    plt.suptitle(f"Measured in channel {ch_name}")
    plt.tight_layout()
    
    duration = time_window
    bin_width = 0.1
    
    # 1. Raster Plot
    ax[1][0].eventplot([train_LIF, train_M_L], colors=['blue', 'green'], lineoffsets=[1, 0], linelengths=0.9)
    ax[1][0].set_yticks([0, 1])
    ax[1][0].set_yticklabels(['M-L', 'LIF'])
    ax[1][0].set_title("Spike Train Raster Plot (LIF vs M-L)")
    ax[1][0].set_ylabel("Neuron Model")
    
    # 2. Histogram
    bins = np.arange(0, duration + bin_width, bin_width)
    hist_LIF, _ = np.histogram(train_LIF, bins=bins)
    hist_ML, _ = np.histogram(train_M_L, bins=bins)
    
    ax[1][1].bar(bins[:-1], hist_LIF, width=bin_width, alpha=0.6, label='LIF', color='blue', align='edge')
    ax[1][1].bar(bins[:-1], hist_ML, width=bin_width, alpha=0.6, label='M-L', color='green', align='edge')
    ax[1][1].set_xlabel("Time (s)")
    ax[1][1].set_ylabel("Spike Count")
    ax[1][1].legend()
    ax[1][1].set_title("Spike Count Histogram")
    
    plt.tight_layout()
    plt.show()



