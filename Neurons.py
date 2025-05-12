#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code calculate behavior of M-L model, which processing CHB-MIT dataset

@author: gibaek
"""
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

from utils import *

#%% Conventional Morris Lecar model
def M_L_con(I_cal, dt, C_m):
    
    # Setting
    start_scope()
    defaultclock.dt = dt * second

    # Parameters
    Cm = C_m * uF/cm**2
    gCa = 4.4 * mS/cm**2
    gK = 8 * mS/cm**2
    gL = 2 * mS/cm**2
    VCa = 120 * mV
    VK = -84 * mV
    VL = -60 * mV
    V1 = -1.2 * mV
    V2 = 18 * mV
    V3 = 2 * mV
    V4 = 30 * mV
    phi = 0.04 / ms
    area = 200 * umetre**2  # neuron area

    # Stimulus
    I_density = I_cal * amp / area
    stimulus = TimedArray(I_density, dt=dt * second)

    # Model equation
    eqs = '''
    dV/dt = (stimulus(t) - gCa*m_inf*(V - VCa) - gK*w*(V - VK) - gL*(V - VL)) / Cm : volt
    dw/dt = phi * (w_inf - w) : 1
    m_inf = 0.5*(1 + tanh((V - V1)/V2)) : 1
    w_inf = 0.5*(1 + tanh((V - V3)/V4)) : 1
    '''

    # Generate Neuron
    neuron = NeuronGroup(1, eqs, method='rk4')
    neuron.V = -60 * mV
    neuron.w = 0.0

    # Monitoring
    M = StateMonitor(neuron, ['V', 'w'], record=True)

    # Computing
    run(len(I_cal) * dt * second)
    
    return M


#%%
def LIF(I_cal, dt, Vr = -65, El = -65, Vt = -50, t_con = 10):
    
    # Neuron parameters
    tau = t_con * ms         # Time constant
    Vt = Vt * mV         # Threshold voltage
    Vr = Vr * mV         # reset voltage
    El = El * mV         # leak voltage
    R = 100 * Mohm        # resistnace
    
    start_scope()
    defaultclock.dt = dt * second
    
    # Define duration from input current
    duration = len(I_cal) * dt * second
    
    # Use TimedArray for time-varying current
    I = TimedArray(I_cal * amp, dt=dt*second)
    
    # Equation
    eqs = '''
    dV/dt = (El - V + R*I(t))/tau : volt
    '''
    
    G = NeuronGroup(1,eqs, threshold = 'V>Vt', reset='V = Vr', method = 'rk4')
    G.V = Vr
    
    M = StateMonitor(G, 'V', record=0)
    
    run(duration)
    
    return M

#%% Excitability threshold, Tonic Spiking, Depolarization Block, Nonlinear
if __name__ == '__main__':
    
    Amp_list = [5e-10, 1e-9, 2e-9]
    
    plt.figure(0)
    for I_amp in Amp_list:
        interval = 1e-4
        I_dc = np.ones(int(2.0 / interval)) * I_amp
        M = LIF(I_dc, dt = interval, Vt = 10, t_con = 500) # LIF
        plt.plot(M.t[:], M.V[0][:], label = f'{I_amp*1e9:.1f} nA', alpha = 0.7)
   
    plt.xlabel('Time (sec)')
    plt.ylabel('Membrane potential (mV)')
    plt.legend(title='Input current', loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)
    plt.grid()
    #%%
    plt.figure(1)
    I_dc = np.zeros(int(1/interval))
    Vr = -50
    El = -65
    t_con = 100
    M = LIF(I_dc, dt = interval, El = El,Vr = Vr, t_con = t_con)
    plt.plot(M.t/ms, M.V[0]/mV)
    plt.axhline(El, ls='--', color='r', label='Leak potential El')
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane potential (mV)')
    plt.title('Leakage Behavior (no input)')
    plt.legend()
    plt.grid()
    plt.show()
    
    #%% Tonic spiking, Hopf Bifurcation
    fig, ax = plt.subplots(2,4, figsize = (16,8))
    Amp_list = [1e-10,3e-10,5e-10,7e-10]
    # Amp_list = np.linspace(1e-10, 9e-10, 4) # 0.1 nA ~ 0.9 nA (for conventional)
    
    fontsize = 14
    handles_total = []
    for idx, I_amp in enumerate(Amp_list):
        interval = 1e-5
        I_dc = np.ones(int(1.0 / interval)) * I_amp
        M = M_L_con(I_cal=I_dc, dt=interval, C_m = 20)
        
        # Morris Lecar parameter
        Cm = 20; gCa = 4.4; gK = 8; gL = 2
        VCa = 120; VK = -84; VL = -60
        V1 = -1.2; V2 = 18; V3 = 2; V4 = 30
        area = 200e-8  # cm²
        I_ext = I_amp / area * 1e6 # A → μA/cm^2
        
        def m_inf(V): return 0.5 * (1 + np.tanh((V - V1) / V2))
        def w_inf(V): return 0.5 * (1 + np.tanh((V - V3) / V4))
        V_range = np.linspace(-70, 60, 500)
        V_nullcline = (I_ext - gCa * m_inf(V_range) * (V_range - VCa) - gL * (V_range - VL)) / (gK * (V_range - VK))
        w_nullcline = w_inf(V_range)
        V_nullcline = np.clip(V_nullcline, 0, 1)
    
        V_sim = M.V[0] / mV  # V → mV
        w_sim = M.w[0]
        
        h1, = ax[0][idx].plot(V_range, V_nullcline, color='orange', label="dV/dt = 0")
        h2, = ax[0][idx].plot(V_range, w_nullcline, color='tomato', label="dw/dt = 0")
        h3, = ax[0][idx].plot(V_sim, w_sim, color='black', lw=1, label="Simulation")
        
        ax[0][idx].set_xlabel("Membrane potential V (mV)", fontsize = fontsize)
        ax[0][idx].set_ylabel("Recovery variable w", fontsize = fontsize)
        ax[0][idx].grid()
        
        ax[1][idx].plot(M.t, M.V[0], label=f'{I_amp * 1e9:.1f} nA')
        ax[1][idx].set_xlabel('Time (sec)', fontsize = fontsize)
        ax[1][idx].set_ylabel('Membrane potential (mV)', fontsize = fontsize)
        ax[1][idx].set_title(f'Input current: {I_amp*1e9:.3f} (nA)', fontsize = fontsize)
        ax[1][idx].grid()
        
        if idx == 0:
            handles_total = [h1, h2, h3]
        
    fig.legend(handles_total, ['dV/dt = 0', 'dw/dt = 0', 'Simulation'],
           loc='upper center', ncol=4, fontsize=12)
    
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.show()
    
    #%% Import raw EEG data
    
    chb_num = '01'
    folder_path = f'{path_g}/physionet.org/files/chbmit/1.0.0/chb{chb_num}/'
    edf_files = sorted(glob.glob(os.path.join(folder_path, '*.edf')))
    seizure_dict = parse_summary_file(f'{folder_path}/chb{chb_num}-summary.txt', verbose=False)
    
    file = edf_files[2]
    file_basename = os.path.basename(file)
    sz_timing = seizure_dict.get(file_basename)
    print(f'\nLoading neuron.. {file_basename}')
    if sz_timing:
        print(f"\n{file_basename}: seizure interval: {sz_timing[0]}\n")
    else:
        print(f"\n{file_basename} has no seizure information.\n")
    
    raw = mne.io.read_raw_edf(file, verbose='error')
    
    fs = 256
    upsample_factor = 100
    cal_duration = 10
    start_t = 3003
    norm_current = 1e-9
    # Convert EEG voltage to current
    times_upsampled, I_stim, _ = EEG_to_Current(raw = raw, upsample_factor = upsample_factor, fs = fs,
                                                norm_current = norm_current, verbose = False, channel_name = 'FP1-F7')
    neuron_dt = times_upsampled[1] - times_upsampled[0]
    
    # compute
    I_stim = I_stim[fs * upsample_factor * start_t:fs * upsample_factor * (start_t + cal_duration)] 
    times_upsampled = times_upsampled[fs * upsample_factor * start_t:fs * upsample_factor * (start_t + cal_duration)]
    M_M_L = M_L_con(I_cal = I_stim, dt = neuron_dt)
    M_LIF, M_spike = LIF(I_cal = I_stim, dt = neuron_dt, Vt = -50)
    
    # visualization
    fig, ax = plt.subplots(1,3, figsize = (17,6))
    fontsize = 15
    
    ax[0].plot(times_upsampled, I_stim, alpha = 0.9)
    ax[0].set_xlabel('$time (sec)$', fontsize = fontsize)
    ax[0].set_ylabel('$EEG\ signal\ (A)$', fontsize = fontsize)
    ax[0].set_title(f'Normalization current: {norm_current*1e9}nA', fontsize = fontsize)
    ax[0].grid()
    
    ax[1].plot(M_M_L.t , M_M_L.V[0], alpha = 0.9)
    ax[1].set_xlabel('Time', fontsize = fontsize)
    ax[1].set_ylabel('Membrane potential (V)', fontsize = fontsize)
    ax[1].set_title('M-L conventional neuron response', fontsize = fontsize)
    ax[1].grid()
    
    ax[2].plot(M_LIF.t , M_LIF.V[0], alpha = 0.9)
    ax[2].set_xlabel('Time', fontsize = fontsize)
    ax[2].set_ylabel('Membrane potential (V)', fontsize = fontsize)
    ax[2].set_title(f'LIF neuron response spike number: {M_spike.count[0]}', fontsize = fontsize)
    ax[2].grid()
    
    plt.tight_layout()
