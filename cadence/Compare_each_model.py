#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 16:21:49 2025

@author: gibaek
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import os, sys

path_c = os.path.dirname(os.path.abspath(__file__))
path_p = os.path.dirname(path_c)
path_g = os.path.dirname(path_p)
root = os.path.dirname(path_g)

sys.path.append(path_c)

ML = []
LIF = []
cap_list = [5,7,10]

for ii in range(3):
    ML.append(pd.read_csv(f'{root}/data/cadence/chbmit/chb01/sim_result/ML_balance_cap_{cap_list[ii]}pF_classification_result.csv'))
    LIF.append(pd.read_csv(f'{root}/data/cadence/chbmit/chb01/sim_result/AH_balance_cap_{cap_list[ii]}pF_classification_result.csv'))

#%%

ch = ['FP2-F4', 'FT10-T8', 'T8-P8']
ch_name = ch[2]

fig, ax = plt.subplots(1,3, figsize = (14,6))
for ii in range(3):
    met_ML = ML[ii].loc[ML[ii]['Unnamed: 0'] == ch_name, ['Precision', 'Sensitivity', 'F1 Score']].values.flatten()
    met_IF = LIF[ii].loc[LIF[ii]['Unnamed: 0'] == ch_name, ['Precision', 'Sensitivity', 'F1 Score']].values.flatten()
    
    bar_width = 0.35
    
    x = np.arange(len(ML[ii].columns[1:]))
    
    ax[ii].bar(x - bar_width / 2, met_ML, width=bar_width, label='M-L', color='orange', alpha = 0.7)
    ax[ii].bar(x + bar_width / 2, met_IF, width=bar_width, label='LIF', color='blue', alpha = 0.7)
    
    ax[ii].set_ylabel('Values')
    ax[ii].set_title(f'Cap: {cap_list[ii]} pF')
    ax[ii].set_xticks(x, ML[0].columns[1:])
    
    
plt.suptitle(f'For channel {ch_name}')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
plt.tight_layout()
plt.show()
