#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  6 16:00:38 2025

@author: gibaek
"""

import os, sys, glob

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
sys.path.append(path_p)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, accuracy_score
from utils import *
from tqdm import tqdm
from feature_extraction_raw import feature_extract
from analysis_raw import analysis_EEG
from training_dataset_balanced import generate_dataset

for ii in range(1,2):

    if ii <10:
        chb_num = f'0{ii}'
    else:
        chb_num = f'{ii}'
    feature_extract(chb_num, verbose = False)
    ch_names = analysis_EEG(chb_num, plot = True)
    generate_dataset(chb_num, ch_names, verbose = False)
    
    
#%%
for ii in range(1, 15):
    if ii <10:
        chb_num = f'0{ii}'
    else:
        chb_num = f'{ii}'
    folder_path = f'/home/gibaek/data/cadence/chbmit/chb{chb_num}/'
    txt_files = sorted(glob.glob(os.path.join(folder_path, '*.txt')))
    
    aa = np.genfromtxt(txt_files[0])
    info = {'maximum time':max(aa[:,0])}
    
    save_path = os.path.join(folder_path, 'info.txt')
    with open(save_path, 'w') as f:
        for key, value in info.items():
            f.write(f'{key}: {value}\n')
    
    