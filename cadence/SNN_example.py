#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 13:19:10 2025

@author: gibaek
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, accuracy_score

import torch
from sklearn.model_selection import train_test_split

import os, sys
#%%
path_c = os.path.dirname(os.path.abspath(__file__))
path_p = os.path.dirname(path_c)
path_g = os.path.dirname(path_p)
root = os.path.dirname(path_g)

sys.path.append(path_c)

# CSV 읽기
cap = 5
chb_num = '01'
model = 'AH'
df = pd.read_csv(f'{root}/data/cadence/chbmit/chb{chb_num}/sim_result/{model}_balance_cap_{cap}pF.csv')


# 공통 시간축
time = df['/Vout_FP2_F4 X'].values

# 세 채널의 voltage
voltages = df[['/Vout_FP2_F4 Y', '/Vout_FT10_T8 Y', '/Vout_T8_P8 Y']].values


# Spike detection: voltage <= -1V일 때 spike
spikes = (voltages <= -1.0).astype(int)  # (전체샘플수, 3)

X = []
y = []
total_samples = len(time)

start_idx = 0
while start_idx < total_samples:
    # 현재 window의 시작 시간
    start_time = time[start_idx]
    end_time = start_time + 2  # 2초 later

    # start_idx부터 end_time 이하까지 포함하는 index들 찾기
    window_mask = (time >= start_time) & (time < end_time)
    window_spikes = spikes[window_mask]

    # window가 비어있으면 break
    if window_spikes.shape[0] == 0:
        break

    # firing rate 계산
    firing_rate = window_spikes.sum(axis=0) / 2  # 2초 기준

    # label 결정: window 중심 시간
    center_time = (start_time + end_time) / 2
    label = 1 if center_time <= 446 else 0

    # 데이터 저장
    X.append(firing_rate)
    y.append(label)

    # 다음 window로 이동: end_time을 포함하는 index 찾기
    start_idx = window_mask.nonzero()[0][-1] + 1
    
# 리스트를 텐서로 변환
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# train/val 분리
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

X_train.shape, y_train.shape, X_val.shape, y_val.shape



import torch
import torch.nn as nn
import snntorch as snn

class FiringRateSNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 6)  # input_dim=3, hidden_dim=32
        self.lif1 = snn.Leaky(beta=0.9)
        self.fc2 = nn.Linear(6, 2)  # hidden_dim=32, output_dim=2 (binary classification)

    def forward(self, x):
        # LIF 뉴런 상태 초기화
        mem1 = self.lif1.init_leaky()
        
        # 첫 번째 layer
        cur1 = self.fc1(x)
        spk1, mem1 = self.lif1(cur1, mem1)
        
        # 출력 layer (linear only)
        out = self.fc2(spk1)
        return out

from torch.utils.data import TensorDataset, DataLoader

# DataLoader 준비
batch_size = 8

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# 모델, optimizer, loss 설정
model = FiringRateSNN()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# 학습 루프
n_epochs = 200

for epoch in range(n_epochs):
    model.train()
    train_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            outputs = model(batch_X)
            preds = outputs.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)
    
    acc = correct / total
    print(f"Epoch {epoch+1}/{n_epochs} | Train Loss: {train_loss:.4f} | Val Acc: {acc*100:.2f}%")

#%%

train = [X_train, y_train]
val = [X_val, y_val]
datas = [train, val]
preds = []
acts = []

for data in datas:
    pred = model(data[0]).detach().numpy()
    pred = np.argmax(pred, axis = 1)
    actual = data[1].numpy()
    preds.append(pred)
    acts.append(actual)

#%%
conf_train = confusion_matrix(acts[0], preds[0])
conf_test = confusion_matrix(acts[1], preds[1])


# visualization
fig, ax = plt.subplots(1,2,figsize = (16,6))
sns.heatmap(conf_train, annot = True, fmt = 'd', cmap = 'Reds', ax = ax[0])
sns.heatmap(conf_test, annot = True, fmt = 'd', cmap = 'Reds', ax = ax[1])

title = ['Train set', 'Validate set']
for ii in range(2):
    ax[ii].set_xlabel("Prediction")
    ax[ii].set_ylabel("Actual")
    ax[ii].set_title(f"SNN classification result {title[ii]}")

# Calculate evaluation metrics

    precision = precision_score(acts[ii], preds[ii], zero_division=0)
    recall = recall_score(acts[ii], preds[ii], zero_division=0)  # = sensitivity
    f1 = f1_score(acts[ii], preds[ii], zero_division=0)
    
    print(f"🔎 Performance Metrics for measure {title[ii]}")
    print(f"Precision     : {precision:.3f}")
    print(f"Sensitivity   : {recall:.3f}")
    print(f"F1 Score      : {f1:.3f}\n")
