#Connectivity
import os
import sys
import warnings
import time
from itertools import product
from typing import List, Union
from tqdm import tqdm
from scipy import linalg
import scipy.io as sio
import numpy as np
import pandas as pd
from scipy import signal
import mne
import matplotlib.pyplot as plt
from pathlib import Path
from mne.epochs import make_fixed_length_epochs
from mne.io.base import BaseRaw
from mne import Epochs
from mne.preprocessing import ICA
from vmdpy import VMD
from pandas import DataFrame, set_option

from mne.filter import notch_filter, filter_data
from mne_features.bivariate import compute_time_corr, compute_phase_lock_val,compute_max_cross_corr
import mne
from electrodes_positions import load_data, get_electrodes_positions, get_electrodes_coordinates, set_electrodes_montage, preprocess_eeg_data
from signals import SignalPreprocessor


raw, info, ch_names, eeg_signal = load_data(r"C:\Users\Ahmed Guebsi\Downloads\ADHD_Data\Control_part1\v45p.mat")
print("eeg siiiiiignal", type(eeg_signal))

raw.plot(duration=10, n_channels=19, scalings='auto', title='EEG channels data')

electrodes_coordinates = get_electrodes_coordinates(ch_names)
# print(electrodes_coordinates)
dig_points = get_electrodes_positions(ch_names, electrodes_coordinates)
raw_with_dig_pts, info = set_electrodes_montage(ch_names, electrodes_coordinates, eeg_signal)

sfreq= raw.info['sfreq']

signal_preprocessor = SignalPreprocessor()
signal_preprocessor.fit(raw_with_dig_pts)

signal_processed = preprocess_eeg_data(raw_with_dig_pts,info)

scan_durn = signal_processed._data.shape[1] / signal_processed.info['sfreq']
        #signal_duration = int(scan_durn)
signal_duration = round((signal_processed._data.shape[1] - 1) / signal_processed.info["sfreq"], 3) # corrected
# epoch_events_num = int(signal_duration // 4)
# signal_processed.crop(tmin=0, tmax=signal_duration // 4 * 4)
# assuming you have a Raw and ICA instance previously fitted
print(type(signal_processed))

epochs = make_fixed_length_epochs(signal_processed, duration=4.0, preload=True, verbose=False)
for index, epoch in enumerate(epochs):
    print(epoch.shape) # (19, 512)
    cross_corr=compute_max_cross_corr(sfreq, epoch, include_diag=False)
    print(cross_corr.shape)

# Resting state Functional Connectivity analysis at the sensor level - Davide Aloi
### Global Variables ###
delta = 1-4
alpha = 8,13
theta = 4,8
beta = 13,30
fmin, fmax = alpha
min_epochs = 5 #Start from epoch n.
max_epochs = 25 #End at epoch n.
# Get the strongest connections
n_con = 124*123 # show up to n_con connections THIS SHOULD BE CHECKED.
min_dist = 3  # exclude sensors that are less than 4cm apart THIS SHOULD BE CHECKED
method = 'pli' # Method used to calculate the connectivity matrix


sfreq = epochs.info['sfreq']  # the sampling frequency
con, freqs, times, n_epochs, n_tapers = spectral_connectivity(
    epochs[min_epochs:max_epochs], method=method, mode='multitaper', sfreq=sfreq, fmin=fmin, fmax=fmax,
    faverage=True, tmin=tmin, mt_adaptive=False, n_jobs=1)

# the epochs contain an EOG channel, which we remove now
ch_names = epochs.ch_names
idx = [ch_names.index(name) for name in ch_names]
con = con[idx][:, idx]
# con is a 3D array where the last dimension is size one since we averaged
# over frequencies in a single band. Here we make it 2D

con = con[:, :, 0] #This connectivity matrix can also be visualized

# Plot the sensor locations
sens_loc = [epochs.info['chs'][picks[i]]['loc'][:3] for i in idx]
sens_loc = np.array(sens_loc)
#Layout
layout = mne.channels.find_layout(epochs.info, exclude=[])
new_loc = layout.pos
threshold = np.sort(con, axis=None)[-n_con]
ii, jj = np.where(con >= threshold)

# Remove close connections
con_nodes = list()
con_val = list()
for i, j in zip(ii, jj):
    if linalg.norm(sens_loc[i] - sens_loc[j]) > min_dist:
        con_nodes.append((i, j))
        con_val.append(con[i, j])

con_val = np.array(con_val)