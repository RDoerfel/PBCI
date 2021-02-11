# -*- coding: utf-8 -*-
"""
# Created by Ruben Dörfel at 09.02.2021

Feature: # Apply CCA on the benchmark dataset

"""
import numpy as np
from scipy.io import loadmat
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt
import os
from datetime import datetime
import pandas as pd
import mne

from functions import *


### Set Working Directory
abspath = os.path.abspath(__file__)
dirname = os.path.dirname(abspath)
os.chdir(dirname)

dir_data = 'data'
dir_figures = 'figures'
dir_results = 'results'

### Load and prepare data
fs = 250  # sampling frequency in hz
mat_locations = np.genfromtxt(os.path.join(dir_data, '64-channel_locations.txt'))

dict_freq_phase = loadmat(os.path.join(dir_data, 'Freq_Phase.mat'))
vec_freq = dict_freq_phase['freqs'][0]
vec_phase = dict_freq_phase['phases'][0]

list_subject_data = loadData(os.path.join(dirname, dir_data), '.mat')  # load all subject data

## Convert to pandas dataframe
df_location = pd.read_table(os.path.join(dir_data, '64-channel_locations.txt'),
                            names=['Electrode', 'Degree', 'Radius', 'Label'])
df_location['Label'] = df_location['Label'].astype('string').str.strip()
df_location['Electrode'] = df_location['Electrode'].astype('int')

## channel selection
list_el = [str('PO5'), str('PO3'), str('POz'), str('PO4'), str('PO6'), str('O1'), str('Oz'),
           str('O2')]  # Electrodes to use
vec_ind_el = df_location[df_location['Label'].isin(list_el)].index  # Vector with indexes of electrodes to use
ind_ref_el = df_location['Electrode'][df_location['Label'] == 'Cz'].index[0]  # Index of reference electrode 'Cz'

## cutting
fs = 250  # sampling frequency in hz
N_pre = int(0.5 * fs)  # pre stim
N_delay = int(0.140 * fs)  # SSVEP delay
N_stim = int(5 * fs)  # stimulation
N_start = N_pre + N_delay - 1
N_stop = N_start + N_stim

vec_t = np.arange(-0.5, 5.5, 1 / fs)  # time vector
Nh = 5              # Number of harmonics
Nf = len(vec_freq)  # Number of frequencies
Nb = 6              # Number of Blocks
Ne = 64             # number of electrodes
Nt = 1500           # number of samples
Ns = len(list_subject_data)

## create mne objects
# montage
df_montage = pd.read_csv(os.path.join(dir_data, 'montage.DAT'), sep='\t', header=None)
df_montage.columns = ['Channel', 'x', 'y', 'z']
df_montage['Channel'] = df_montage['Channel'].str.strip()

loc = df_montage.to_numpy()
loc = loc[3:, 1:]
loc = loc.astype(float)
names = df_montage['Channel'][3:].to_list()
dict_montage = dict(zip(names, loc))
mne_montage = mne.channels.make_dig_montage(dict_montage)

# info
info = mne.create_info(ch_names=mne_montage.ch_names, sfreq=250.,ch_types='eeg')

# events
N_events = Nf * Nb

events = np.zeros((N_events, 3))
events[:, 0] = N_start + N_start * np.arange(N_events)  # Events sample.
events[:, 1] = np.concatenate([[[0] * Nf], [[1] * Nf], [[2] * Nf], [[3] * Nf], [[4] * Nf], [[5] * Nf]], 1)
events[:, 2] = np.concatenate([np.arange(0, 40)] * Nb)  # All events have the sample id.
events = events.astype(int)

# epochs
lEpochs = []
for i in range(Ns):
    data = list_subject_data[i]
    data = data.swapaxes(2, 3)
    data = data.reshape([Ne, Nt, Nb * Nf])
    data = data.swapaxes(1, 2)
    data = data.swapaxes(0, 1)
    lEpochs.append(mne.EpochsArray(data, info, events, tmin=-0.5, verbose=False))
    lEpochs[i].set_montage(mne_montage)


## Simulated referenc signals with harmonics
mat_Y = np.zeros([Nf, Nh * 2, N_stim])  # [Frequency, Harmonics * 2, Samples]
for k in range(0, Nf):
    for i in range(1, Nh + 1):
        mat_Y[k, i - 1, :] = np.sin(2 * np.pi * i * vec_freq[k] * vec_t[N_start:N_stop] + vec_phase[k])
        mat_Y[k, i, :] = np.cos(2 * np.pi * i * vec_freq[k] * vec_t[N_start:N_stop] + vec_phase[k])

### Frequency detection using CCA
list_result_cca = []  # list to store the subject wise results
list_time_cca = []
num_iter = 0

iS = 0  # iteration subject
iE = 0  # iteration epoch
#for iS, subject in enumerate(lEpochs):
for iS in range(0,1):
    subject = lEpochs[iS]
    t_start = datetime.now()
    mat_ind_max = np.zeros([1, Nb * Nf])  # index of maximum cca
    mat_time = np.zeros([1, Nb * Nf], dtype='object')  # matrix to store time needed

    # pre-processing
    subject.apply_baseline((None, 0))   # include 140 ms visual delay
    subject.set_eeg_reference(['Cz'])     # set common reference
    subject = subject.pick(list_el)     # pick only electrodes in visual area
    subject.crop(0.140, 0.140+5, include_tmax=False)
    subject.filter(l_freq=7, h_freq=70, method='fir', phase='zero-double', verbose=False)
    # subject.drop_bad(dict(eeg=40))     # drop bads based on threshold

    # classify
    for iE, epoch in enumerate(subject):
        t_trial_start = datetime.now()
        vec_rho = np.zeros(Nf)

        # apply cca
        for k in range(0, Nf):
            vec_rho[k] = apply_cca(epoch, mat_Y[k, :, :])

        t_trial_end = datetime.now()
        mat_time[iS, iE] = t_trial_end - t_trial_start
        mat_ind_max[iS, iE] = np.argmax(vec_rho)  # get index of maximum -> frequency -> letter
        num_iter = num_iter + 1
        print("CCA: Trial " + str(iE+1) + " of " + str(Nf * Nb), flush=True)

    list_time_cca.append(mat_time.reshape(Nf, Nb, order='F'))
    list_result_cca.append(mat_ind_max.reshape(Nf, Nb, order='F'))  # store results per subject
    t_end = datetime.now()
    print("CCA: Elapsed time for subject: " + str(iS + 1) + ": " + str((t_end - t_start)), flush=True)

mat_result_cca = np.concatenate(list_result_cca, axis=1)
mat_time_cca = np.concatenate(list_time_cca, axis=1)
### analysis
gof_cca = gof(vec_freq, mat_result_cca)
accuracy_cca = accuracy(vec_freq, mat_result_cca)

print("CCA: gof: " + str(gof_cca))
print("CCA: accuracy: " + str(accuracy_cca))

plt.figure()
plt.imshow(mat_result_cca)

np.save(os.path.join(dir_results, 'mat_result_cca_drop'), mat_result_cca)
np.save(os.path.join(dir_results, 'mat_time_cca_drop'), mat_time_cca)
