# -*- coding: utf-8 -*-
"""
# Created by Ruben Dörfel at 07.02.2021
"""

import os
from scipy.io import loadmat
import numpy as np
import pandas as pd
import mne
from functions import *
from scipy.io import loadmat


### Functions
def loadData(fpath, fname):
    """load all files of the same data and safe them in one list

    Parameters
    ----------
    fname : string
        file name to look for
    fpath : string
        file path

    Return
    -------
    data : list
        List of all elements

    """
    counter = 0
    csvfiles = []
    for file in os.listdir(fpath):
        try:
            if file.endswith(fname) and file != "Freq_Phase.mat":
                file_load = os.path.join(fpath, str(file))
                print(".mat Files found:\t", file)
                csvfiles.append(loadmat(str(file_load))['data'])
                counter += 1
        except Exception as e:
            raise e
            print("No files found here!")
    return csvfiles


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
list_el = [str('Pz'), str('PO5'), str('PO3'), str('POz'), str('PO4'), str('PO6'), str('O1'), str('Oz'),
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
Nh = 5  # Number of harmonics
Nf = len(vec_freq)  # Number of frequencies
Nb = 6  # Number of Blocks
Ne = 64  # number of electrodes
Nt = 1500  # number of samples
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
n_channels = len(df_montage['Channel'][3:])

# info
info = mne.create_info(ch_names=mne_montage.ch_names, sfreq=250.,
                       ch_types='eeg')

# events
N_events = Nf * Nb

events = np.zeros((N_events, 3))
events[:, 0] = N_start + N_start * np.arange(N_events)  # Events sample.
events[:, 1] = np.concatenate([[[0] * Nf], [[1] * Nf], [[2] * Nf], [[3] * Nf], [[4] * Nf], [[5] * Nf]], 1)
events[:, 2] = np.concatenate([np.arange(0, 40)] * Nb)  # All events have the sample id.
events = events.astype(int)

lEpochs = []
for i in range(Ns):
    data = list_subject_data[i]
    data_2 = np.zeros([Nb * Nf, Ne, Nt])
    for b in range(Nb):
        for f in range(Nf):
            data_2[b * Nf + f, :, :] = data[:, :, f, b]

    lEpochs.append(mne.EpochsArray(data_2, info, events, tmin=-0.5, verbose=False))
    lEpochs[i].set_montage(mne_montage)
    lEpochs[i].apply_baseline((None, 0))   # include 140 ms visual delay
    lEpochs[i].set_eeg_reference(['Cz'])     # set common reference
    lEpochs[i] = lEpochs[i].pick(list_el)  # pick only electrodes in visual area
    lEpochs[i].crop(0.140, 0.140+2, include_tmax=False)
    lEpochs[i] = lEpochs[i].filter(l_freq=7, h_freq=70, method='fir', phase='zero-double', verbose=False)

for s in range(0, Ns):
    for b in range(0, Nb):
        for f in range(0, Nf):
            # Referencing and baseline correction
            mat_data = preprocess(list_subject_data[s][:, :, f, b], vec_ind_el, ind_ref_el, N_start, N_stop)

            # Filter data
            mat_filt = mne.filter.filter_data(mat_data, fs, 7, 70, method='fir', phase='zero-double', verbose=False)

subject = lEpochs[i].crop(0.140, 5 + 0.140, include_tmax=False)
a = subject[239].get_data()
a = np.squeeze(a)
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(subject.times, a.T)
plt.subplot(2, 1, 2)
t = subject.times
plt.plot(t, mat_data.T)
