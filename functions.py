# -*- coding: iso-8859-1 -*-
"""
# Created by Ruben Dörfel at 08.02.2021

Feature: Function
This File contains functions that are often used. It is to be imported at the beginning of all scripts.

import functions.py
"""

import os
import seaborn as sns
import numpy as np
from sklearn.cross_decomposition import CCA
from scipy.io import loadmat
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from datetime import datetime
import pandas as pd
import mne
import sys

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
                csvfiles.append(loadmat(str(file_load))['data'])
                counter += 1
        except Exception as e:
            raise e
            print("No files found here!")

    print(".mat Files found:\t" + str(counter))
    return csvfiles


def preprocess(mat_input, vec_pick_el, i_ref_el, n_start, n_stop):
    """Extract and preprocess the data by applying referencing, baseline correction, channel selection, and cropping
    Parameters
    ----------
    mat_input : array, shape(n_channel,n_samples)
        Array containing the data for one trial.
    vec_pick_el : Int64Index, size(n_channel)
        The indices of the selected electrodes.
    i_ref_el : int
        Index for reference electrode.
    n_start : int
        Index for start sample.
    n_stop : int
        Index for stop sample.

    Returns
    -------
    mat_output : array, shape(n_channel,n_sample)
        The preprocessed data.
    """
    ## Referencing and baseline correction
    baseline = np.mean(np.mean(mat_input[:, 0:n_start], axis=0))  # get baseline (DC offset)
    mat_input = mat_input - baseline  # apply baseline

    mat_output = mat_input - mat_input[i_ref_el, :]  # reference
    # mat_data = mat_data - np.mean(mat_data, axis=0) # common-average-referencing
    mat_output = mat_output[vec_pick_el, n_start:n_stop] - baseline  # channel selection
    return mat_output


def apply_cca(X, Y):
    """computes the maximum canonical correltion via cca
    Parameters
    ----------
    X : array, shape (n_channels, n_times)
        Input data.
    Y : array, shape (n_signals, n_times)
        Reference signal to find correlation with

    Returns
    -------
    rho : int
        The maximum canonical correlation coeficent
    """
    n_comp = 1
    cca = CCA(n_components=n_comp)
    cca.fit(X.transpose(), Y.transpose())  # transpose to bring into shape(n_sample,n_feature)
    x, y = cca.transform(X.transpose(), Y.transpose())
    rho = np.diag(np.corrcoef(x, y, rowvar=False)[:n_comp, n_comp:])

    return rho


def apply_ext_cca(X, Y, X_Train):
    """computes the maximum canonical correltion via cca
    Parameters
    ----------
    X : array, shape (n_channels, n_times)
        Input data.
    X_Train : array, shape (n_channels, b_times)
        Second Reference data
    Y : array, shape (n_signals, n_times)
        Reference signal to find correlation with

    Returns
    -------
    rho : int
        The maximum canonical correlation coeficent
    """
    n_comp = 1
    cca1 = CCA(n_components=n_comp)
    cca2 = CCA(n_components=n_comp)
    cca3 = CCA(n_components=n_comp)
    cca4 = CCA(n_components=n_comp)

    cca1.fit(X.transpose(), Y.transpose())
    x, y = cca1.transform(X.transpose(), Y.transpose())
    rho_1 = np.diag(np.corrcoef(x, y, rowvar=False)[:n_comp, n_comp:])
    cca2.fit(X.transpose(), X_Train.transpose())
    w_xxt = cca2.x_weights_
    cca3.fit(X.transpose(), Y.transpose())
    w_xy = cca3.x_weights_
    cca4.fit(X_Train.transpose(), Y.transpose())
    w_xty = cca4.x_weights_
    rho_2 = np.diag(
        np.corrcoef(np.matmul(X.transpose(), w_xxt), np.matmul(X_Train.transpose(), w_xxt), rowvar=False)[:n_comp,
        n_comp:])
    rho_3 = np.diag(
        np.corrcoef(np.matmul(X.transpose(), w_xy), np.matmul(X_Train.transpose(), w_xy), rowvar=False)[:n_comp,
        n_comp:])
    rho_4 = np.diag(
        np.corrcoef(np.matmul(X.transpose(), w_xty), np.matmul(X_Train.transpose(), w_xty), rowvar=False)[:n_comp,
        n_comp:])

    # eq 8, Chen 2015,PNAS
    rho = np.sign(rho_1) * rho_1 ** 2 + np.sign(rho_2) * rho_2 ** 2 + np.sign(rho_3) * rho_3 ** 2 + np.sign(
        rho_4) * rho_4 ** 2

    return rho


def apply_ext_fbcca(X, Y, X_Train):
    """computes the maximum canonical correltion via cca
    Parameters
    ----------
    X : array, shape (n_channels, n_times)
        Input data.
    X_Train : array, shape (n_channels, b_times)
        Second Reference data
    Y : array, shape (n_signals, n_times)
        Reference signal to find correlation with

    Returns
    -------
    rho : int
        The maximum canonical correlation coeficent
    """
    n_comp = 1
    cca1 = CCA(n_components=n_comp)
    cca2 = CCA(n_components=n_comp)
    cca3 = CCA(n_components=n_comp)
    cca4 = CCA(n_components=n_comp)

    cca1.fit(X.transpose(), Y.transpose())
    x, y = cca1.transform(X.transpose(), Y.transpose())
    rho_1 = np.diag(np.corrcoef(x, y, rowvar=False)[:n_comp, n_comp:])
    cca2.fit(X.transpose(), X_Train.transpose())
    w_xxt_x = cca2.x_weights_
    w_xxt_y = cca2.y_weights_
    cca3.fit(X.transpose(), Y.transpose())
    w_xy = cca3.x_weights_
    cca4.fit(X_Train.transpose(), Y.transpose())
    w_xty = cca4.x_weights_
    rho_2 = np.diag(
        np.corrcoef(np.matmul(X.transpose(), w_xxt_x), np.matmul(X_Train.transpose(), w_xxt_x), rowvar=False)[:n_comp,
        n_comp:])
    rho_3 = np.diag(
        np.corrcoef(np.matmul(X.transpose(), w_xy), np.matmul(X_Train.transpose(), w_xy), rowvar=False)[:n_comp,
        n_comp:])
    rho_4 = np.diag(
        np.corrcoef(np.matmul(X.transpose(), w_xty), np.matmul(X_Train.transpose(), w_xty), rowvar=False)[:n_comp,
        n_comp:])
    rho_5 = np.diag(
        np.corrcoef(np.matmul(X_Train.transpose(), w_xxt_x), np.matmul(X_Train.transpose(), w_xxt_y), rowvar=False)[
        :n_comp,
        n_comp:])

    # eq 8, Chen 2015,PNAS
    rho = np.sign(rho_1) * rho_1 ** 2 + np.sign(rho_2) * rho_2 ** 2 + np.sign(rho_3) * rho_3 ** 2 + np.sign(
        rho_4) * rho_4 ** 2 + np.sign(rho_5) * rho_5 ** 2

    return rho


def accuracy(freqs, result):
    """computes the accuracy
    Parameters
    ----------
    freqs : array, shape (n_freqs, 1)
        Frequencies / labels.
    result : array, shape (n_freqs, n_trials)
        The estimated frequencies

    Returns
    -------
    accuracy : float
        The accuracy in percent
    """
    n_correct = np.sum(freqs.reshape(40, 1) == freqs[result.astype(int)])
    return 100 * n_correct / (np.size(result))


def weight(n):
    """computes the weight for fbcca coefficients

    Follow computation in paper "Filter bank canonical correlation analysis for implementing a high-speed SSVEP-based
    brain-computer interface", Chen et al., 2015 J.Neural Engl.

    Set a and b to the values they estimated to be the best
    a = 1.25
    b = 0.25

    Parameters
    ----------
    n : int
        index of subband

    Returns
    -------
    weight : float
        weight
    """
    a = 1.25
    b = 0.25
    return np.power(n, -a) + b  # eq. 7


def make_df(results, freqs, phase, n_freq, n_sub, n_blocks, time=None):
    """create dataframe

    Parameters
    ----------
    results : array, shape(n_freq, n_subjects * n_blocks)
        results per trial
    freqs : array, shape(n_freq,)
        The stimulation frequencies
    phase : array, shape(n_freq,)
        The stimulation phases
    n_freq : int
        number of frequencies
    n_sub : int
        number of subjects
    n_blocks : int
        number of blocks
    time : array, shape(n_freq, n_subjects * n_blocks)
        time needed per trial in ms
    Return
    -------
    df : DataFrame, shape(n_trials,['Estimation','Frequency','Phase','Subject','Block'])
        The DataFrame
    """

    list_col_names = ['Estimation', 'Frequency', 'Phase', 'Subject', 'Block']
    df = pd.DataFrame(columns=list_col_names)

    df['Estimation'] = freqs[results.astype(int)].flatten('F')
    df['Frequency'] = np.concatenate(n_sub * n_blocks * [freqs])
    df['Phase'] = np.concatenate(n_sub * n_blocks * [phase])
    if time is not None:
        df['Time'] = (pd.to_timedelta(time.flatten('F'))).astype('timedelta64[ms]')

    for s in range(n_sub):
        df['Subject'][s * n_blocks * n_freq:s * n_blocks * n_freq + n_blocks * n_freq] = np.full(n_blocks * n_freq,
                                                                                                 s + 1, dtype=int)
        for b in range(n_blocks):
            df['Block'][s * n_blocks * n_freq + b * n_freq:s * n_blocks * n_freq + b * n_freq + n_freq] = np.full(
                n_freq, b + 1, dtype=int)

    df['Subject'].astype(int)
    df['Block'].astype(int)
    df['Error'] = (df['Estimation'] - df['Frequency']).abs()
    df['Compare'] = df['Estimation'] == df['Frequency']
    return df


def mk_df(results, threshold, time, rho, max, freqs, n_freq, n_sub, n_blocks):
    """create dataframe

    Parameters
    ----------
    results : array, shape(n_freq, n_subjects * n_blocks)
        classification results per trial
    threshold : array, shape(n_freq, n_subjects * n_blocks)
        results with applied thresholds. rejected trials are stored as -1
    time : array, shape(n_freq, n_subjects * n_blocks)
        time needed per trial in ms
    max : array, shape(n_freq, n_subjects * n_blocks)
        The maximum value per trial
    max : array, shape(n_freq, n_subjects * n_blocks)
        The maximum rho per trial
    freqs : array, shape(n_freq,)
        The stimulation frequencies
    n_freq : int
        number of frequencies
    n_sub : int
        number of subjects
    n_blocks : int
        number of blocks
    Return
    -------
    df : DataFrame, shape(n_trials,['Subject', 'Block', 'Frequency', 'Estimation', 'Threshold', 'Max', 'Rho', 'Compare', 'Time'])
        The DataFrame
    """

    list_col_names = ['Subject', 'Block', 'Frequency', 'Estimation', 'Threshold', 'Max', 'Rho', 'Compare', 'Time']
    df = pd.DataFrame(columns=list_col_names)

    df['Estimation'] = freqs[results.astype(int)].flatten('F')
    df['Threshold'] = threshold.flatten('F')
    df['Max'] = max.flatten('F')
    df['Rho'] = rho.flatten('F')
    df['Frequency'] = np.concatenate(n_sub * n_blocks * [freqs])
    df['Time'] = (pd.to_timedelta(time.flatten('F'))).astype('timedelta64[ms]')

    for s in range(n_sub):
        df['Subject'][s * n_blocks * n_freq:s * n_blocks * n_freq + n_blocks * n_freq] = np.full(n_blocks * n_freq,
                                                                                                 s + 1, dtype=int)
        for b in range(n_blocks):
            df['Block'][s * n_blocks * n_freq + b * n_freq:s * n_blocks * n_freq + b * n_freq + n_freq] = np.full(
                n_freq, b + 1, dtype=int)

    df['Subject'].astype(int)
    df['Block'].astype(int)
    df['Compare'] = df['Estimation'] == df['Frequency']
    return df


def set_style(fig, ax=None):
    sns.despine(ax=ax, top=True, right=True, left=False, bottom=False, offset={'left': 10, 'bottom': 5})

    if ax:
        ax.yaxis.label.set_size(10)
        ax.xaxis.label.set_size(10)
        ax.grid(axis='y', color='C7', linestyle='--', lw=.5)
        ax.tick_params(which='major', direction='out', length=3, width=1, bottom=True, left=True)
        ax.tick_params(which='minor', direction='out', length=2, width=0.5, bottom=True, left=True)
        plt.setp(ax.spines.values(), linewidth=.8)
    return fig, ax


def set_size(fig, a, b):
    fig.set_size_inches(a, b)
    fig.set_tight_layout(True)
    return fig


def itr(df, t):
    m = 40
    p = df / 100
    if p == 100.0:
        p = 0.99
    return (np.log2(m) + p * np.log2(p) + (1 - p) * np.log2((1 - p) / (m - 1))) * 60 / t


def plot_trial(results):
    """plot the passed matrix as heatmap/imshow

    Parameters
    ----------
    results : array, shape(n_freq,n_block*n_subject)
        data that contains the results of classification

    Return
    -------
    fig, ax

    """
    n_freq = np.shape(results)[0]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(results)
    ax.set_ylabel('Trial')
    ax.set_xlabel('Subjects')
    ax.set_xticks(np.arange(0, 210, 6))
    ax.set_xticklabels(np.arange(1, 36))
    set_size(fig, 8, 2.5)
    fig.tight_layout()
    return fig, ax


##Style und Co
def setPgf(bDoPgf):
    if bDoPgf:
        mpl.use("pgf")
        mpl.rcParams.update({
            "savefig.dpi": 300,
            "pgf.texsystem": "pdflatex",
            "font.family": "serif",
            'text.usetex': True,
            'pgf.rcfonts': False,
            "axes.labelsize": 8,  # LaTeX default is 10pt font.
            "legend.fontsize": 8,  # Make the legend/label fonts a little smaller
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pgf.preamble": r"\usepackage[utf8x]{inputenc} \usepackage[T1]{fontenc} \usepackage{cmbright}"
        })


def figsize(scale):
    fig_width_pt = 345./2                          # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
    fig_height = fig_width*golden_mean              # height in inches
    fig_size = [fig_width,fig_height]
    return fig_size


### Lambdas
acc = lambda mat: np.sum(mat[mat > 0]) / (np.size(mat) - np.size(mat[mat == -1])) * 100
standardize = lambda mat: (mat - np.mean(mat, axis=1)[:, None]) / np.std(mat, axis=1)[:, None]
