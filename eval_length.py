import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.io import loadmat
import os
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

pd.options.mode.chained_assignment = None  # default='warn'


### Functions
def set_style(fig, ax=None):
    sns.despine(ax=ax, top=True, right=True, left=False, bottom=False, offset={'left': 10, 'bottom': 5})

    if ax:
        ax.yaxis.label.set_size(10)
        ax.xaxis.label.set_size(10)
        ax.grid(axis='y', color='C7', linestyle='--', lw=.3)
        ax.tick_params(which='major', direction='out', length=3, width=0.8, bottom=True, left=True)
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
            "legend.fontsize": 6,  # Make the legend/label fonts a little smaller
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "pgf.preamble": r"\usepackage[utf8x]{inputenc} \usepackage[T1]{fontenc} \usepackage{cmbright}"
        })
    else:
        mpl.use("Qt5Agg")


def figsize(scale):
    fig_width_pt = 345. * 2/3                       # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
    fig_height = fig_width*golden_mean              # height in inches
    fig_size = [fig_width, fig_height]
    return fig_size


### Parser
parser = argparse.ArgumentParser(description='Add some integers.')

parser.add_argument('--subjects', action='store', type=int, default=35,
                    help='Number of subjects to use [1,35].')

parser.add_argument('--tag', action='store', default='ref',
                    help='Tag to add to the files.')

parser.add_argument('--tex', action='store', default=False, type=bool,
                    help='Store files as .pgf or not.')

args = parser.parse_args()
Ns = args.subjects
sTag = args.tag
bPgf = args.tex

print("Evaluation Data Length: " + sTag + ", Subjects: " + str(Ns) + ", Pgf: " + str(bPgf))

### Set Working Directory
abspath = os.path.abspath(__file__)
dirname = os.path.dirname(abspath)
os.chdir(dirname)

dir_data = 'data'
dir_figures = 'figures'
dir_results = 'results'

### Load and prepare data
dict_freq_phase = loadmat(os.path.join(dir_data, 'Freq_Phase.mat'))
vec_freq = dict_freq_phase['freqs'][0]
vec_phase = dict_freq_phase['phases'][0]

sNs = '_' + str(Ns)
sSec = '_' + str(5).replace('.', '_')
if sTag != '':
    sTag = '_' + str(sTag)

n_sub = 35
n_blocks = 6
n_freq = 40

vec_length = np.arange(0.25, 5.25, 0.25)

lDf = []
l_df_acc = []
l_df_itr = []

for l in vec_length:
    sSec = '_' + str(l)
    fname_ext_fbcca = 'ext_fbcca_mat_result' + sSec + sNs + sTag + '.npy'
    fname_ext_cca = 'ext_cca_mat_result' + sSec + sNs + sTag + '.npy'
    fname_fbcca = 'fbcca_mat_result' + sSec + sNs + sTag + '.npy'
    fname_cca = 'cca_mat_result' + sSec + sNs + sTag + '.npy'

    cca_mat_result = np.load(os.path.join(dir_results, fname_cca))
    fbcca_mat_result = np.load(os.path.join(dir_results, fname_fbcca))
    ext_cca_mat_result = np.load(os.path.join(dir_results, fname_ext_cca))
    ext_fbcca_mat_result = np.load(os.path.join(dir_results, fname_ext_fbcca))

    list_col_names = ['Frequency', 'Subject', 'Block', 'Length']
    df = pd.DataFrame(columns=list_col_names)

    df['CCA'] = vec_freq[cca_mat_result.astype(int)].flatten('F')
    df['FBCCA'] = vec_freq[fbcca_mat_result.astype(int)].flatten('F')
    df['Ext_CCA'] = vec_freq[ext_cca_mat_result.astype(int)].flatten('F')
    df['Ext_FBCCA'] = vec_freq[ext_fbcca_mat_result.astype(int)].flatten('F')

    df['Frequency'] = np.concatenate(n_sub * n_blocks * [vec_freq])
    df['Length'] = l

    for s in range(n_sub):
        df['Subject'][s * n_blocks * n_freq:s * n_blocks * n_freq + n_blocks * n_freq] = np.full(n_blocks * n_freq,
                                                                                                 s + 1, dtype=int)
        for b in range(n_blocks):
            df['Block'][s * n_blocks * n_freq + b * n_freq:s * n_blocks * n_freq + b * n_freq + n_freq] = np.full(
                n_freq, b + 1, dtype=int)

    df['Subject'].astype(int)
    df['Block'].astype(int)

    df['bCCA'] = df['CCA'] == df['Frequency']
    df['bFBCCA'] = df['FBCCA'] == df['Frequency']
    df['bExt_CCA'] = df['Ext_CCA'] == df['Frequency']
    df['bExt_FBCCA'] = df['Ext_FBCCA'] == df['Frequency']

    df_acc = pd.DataFrame()
    df_itr = pd.DataFrame()

    df_acc['CCA'] = df.groupby(['Subject']).sum()['bCCA'] / (n_blocks * n_freq) * 100
    df_acc['FBCCA'] = df.groupby(['Subject']).sum()['bFBCCA'] / (n_blocks * n_freq) * 100
    df_acc['Ext CCA'] = df.groupby(['Subject']).sum()['bExt_CCA'] / (n_blocks * n_freq) * 100
    df_acc['Ext FBCCA'] = df.groupby(['Subject']).sum()['bExt_FBCCA'] / (n_blocks * n_freq) * 100

    df_itr['CCA'] = df_acc['CCA'].apply((lambda x: itr(x, l + 0.5)))
    df_itr['FBCCA'] = df_acc['FBCCA'].apply((lambda x: itr(x, l + 0.5)))
    df_itr['Ext CCA'] = df_acc['Ext CCA'].apply((lambda x: itr(x, l + 0.5)))
    df_itr['Ext FBCCA'] = df_acc['Ext FBCCA'].apply((lambda x: itr(x, l + 0.5)))

    df_acc['Length'] = l
    df_itr['Length'] = l

    l_df_acc.append(df_acc)
    l_df_itr.append(df_itr)

    lDf.append(df)

df = pd.concat(lDf)
df_acc = pd.concat(l_df_acc)
df_itr = pd.concat(l_df_itr)

df_melt_acc = df_acc.melt(['Length'])
df_melt_itr = df_itr.melt(['Length'])
df_melt_acc = df_melt_acc.rename(columns={'variable': 'Method', 'value': 'Accuracy'})
df_melt_itr = df_melt_itr.rename(columns={'variable': 'Method', 'value': 'ITR'})

## Plots
palette = sns.color_palette('colorblind')
lLabels = ['CCA', 'FBCCA', 'Extended \n CCA', 'Extended \n FBCCA']

dict_params = {"estimator": np.mean,
               "ci": 95,
               "err_style": "bars",
               "markers": True,
               "palette": palette[0:4],
               "linewidth": 0.8,
               "err_kws": {"capsize": 2, "capthick": .8, "lw": .5}}

setPgf(bPgf)
fsize = figsize(0.9)
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
sns.lineplot(ax=ax1, data=df_melt_acc, x='Length', y='Accuracy', hue='Method', **dict_params)
ax1.set_ylabel('Accuracy in %')
ax1.set_xlabel('Length in s')

ax1.yaxis.set_major_locator(MultipleLocator(25))
ax1.xaxis.set_major_locator(MultipleLocator(1))
ax1.xaxis.set_minor_locator(MultipleLocator(0.5))
ax1.set_xlim([0, 5])

handles, labels = ax1.get_legend_handles_labels()
ax1.legend(handles=handles, labels=labels)

set_style(fig1, ax1)
set_size(fig1, 2.8, 2)

fig1.savefig(os.path.join(dir_figures, 'accuracy_all.pdf'), dpi=300)
fig1.savefig(os.path.join(dir_figures, 'accuracy_all.png'), dpi=300)
if bPgf:
    fig1.savefig(os.path.join(dir_figures, 'accuracy_all.pgf'))
    plt.close(fig1)

setPgf(bPgf)
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
sns.lineplot(ax=ax2, data=df_melt_itr, x='Length', y='ITR', hue='Method', **dict_params)
ax2.set_ylabel('ITR in Bits/min')
handles, labels = ax2.get_legend_handles_labels()
ax2.legend(handles=handles, labels=labels)
ax2.set_xlabel('Length in s')
ax2.yaxis.set_major_locator(MultipleLocator(50))
ax2.xaxis.set_major_locator(MultipleLocator(1))
ax2.xaxis.set_minor_locator(MultipleLocator(0.5))
ax2.set_xlim([0, 5])

set_style(fig2, ax2)
set_size(fig2, 2.8, 2)

fig2.savefig(os.path.join(dir_figures, 'itr_all.pdf'), dpi=300)
fig2.savefig(os.path.join(dir_figures, 'itr_all.png'), dpi=300)
if bPgf:
    fig2.savefig(os.path.join(dir_figures, 'itr_all.pgf'), dpi=300)
    plt.close(fig2)


### Evaluation
df_itr_gl = pd.DataFrame()
df_itr_gl = df_itr.groupby('Length').mean()

print(df_itr_gl.max())
print(df_itr_gl.idxmax())

df_acc_gl = pd.DataFrame()
df_acc_gl = df_acc.groupby('Length').mean()

print(df_acc_gl.max())
print(df_acc_gl.idxmax())
