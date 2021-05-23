import pandas as pd

from functions import *
import argparse
from joblib import Parallel, delayed
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
### Set Working Directory
abspath = os.path.abspath(__file__)
dirname = os.path.dirname(abspath)
os.chdir(dirname)

dir_data = 'data'
dir_figures = 'figures'
dir_results = 'results'

bPgf = False
### Load data

matX = np.load(os.path.join(dir_results, 'cca_mat_rho.npy'))
matX = matX.transpose()
mat_info = np.load(os.path.join(dir_results, 'cca_mat_info.npy'))

Ns = 35
Nf = 40
Nb = 6
vec_accuracy = np.zeros([Ns, 1])
num_iter = 0

### Load and prepare data
dict_freq_phase = loadmat(os.path.join(dir_data, 'Freq_Phase.mat'))
vec_freq = dict_freq_phase['freqs'][0]

vec_length = np.arange(0.25, 5.25, 0.25)

lDf = []
l_df_acc = []
l_df_itr = []

sNs = '_s' + str(Ns)
sTag = '_ext'
vec_length = vec_length[10:11]
for i,l in enumerate(vec_length):
    sSec = '_l' + str(l).replace('.', '_')
    fname_cca_res = 'cca_mat_result' + sSec + sNs + sTag + '.npy'
    fname_cca_rho = 'cca_mat_rho' + sSec + sNs + sTag + '.npy'
    fname_cca_inf = 'cca_mat_info' + sSec + sNs + sTag + '.npy'

    cca_mat_result = np.load(os.path.join(dir_results, fname_cca_res))
    matX = np.load(os.path.join(dir_results, fname_cca_rho)).transpose()
    cca_mat_inf = np.load(os.path.join(dir_results, fname_cca_inf))

    list_col_names = ['Frequency', 'Subject', 'Block', 'Length']
    df = pd.DataFrame(columns=list_col_names)

    df['CCA'] = cca_mat_result.astype(int).flatten('F')

    df['Length'] = l
    df['Subject'] = cca_mat_inf[:, 0]
    df['Block'] = cca_mat_inf[:, 1]
    df['Frequency'] = cca_mat_inf[:, 2]

    df['Subject'].astype(int)
    df['Block'].astype(int)

    ## predict f
    l_pred_clr = []
    l_pred_clnb = []
    for s in range(0, Ns):
        X_train = matX[cca_mat_inf[:, 0] != s]
        y_train = cca_mat_inf[cca_mat_inf[:, 0] != s, 2]
        X_test = matX[cca_mat_inf[:, 0] == s]
        y_test = cca_mat_inf[cca_mat_inf[:, 0] == s, 2]

        clnb = MultinomialNB()
        clr = LogisticRegression(C=100, class_weight='balanced', max_iter=1000, penalty='l2', multi_class='multinomial', solver='newton-cg', tol=1e-4)
        clr.fit(X_train, y_train)
        clnb.fit(X_train, y_train)
        l_pred_clr.append(clr.predict(X_test))
        l_pred_clnb.append(clnb.predict(X_test))
        print("Accuracy LR - %.2f, NB - %.2f for length %.2f in subject %i." % (clr.score(X_test,y_test)*100,clnb.score(X_test,y_test)*100,l,s))

    df['LR'] = np.concatenate(l_pred_clr).astype(int)
    df['NB'] = np.concatenate(l_pred_clnb).astype(int)

    df['bCCA'] = df['CCA'] == df['Frequency']
    df['bLR'] = df['LR'] == df['Frequency']
    df['bNB'] = df['NB'] == df['Frequency']

    df_acc = pd.DataFrame()
    df_itr = pd.DataFrame()

    df_acc['CCA'] = df.groupby(['Subject']).sum()['bCCA'] / (Nb * Nf) * 100
    df_acc['LR'] = df.groupby(['Subject']).sum()['bLR'] / (Nb * Nf) * 100
    df_acc['NB'] = df.groupby(['Subject']).sum()['bNB'] / (Nb * Nf) * 100

    df_itr['CCA'] = df_acc['CCA'].apply((lambda x: itr(x, l + 0.5)))
    df_itr['LR'] = df_acc['LR'].apply((lambda x: itr(x, l + 0.5)))
    df_itr['NB'] = df_acc['NB'].apply((lambda x: itr(x, l + 0.5)))

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
lLabels = ['CCA', 'LR', 'NB']

dict_params = {"estimator": np.mean,
               "ci": 95,
               "err_style": "bars",
               "markers": True,
               "palette": palette[0:3],
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

fig1.savefig(os.path.join(dir_figures, 'accuracy_class.pdf'), dpi=300)
fig1.savefig(os.path.join(dir_figures, 'accuracy_class.png'), dpi=300)
# if bPgf:
#     fig1.savefig(os.path.join(dir_figures, 'accuracy_all.pgf'))
#     plt.close(fig1)

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

fig2.savefig(os.path.join(dir_figures, 'itr_class.pdf'), dpi=300)
fig2.savefig(os.path.join(dir_figures, 'itr_class.png'), dpi=300)

### Evaluation
df_itr_gl = pd.DataFrame()
df_itr_gl = df_itr.groupby('Length').mean()

print(df_itr_gl.max())
print(df_itr_gl.idxmax())

df_acc_gl = pd.DataFrame()
df_acc_gl = df_acc.groupby('Length').mean()

print(df_acc_gl.max())
print(df_acc_gl.idxmax())
