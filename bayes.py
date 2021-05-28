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

### Parser
parser = argparse.ArgumentParser(description='Add some integers.')

parser.add_argument('--method', action='store', default='cca',
                    help='Tag to add to the files.')
parser.add_argument('--tag', action='store', default='ref',
                    help='Tag to add to the files.')

args = parser.parse_args()
sTag = args.tag
sMethod = args.method

print("Classification: " + sMethod)


### functions
def train_clr(X_train, X_test, y_train, y_test):
    clr = LogisticRegression(C=10, class_weight='balanced', max_iter=1000, penalty='l2', multi_class='multinomial',
                             solver='newton-cg', tol=1e-4)
    clr.fit(X_train, y_train)
    return clr.predict(X_test)


### Set Working Directory
abspath = os.path.abspath(__file__)
dirname = os.path.dirname(abspath)
os.chdir(dirname)

dir_data = 'data'
dir_figures = 'figures'
dir_results = 'results'

bPgf = False
### Load data

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

for i, l in enumerate(vec_length):
    sSec = '_l' + str(l).replace('.', '_')

    fRho = sMethod + '_mat_rho' + sSec + sNs + sTag + '.npy'
    fInfo = sMethod + '_mat_info' + sSec + sNs + sTag + '.npy'
    fRes = sMethod + '_mat_result' + sSec + sNs + sTag + '.npy'

    mat_result = np.load(os.path.join(dir_results, fRes))
    mat_info = np.load(os.path.join(dir_results, fInfo))

    if sMethod == 'fbcca' or sMethod == 'ext_fbcca':
        mat_X = np.load(os.path.join(dir_results, fRho))
    else:
        mat_X = np.load(os.path.join(dir_results, fRho)).transpose()

    list_col_names = ['Frequency', 'Subject', 'Block', 'Length']
    df = pd.DataFrame(columns=list_col_names)

    df[sMethod] = mat_result.astype(int).flatten('F')

    df['Length'] = l
    df['Subject'] = mat_info[:, 0]
    df['Block'] = mat_info[:, 1]
    df['Frequency'] = mat_info[:, 2]

    df['Subject'].astype(int)
    df['Block'].astype(int)

    l_pred_clr = Parallel(n_jobs=-1)(
        delayed(train_clr)(mat_X[mat_info[:, 0] != s], mat_X[mat_info[:, 0] == s], mat_info[mat_info[:, 0] != s, 2],
                           mat_info[mat_info[:, 0] == s, 2]) for s in range(0, Ns))

    df['LR'] = np.concatenate(l_pred_clr).astype(int)

    df['b' + sMethod] = df[sMethod] == df['Frequency']
    df['bLR'] = df['LR'] == df['Frequency']

    df_acc = pd.DataFrame()
    df_itr = pd.DataFrame()

    df_acc[sMethod] = df.groupby(['Subject']).sum()['b' + sMethod] / (Nb * Nf) * 100
    df_acc['LR'] = df.groupby(['Subject']).sum()['bLR'] / (Nb * Nf) * 100

    df_itr[sMethod] = df_acc[sMethod].apply((lambda x: itr(x, l + 0.5)))
    df_itr['LR'] = df_acc['LR'].apply((lambda x: itr(x, l + 0.5)))

    df_acc['Length'] = l
    df_itr['Length'] = l

    l_df_acc.append(df_acc)
    l_df_itr.append(df_itr)

    lDf.append(df)
    print("Length %.2f of %i" % (l, len(vec_length)))

df = pd.concat(lDf)
df_acc = pd.concat(l_df_acc)
df_itr = pd.concat(l_df_itr)

df_melt_acc = df_acc.melt(['Length'])
df_melt_itr = df_itr.melt(['Length'])
df_melt_acc = df_melt_acc.rename(columns={'variable': 'Method', 'value': 'Accuracy'})
df_melt_itr = df_melt_itr.rename(columns={'variable': 'Method', 'value': 'ITR'})

## Plots
palette = sns.color_palette('colorblind')
lLabels = [sMethod, 'LR']

dict_params = {"estimator": np.median,
               "ci": 'sd',
               "err_style": "bars",
               "markers": True,
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

fig1.savefig(os.path.join(dir_figures, 'accuracy_class_' + sMethod + '.pdf'), dpi=300)
fig1.savefig(os.path.join(dir_figures, 'accuracy_class_' + sMethod + '.png'), dpi=300)
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

fig2.savefig(os.path.join(dir_figures, 'itr_class_' + sMethod + '.pdf'), dpi=300)
fig2.savefig(os.path.join(dir_figures, 'itr_class_' + sMethod + '.png'), dpi=300)

### Evaluation
df_itr_gl = pd.DataFrame()
df_itr_gl = df_itr.groupby('Length').mean()

print(df_itr_gl.max())
print(df_itr_gl.idxmax())

df_acc_gl = pd.DataFrame()
df_acc_gl = df_acc.groupby('Length').mean()

print(df_acc_gl.max())
print(df_acc_gl.idxmax())
