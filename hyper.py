import numpy as np
import pandas as pd

from functions import *
import argparse
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from sklearn.svm import LinearSVC

### Parser
parser = argparse.ArgumentParser(description='Add some integers.')

parser.add_argument('--method', action='store', default='ref',
                    help='Tag to add to the files.')
parser.add_argument('--tag', action='store', default='ref',
                    help='Tag to add to the files.')
parser.add_argument('--length', action='store', type=float, default=5,
                    help='Length of data to take into account (0,5].')

args = parser.parse_args()
sTag = args.tag
sMethod = args.method
N_sec = args.length

### Set Working Directory
abspath = os.path.abspath(__file__)
dirname = os.path.dirname(abspath)
os.chdir(dirname)

dir_data = 'data'
dir_figures = 'figures'
dir_results = 'results'

### Load Data
Ns = 35
Nf = 40
Nb = 6

sSec = '_l' + str(N_sec).replace('.', '_')
sNs = '_s' + str(Ns)
if sTag != '':
    sTag = '_' + str(sTag)

fRho = sMethod + '_mat_rho' + sSec + sNs + sTag + '.npy'
fInfo = sMethod + '_mat_info' + sSec + sNs + sTag + '.npy'
fRes = sMethod + '_mat_result' + sSec + sNs + sTag + '.npy'

mat_result = np.load(os.path.join(dir_results, fRes))
mat_X = np.load(os.path.join(dir_results, fRho)).transpose()
mat_info = np.load(os.path.join(dir_results, fInfo))

## predict f
l_pred_clr = []
l_pred_clsvc = []

mat_param = np.zeros([Ns, 2])

for s in range(0, Ns):
    X_train = mat_X[mat_info[:, 0] != s]
    y_train = mat_info[mat_info[:, 0] != s, 2]
    X_test = mat_X[mat_info[:, 0] == s]
    y_test = mat_info[mat_info[:, 0] == s, 2]

    # configure the cross-validation procedure
    cv_inner = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

    C = [0.001, 0.01, 0.1, 1.0, 10, 100, 1000, 10000]

    regularization = {'C': C}

    clr = LogisticRegression(C=100, class_weight='balanced', max_iter=1000, penalty='l2', multi_class='multinomial',
                             solver='newton-cg', tol=1e-4)
    clsvc = LinearSVC(penalty='l2', loss='squared_hinge', dual=False, tol=0.0001, C=1.0, multi_class='ovr',
                      fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None,
                      max_iter=1000)

    search_clr = GridSearchCV(clr, scoring='accuracy', param_grid=regularization, cv=cv_inner, refit=True,
                              n_jobs=-1)
    search_clsvc = GridSearchCV(clsvc, scoring='accuracy', param_grid=regularization, cv=cv_inner, refit=True,
                              n_jobs=-1)

    search_clr.fit(X_train, y_train)
    search_clsvc.fit(X_train, y_train)

    l_pred_clr.append(search_clr.predict(X_test))
    l_pred_clsvc.append(search_clsvc.predict(X_test))

    print("Eval for subject %i, length %.2f" % (s, N_sec))
    print("LR: Inner Accuracy %.2f, Outer Accuracy %.2f, Config: %s" % (
        search_clr.best_score_ * 100, search_clr.score(X_test, y_test) * 100, search_clr.best_estimator_))
    print("SVC: Inner Accuracy %.2f, Outer Accuracy %.2f, Config: %s" % (
        search_clsvc.best_score_ * 100, search_clsvc.score(X_test, y_test) * 100, search_clsvc.best_estimator_))

df = pd.DataFrame()
df_acc = pd.DataFrame()
df_itr = pd.DataFrame()

df['Length'] = N_sec
df['Subject'] = mat_info[:, 0]
df['Block'] = mat_info[:, 1]
df['Frequency'] = mat_info[:, 2]
df['Subject'].astype(int)
df['Block'].astype(int)
df['LR'] = np.concatenate(l_pred_clr).astype(int)
df['SVC'] = np.concatenate(l_pred_clsvc).astype(int)
df['bLR'] = df['LR'] == df['Frequency']
df['bSVC'] = df['SVC'] == df['Frequency']

df_acc['LR'] = df.groupby(['Subject']).sum()['bLR'] / (Nb * Nf) * 100
df_acc['SVC'] = df.groupby(['Subject']).sum()['bSVC'] / (Nb * Nf) * 100

df_itr['LR'] = df_acc['LR'].apply((lambda x: itr(x, N_sec + 0.5)))
df_itr['SVC'] = df_acc['SVC'].apply((lambda x: itr(x, N_sec + 0.5)))
