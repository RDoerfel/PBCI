from functions import *
import argparse
from joblib import Parallel, delayed
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

### Set Working Directory
abspath = os.path.abspath(__file__)
dirname = os.path.dirname(abspath)
os.chdir(dirname)

dir_data = 'data'
dir_figures = 'figures'
dir_results = 'results'

### Load data

matX = np.load(os.path.join(dir_results, 'cca_mat_rho.npy'))
matX = matX.transpose()
mat_info = np.load(os.path.join(dir_results, 'cca_mat_info.npy'))

Ns = 35
Nf = 40
Nb = 6
vec_accuracy = np.zeros([Ns,1])
num_iter = 0

for s in range(0, Ns):
    X_train = matX[mat_info[:, 0] != s]
    y_train = mat_info[mat_info[:, 0] != s, 2]
    X_test = matX[mat_info[:, 0] == s]
    y_test = mat_info[mat_info[:, 0] == s, 2]
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    vec_accuracy[s] = clf.score(X_test, y_test)

print("%0.2f accuracy with a standard deviation of %0.2f" % (vec_accuracy.mean(), vec_accuracy.std()))

