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
    clr = LogisticRegression(C=100, class_weight='balanced', max_iter=1000, penalty='l2', multi_class='multinomial', solver='newton-cg', tol=1e-4)
    clr.fit(X_train, y_train)
    vec_accuracy[s] = clr.score(X_test, y_test)
print("LogReg: %0.2f accuracy" % (vec_accuracy.mean()))

# clr = LogisticRegression(class_weight='balanced', max_iter=1000, penalty='l2', multi_class='multinomial',
#                          solver='newton-cg', tol=1e-4, C=100)
#
# cknn = KNeighborsClassifier(metric='minkowski', weights='distance', p=1)
#
# cbayes = MultinomialNB()
#
# # define search parameters
# cv_inner = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
#
# C = [0.001, 0.01, 0.1, 1.0, 10, 100, 1000, 10000]
# regularization = {'C': C}
# neighbors = {'n_neighbors': np.arange(5,30)}
# alpha = {'alpha': np.arange(0,5,0.1)}
#
#
# search_clr = GridSearchCV(clr, scoring='accuracy', param_grid=regularization, cv=cv_inner, refit=True,
#                           n_jobs=-1)
# search_cknn = GridSearchCV(cknn, scoring='accuracy', param_grid=neighbors, cv=cv_inner, refit=True, n_jobs=-1)
# search_cnb = GridSearchCV(cbayes, scoring='accuracy', param_grid=alpha, cv=cv_inner, refit=True, n_jobs=-1)
#
# X_train, X_test, y_train, y_test = train_test_split(matX, mat_info[:,2], test_size=0.33, random_state=42)
#
# search_clr.fit(matX, mat_info[:,2])
# search_cknn.fit(matX, mat_info[:,2])
# search_cnb.fit(matX, mat_info[:,2])
#
# print("LogReg: %0.2f accuracy with cfg=%s" % (search_clr.best_score_, search_clr.best_params_))
# print("KNN: %0.2f accuracy with cfg=%s" % (search_cknn.best_score_, search_cknn.best_params_))
# print("NB: %0.2f accuracy with cfg=%s" % (search_cnb.best_score_, search_cnb.best_params_))
