from functions import *
import argparse
from sklearn.linear_model import LogisticRegression

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


def load_data(Ns, N_sec, sMethod, sTag, dir_results):
    sSec = '_l' + str(N_sec).replace('.', '_')
    sNs = '_s' + str(Ns)
    if sTag != '':
        sTag = '_' + str(sTag)
    fRho = sMethod + '_mat_rho' + sSec + sNs + sTag + '.npy'
    fInfo = sMethod + '_mat_info' + sSec + sNs + sTag + '.npy'
    fRes = sMethod + '_mat_result' + sSec + sNs + sTag + '.npy'

    mat_result = np.load(os.path.join(dir_results, fRes))
    mat_rho = np.load(os.path.join(dir_results, fRho)).transpose()
    mat_info = np.load(os.path.join(dir_results, fInfo))

    return mat_rho, mat_result, mat_info


### Set Working Directory
abspath = os.path.abspath(__file__)
dirname = os.path.dirname(abspath)
os.chdir(dirname)

dir_data = 'data'
dir_figures = 'figures'
dir_results = 'results'

### CLI
parser = argparse.ArgumentParser(description='Add some integers.')

parser.add_argument('--method', action='store', default='cca',
                    help='Tag to add to the files.')
parser.add_argument('--tag', action='store', default='ext',
                    help='Tag to add to the files.')
parser.add_argument('--length', action='store', type=float, default=5.0,
                    help='Length of data to take into account (0,5].')

args = parser.parse_args()
sTag = args.tag
sMethod = args.method
N_sec = args.length

### Load Data
Ns = 35
Nf = 40
Nb = 6


mat_rho_1_75, mat_result_1_75, mat_info_1_75 = load_data(Ns,1.75,sMethod,sTag,dir_results)
mat_rho_5, mat_result_5, mat_info_5 = load_data(Ns,5.0,sMethod,sTag,dir_results)

i = 5
fig1 = plt.figure()
ax11 = fig1.add_subplot(211)
ax12 = fig1.add_subplot(212)

vec_x = np.arange(0, 40)
mean_1_75 = np.mean(mat_rho_1_75[mat_info_1_75[:, 2] == i, :], axis=0)
std_1_75 = np.std(mat_rho_1_75[mat_info_1_75[:, 2] == i, :], axis=0)
mean_5 = np.mean(mat_rho_5[mat_info_5[:, 2] == i, :], axis=0)
std_5 = np.std(mat_rho_5[mat_info_5[:, 2] == i, :], axis=0)

ax11.fill_between(x=vec_x, y1=mean_1_75 + std_1_75, y2=mean_1_75 - std_1_75, color='C7',alpha=.5)
ax11.plot(mean_1_75)
ax12.fill_between(x=vec_x, y1=mean_5 + std_5, y2=mean_5 - std_5, color='C7', alpha=.5)
ax12.plot(mean_5)
set_style(fig1)

### Test new classification
l_pred_clr_5 = []
l_pred_clr_1_75 = []

for s in range(0, Ns):
    X_train_5 = mat_rho_5[mat_info_5[:, 0] != s]
    y_train_5 = mat_info_5[mat_info_5[:, 0] != s, 2]
    X_train_1_75 = mat_rho_1_75[mat_info_1_75[:, 0] != s]
    y_train_1_75 = mat_info_1_75[mat_info_1_75[:, 0] != s, 2]
    X_test_1_75 = mat_rho_1_75[mat_info_1_75[:, 0] == s]
    y_test_1_75 = mat_info_1_75[mat_info_1_75[:, 0] == s, 2]

    # configure the cross-validation procedure

    clr_5 = LogisticRegression(C=10, class_weight='balanced', max_iter=1000, penalty='l2', multi_class='multinomial',
                             solver='newton-cg', tol=1e-4)
    clr_1_75 = LogisticRegression(C=10, class_weight='balanced', max_iter=1000, penalty='l2', multi_class='multinomial',
                             solver='newton-cg', tol=1e-4)
    clr_5.fit(X_train_5, y_train_5)
    clr_1_75.fit(X_train_1_75, y_train_1_75)
    l_pred_clr_5.append(clr_5.predict(X_test_1_75))
    l_pred_clr_1_75.append(clr_1_75.predict(X_test_1_75))

    print("Eval for subject %i of %i" % (s, Ns))
    print("LR 5->1.75: Accuracy %.2f" % (clr_5.score(X_test_1_75, y_test_1_75) * 100))
    print("LR 1.75->1.75: Accuracy %.2f" % (clr_1_75.score(X_test_1_75, y_test_1_75) * 100))

df = pd.DataFrame()
df_acc = pd.DataFrame()
df_itr = pd.DataFrame()

df['Subject'] = mat_info_5[:, 0]
df['Block'] = mat_info_5[:, 1]
df['Frequency'] = mat_info_5[:, 2]
df['Subject'].astype(int)
df['Block'].astype(int)
df['LR 5'] = np.concatenate(l_pred_clr_5).astype(int)
df['bLR 5'] = df['LR 5'] == df['Frequency']
df['LR 1_75'] = np.concatenate(l_pred_clr_1_75).astype(int)
df['bLR 1_75'] = df['LR 1_75'] == df['Frequency']
df_acc['LR 5'] = df.groupby(['Subject']).sum()['bLR 5'] / (Nb * Nf) * 100
df_acc['LR 1_75'] = df.groupby(['Subject']).sum()['bLR 1_75'] / (Nb * Nf) * 100

df_itr['LR 5'] = df_acc['LR 5'].apply((lambda x: itr(x, N_sec + 0.5)))
df_itr['LR 1_75'] = df_acc['LR 1_75'].apply((lambda x: itr(x, N_sec + 0.5)))

### Evaluation
df_itr_gl = pd.DataFrame()
df_itr_gl = df_itr.mean()

print(df_itr_gl.max())
print(df_itr_gl.idxmax())

df_acc_gl = pd.DataFrame()
df_acc_gl = df_acc.mean()

print(df_acc_gl.max())
print(df_acc_gl.idxmax())