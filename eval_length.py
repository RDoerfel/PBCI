from functions import *
import argparse

pd.options.mode.chained_assignment = None  # default='warn'

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
lDfS = []

for l in vec_length:
    sSec = '_' + str(l)
    fname_ext_fbcca = 'ext_fbcca_mat_result' + sSec + sNs + '_ref' + '.npy'
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

    df_subject = pd.DataFrame()

    df_subject['Accuracy CCA'] = df.groupby(['Subject']).sum()['bCCA'] / (n_blocks * n_freq) * 100
    df_subject['Accuracy FBCCA'] = df.groupby(['Subject']).sum()['bFBCCA'] / (n_blocks * n_freq) * 100
    df_subject['Accuracy Ext CCA'] = df.groupby(['Subject']).sum()['bExt_CCA'] / (n_blocks * n_freq) * 100
    df_subject['Accuracy Ext FBCCA'] = df.groupby(['Subject']).sum()['bExt_FBCCA'] / (n_blocks * n_freq) * 100

    df_subject['ITR CCA'] = df_subject['Accuracy CCA'].apply((lambda x: itr(x, l + 0.5)))
    df_subject['ITR FBCCA'] = df_subject['Accuracy FBCCA'].apply((lambda x: itr(x, l + 0.5)))
    df_subject['ITR Ext CCA'] = df_subject['Accuracy Ext CCA'].apply((lambda x: itr(x, l + 0.5)))
    df_subject['ITR Ext FBCCA'] = df_subject['Accuracy Ext FBCCA'].apply((lambda x: itr(x, l + 0.5)))

    df_subject['Length'] = l

    lDfS.append(df_subject)
    lDf.append(df)

df = pd.concat(lDf)
df_s = pd.concat(lDfS)

palette = sns.color_palette('Greys')
lLabels = ['CCA', 'FBCCA', 'Extended \n CCA', 'Extended \n FBCCA']

setPgf(bPgf)
figsize = figsize(0.9)
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
sns.lineplot(ax=ax1, data=df_s, x='Length', y='Accuracy CCA', estimator=np.median, ci=95, err_style='bars', markers=True)
sns.lineplot(ax=ax1, data=df_s, x='Length', y='Accuracy FBCCA', estimator=np.median, ci=95, err_style='bars', markers=True)
sns.lineplot(ax=ax1, data=df_s, x='Length', y='Accuracy Ext CCA', estimator=np.median, ci=95, err_style='bars', markers=True)
sns.lineplot(ax=ax1, data=df_s, x='Length', y='Accuracy Ext FBCCA', estimator=np.median, ci=95, err_style='bars', markers=True)
ax1.set_ylabel('Accuracy in %')
set_style(fig1, ax1)
set_size(fig1, figsize[1], figsize[0])

setPgf(bPgf)
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
sns.lineplot(ax=ax2, data=df_s, x='Length', y='ITR CCA', estimator=np.median, ci=95, err_style='bars', markers=True)
sns.lineplot(ax=ax2, data=df_s, x='Length', y='ITR FBCCA', estimator=np.median, ci=95, err_style='bars', markers=True)
sns.lineplot(ax=ax2, data=df_s, x='Length', y='ITR Ext CCA', estimator=np.median, ci=95, err_style='bars', markers=True)
sns.lineplot(ax=ax2, data=df_s, x='Length', y='ITR Ext FBCCA', estimator=np.median, ci=95, err_style='bars', markers=True)
ax2.set_ylabel('ITR ib Bits/min')
set_style(fig2, ax2)
set_size(fig2, figsize[1], figsize[0])