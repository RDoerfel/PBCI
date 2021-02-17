from functions import *

print("Extended CCA: Tag: " + sTag + ", Subjects: " + str(Ns) + ", Data length: " + str(N_sec))


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
list_el = [str('PZ'), str('PO5'), str('PO3'), str('POz'), str('PO4'), str('PO6'), str('O1'), str('Oz'),
           str('O2')]  # Electrodes to use
vec_ind_el = df_location[df_location['Label'].isin(list_el)].index  # Vector with indexes of electrodes to use
ind_ref_el = df_location['Electrode'][df_location['Label'] == 'Cz'].index[0]  # Index of reference electrode 'Cz'

fs = 250  # sampling frequency in hz
N_pre = int(0.5 * fs)  # pre stim
N_delay = int(0.140 * fs)  # SSVEP delay
N_stim = int(N_sec * fs)  # stimulation
N_start = N_pre + N_delay - 1
N_stop = N_start + N_stim

### Create Reference signals
vec_t = np.arange(-0.5, 5.5, 1 / 250)  # time vector
Nh = 5  # Number of harmonics
Nf = len(vec_freq)  # Number of frequencies
Nb = 6  # Number of Blocks

mat_Y = np.zeros([Nf, Nh * 2, N_stim])  # [Frequency, Harmonics * 2, Samples]

for k in range(0, Nf):
    for i in range(1, Nh + 1):
        mat_Y[k, i - 1, :] = np.sin(2 * np.pi * i * vec_freq[k] * vec_t[N_start:N_stop] + vec_phase[k])
        mat_Y[k, i-1+Nh, :] = np.cos(2 * np.pi * i * vec_freq[k] * vec_t[N_start:N_stop] + vec_phase[k])

Ns = 35
Nb = 6

### Frequency detection using advanced CCA
list_result = []  # list to store the subject wise results
list_time = []  # list to store the time per trial
list_bool_result = []  # list to store the classification as true/false
list_bool_thresh = []  # list to store the classification with thresholds
list_rho = []
list_max = []

num_iter = 0
mat_filtered = np.zeros([Ns, Nb, Nf, 9, N_stim])
for s in range(0, Ns):
    for b in range(0, Nb):
        for f in range(0, Nf):
            # Referencing and baseline correction
            mat_data = preprocess(list_subject_data[s][:, :, f, b], vec_ind_el, ind_ref_el, N_start, N_stop)
            # Filter data
            mat_filt = mne.filter.filter_data(mat_data, fs, 7, 70, method='fir', phase='zero-double', verbose=False)
            mat_filtered[s, b, f, :, :] = mat_filt

for s in range(0, Ns):
    mat_ind_max = np.zeros([Nf, Nb])  # index of maximum cca
    mat_bool = np.zeros([Nf, Nb])
    mat_bool_thresh = np.zeros([Nf, Nb])
    mat_rho = np.zeros([Nf, Nb])
    mat_max = np.zeros([Nf, Nb])
    mat_time = np.zeros([Nf, Nb], dtype='object')  # matrix to store time needed

    t_start = datetime.now()

    # average over subjects
    for b in range(0, Nb):

        # average over subjects
        mat_blocks_dropped = np.delete(mat_filtered[s], b, axis=0)
        mat_X_train = np.mean(mat_blocks_dropped, axis=0)

        for f in range(0, Nf):
            t_trial_start = datetime.now()

            # Apply CCA
            vec_rho = np.zeros(Nf)
            for k in range(0, Nf):
                vec_rho[k] = apply_ext_cca(mat_filtered[s, b, f, :, :], mat_Y[k, :, :], mat_X_train[k, :, :])

            t_trial_end = datetime.now()
            mat_time[f, b] = t_trial_end - t_trial_start
            mat_rho[f, b] = np.max(vec_rho)
            mat_ind_max[f, b] = np.argmax(vec_rho)  # get index of maximum -> frequency -> letter
            num_iter = num_iter + 1

            # apply threshold
            mat_stand = standardize(mat_filt)
            mat_max[f, b] = np.max(np.abs(mat_stand))
            thresh = 6
            if np.max(np.abs(mat_stand)) > thresh:
                # minus 1 if it is going to be removed
                mat_bool_thresh[f, b] = -1

    list_result.append(mat_ind_max)  # store results per subject
    list_time.append(mat_time)  # store results per subject
    list_bool_result.append(mat_bool)
    list_bool_thresh.append(mat_bool_thresh)
    list_rho.append(mat_rho)
    list_max.append(mat_max)

    t_end = datetime.now()
    print("Extended CCA: Elapsed time for subject: " + str(s + 1) + ": " + str((t_end - t_start)), flush=True)

mat_result = np.concatenate(list_result, axis=1)
mat_time = np.concatenate(list_time, axis=1)
mat_b = np.concatenate(list_bool_result, axis=1)
mat_b_thresh = np.concatenate(list_bool_thresh, axis=1)
mat_max = np.concatenate(list_max, axis=1)
mat_rho = np.concatenate(list_rho, axis=1)

### Analysis
accuracy_all = accuracy(vec_freq, mat_result)
accuracy_drop = acc(mat_b_thresh)

print("Extended CCA: accuracy: " + str(accuracy_all))
print("Extended CCA: accuracy dropped: " + str(accuracy_drop))

sTag = '_' + str(sTag)
sSec = '_' + str(N_sec)
sNs = '_' + str(Ns)

np.save(os.path.join(dir_results, 'ext_cca_mat_result' + sSec + sNs + sTag), mat_result)
np.save(os.path.join(dir_results, 'ext_cca_mat_time' + sSec + sNs + sTag), mat_time)
np.save(os.path.join(dir_results, 'ext_cca_mat_b' + sSec + sNs + sTag), mat_b)
np.save(os.path.join(dir_results, 'ext_cca_mat_b_thresh' + sSec + sNs + sTag), mat_b_thresh)
np.save(os.path.join(dir_results, 'ext_cca_mat_max' + sSec + sNs + sTag), mat_max)
np.save(os.path.join(dir_results, 'ext_cca_mat_rho' + sSec + sNs + sTag), mat_rho)
