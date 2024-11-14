# %%
import pandas as pd
from utils import *

X_train, y_train, X_test = load_Xtrain_ytrain_Xtest()


# %%
processing = "X_test"
X = X_test if processing == "X_test" else X_train

# %%
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(-100, 100))

# Scale each row individually for X_train
X = X.apply(lambda row: pd.Series(scaler.fit_transform(row.values.reshape(-1, 1)).ravel()), axis=1)

# Scale each row individually for X_test
X.shape



# %%
import biosppy.signals.ecg as ecg
import numpy as np

def extract_ecg_features(data, i):
    row = data.loc[i].dropna().to_numpy(dtype='float32')
    ts, filtered, rpeaks, templates_ts, templates, heart_rate_ts, heart_rate = ecg.ecg(signal=row, sampling_rate=300, show=False)

    return ts, filtered, rpeaks, templates_ts, templates, heart_rate_ts, heart_rate


ecg_features = ['ts', 'filtered', 'rpeaks', 'templates_ts', 'mean_templates', 'heart_rate_ts', 'heart_rate']
ecg_df = pd.DataFrame(columns=ecg_features)
ecg_df.index.name = 'id'
for i in range(X.shape[0]):
    ts, filtered, rpeaks, templates_ts, templates, heart_rate_ts, heart_rate = extract_ecg_features(X, i)
    mean_templates = np.mean(templates, axis=0)
    ecg_df.loc[len(ecg_df)] = [ts, filtered, rpeaks, templates_ts, mean_templates, heart_rate_ts, heart_rate]
    print(i, flush=True)

ecg_df.to_csv("features/ecg_data.csv")

# %%
from numpy.fft import fft

def extract_fft(data, fs=300):
     # Compute the FFT of the signal
    if(data.shape[0] == 0):
        return np.array([])
    fft_values = fft(data)
    fft_values = 2.0*np.abs(fft_values[:fs//2])/len(data)

    return fft_values

import numpy as np

fft_features = ['fft_' + e for e in ecg_features]
fft_df = pd.DataFrame(columns=fft_features, index=ecg_df.index)
for ec in ecg_features:
    fft_df['fft_' + ec] = ecg_df[ec].map(extract_fft)
    print(ec, end='\r')

fft_df.to_csv("features/fft_data.csv")

# %%
from scipy.signal import welch

def extract_psd(data, fs=300):
    if(data.shape[0] == 0):
        return np.array([])
    freqs, psd_values = welch(data, fs)
    return psd_values

psd_features = ['psd_' + e for e in ecg_features]
psd_df = pd.DataFrame(columns=psd_features, index=ecg_df.index)
for ec in ecg_features:
    psd_df['psd_' + ec] = ecg_df[ec].map(extract_psd)
    print(ec, end='\r')

psd_df.to_csv("features/psd_data.csv")

# %%
all_data = pd.concat([ecg_df, fft_df, psd_df], axis=1)


# %%
from scipy.stats import skew, kurtosis

def extract_time_features(data, signal_name):
    features = {}
    if(type(data) != np.ndarray):
        return features
    if(data.shape[0] == 0):
        return features

    # Initialize dictionary to hold features

    # Basic statistics
    features[signal_name + '_mean'] = np.mean(data)
    features[signal_name + '_median'] = np.median(data)
    features[signal_name + '_std'] = np.std(data)
    features[signal_name + '_var'] = np.var(data)
    features[signal_name + '_max'] = np.max(data)
    features[signal_name + '_min'] = np.min(data)
    features[signal_name + '_rms'] = np.sqrt(np.mean(np.square(data)))
    features[signal_name + '_peak_to_peak'] = features[signal_name + '_max'] - features[signal_name + '_min']
    features[signal_name + '_skewness'] = skew(data)
    features[signal_name + '_kurtosis'] = kurtosis(data)

    # Zero Crossing Rate
    features[signal_name + '_zero_crossing_rate'] = ((data[:-1] * data[1:]) < 0).sum()

    # Signal Magnitude Area
    features[signal_name + '_sma'] = np.sum(np.abs(data))

    # Energy
    features[signal_name + '_energy'] = np.sum(np.square(data))

    # Entropy
    p_signal = data / np.sum(data)  # normalize signal
    features[signal_name + '_entropy'] = -np.sum(p_signal*np.log2(p_signal))

    # Crest Factor
    features[signal_name + '_crest_factor'] = features[signal_name + '_max'] / features[signal_name + '_rms']

    # Impulse Factor
    features[signal_name + '_impulse_factor'] = features[signal_name + '_max'] / features[signal_name + '_mean']

    # Shape Factor
    features[signal_name + '_shape_factor'] = features[signal_name + '_rms'] / (np.sum(np.abs(data)) / len(data))

    # Clearance Factor
    features[signal_name + '_clearance_factor'] = features[signal_name + '_max'] / np.sqrt(np.mean(np.square(np.abs(data))))

    # Return the extracted features
    return features


time_df = pd.DataFrame(index=ecg_df.index)
for ec in all_data.columns.to_list():
    tf_features = all_data[ec].apply(lambda x : extract_time_features(x, ec)).apply(pd.Series)
    time_df = pd.concat([time_df, tf_features], axis=1)
    print(ec, end='\r')
        
time_df.to_csv("features/time_features.csv")

# %%
from scipy.stats import gmean
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')
def extract_frequency_features(data, signal_name, fs=300):
    features = {}
    if(type(data) != np.ndarray):
        return features
    if(data.shape[0] == 0):
        return features

    # Compute the power spectral density of the signal
    freqs, psd_values = welch(data, fs=fs)

    # Compute the peak frequency
    features[signal_name + '_peak_freq'] = freqs[np.argmax(psd_values)]

    # Compute the median frequency
    features[signal_name + '_median_freq'] = freqs[len(freqs)//2]

    # Compute the spectral entropy
    features[signal_name + '_spectral_entropy'] = entropy(psd_values)

    # Compute the spectral energy
    spectral_energy = np.sum(psd_values)
    features[signal_name + '_spectral_energy'] = spectral_energy

    # Compute the spectral centroid
    spectral_centroid = np.sum(freqs * psd_values) / np.sum(psd_values)
    features[signal_name + '_spectral_centroid'] = spectral_centroid

    # Compute the spectral spread
    spectral_spread = np.sqrt(np.sum((freqs - spectral_centroid)**2 * psd_values) / np.sum(psd_values))
    features[signal_name + '_spectral_spread'] = spectral_spread

    # Compute the spectral flatness
    features[signal_name + '_spectral_flatness'] = gmean(psd_values) / np.mean(psd_values)


    # Return the extracted features
    return features

frequency_df = pd.DataFrame(index=ecg_df.index)

for ec in all_data.columns.to_list():
    ff = all_data[ec].apply(lambda x : extract_frequency_features(x, ec)).apply(pd.Series)
    frequency_df = pd.concat([frequency_df, ff], axis=1)
    print(ec, end='\r')

frequency_df.to_csv("features/frequency_features.csv")

# %%
def calculate_poincare_descriptors(data, signal_name):
    features = {}

    if(type(data) != np.ndarray):
        return features
    if(data.shape[0] == 0):
        return features

    # Calculate the differences between successive RR intervals
    rr_diff = np.diff(data)

    # Calculate SD1 and SD2
    features[signal_name + '_SD1'] = np.sqrt(np.std(rr_diff) ** 2 * 0.5)
    features[signal_name + '_SD2'] = np.sqrt(2 * np.std(data) ** 2 - 0.5 * np.std(rr_diff) ** 2)

    # Calculate SD1/SD2 ratio
    features[signal_name + '_SD1/SD2'] = features[signal_name + '_SD1'] / features[signal_name + '_SD2']

    # Calculate the area of the Poincaré ellipse
    features[signal_name + '_ellipse_area'] = np.pi * features[signal_name + '_SD1'] * features[signal_name + '_SD2']

    return features

poincarre_df = pd.DataFrame(index=ecg_df.index)

for ec in all_data.columns.to_list():
    ff = all_data[ec].apply(lambda x : calculate_poincare_descriptors(x, ec)).apply(pd.Series)
    poincarre_df = pd.concat([poincarre_df, ff], axis=1)
    print(ec, end='\r')
poincarre_df.to_csv("features/poincarre_features.csv")

# %%
import pywt
import pandas as pd

def extract_wavelet_features(signal, signal_name, wavelet):
    features = {}
    if type(signal) != np.ndarray:
        return features
    if signal.shape[0] == 0:
        return features

    # Compute wavelet coefficients
    coeffs = pywt.wavedec(signal, wavelet, level=3, mode='periodic')

    # Extract features from each level of coefficients
    for i, coeff in enumerate(coeffs):
        features.update(extract_time_features(coeff, f"{signal_name}_{wavelet}_coeff_{i}"))

    return features

# Assuming 'signals' is a list of your signals and 'signal_names' is a list of corresponding names
wavelets = ['db4']
wavelets_df = pd.DataFrame(index=ecg_df.index)


for signal_name in all_data.columns.to_list():
    for wavelet in wavelets:
        features = all_data[signal_name].map(lambda s : extract_wavelet_features(s, signal_name, wavelet)).apply(pd.Series)
        wavelets_df = pd.concat([wavelets_df, features], axis=1)
        print(signal_name, wavelet, end='\r')
        
# Write the DataFrame to a CSV file
wavelets_df.to_csv('features/wavelet_features.csv')


# %%
def extract_morphological_features(data, signal_name):
    features = {}
    if(type(data) != np.ndarray):
        return features
    if (data.shape[0] == 0):
        return features

    # Compute the R-peak amplitude
    r_amplitude = np.max(data)
    features[signal_name + '_r_amplitude'] = r_amplitude
    r_index = np.argmax(data)
    features[signal_name + '_r_index'] = r_index

    # Compute the Q and S wave amplitudes
    if np.argmax(data) != 0:
        q_amplitude = np.min(data[:np.argmax(data)])
        features[signal_name + '_q_amplitude'] = q_amplitude
        q_index = np.argmin(data[:np.argmax(data)])
        features[signal_name + '_q_index'] = q_index

    if np.argmax(data) < data.shape[0]:
        s_amplitude = np.min(data[np.argmax(data):])
        features[signal_name + '_s_amplitude'] = s_amplitude
        s_index = np.argmin(data[np.argmax(data):])
        features[signal_name + '_s_index'] = s_index

    # Compute the T wave amplitude
    
    if np.argmax(data) < data.shape[0]:
        t_amplitude = np.max(data[np.argmax(data):])
        features[signal_name + '_t_amplitude'] = t_amplitude
        t_index = np.argmax(data[np.argmax(data):])
        features[signal_name + '_t_index'] = t_index

    # Compute the P wave amplitude
    if np.argmin(data) > 0:
        p_amplitude = np.max(data[:np.argmin(data)])
        features[signal_name + '_p_amplitude'] = p_amplitude
        p_index = np.argmax(data[:np.argmin(data)])
        features[signal_name + '_p_index'] = p_index

    # Compute the QRS complex duration
    if np.argmax(data) > 0:
        qrs_duration = s_index - q_index
        features[signal_name + '_qrs_duration'] = qrs_duration

    # Compute the QT interval
    if np.argmax(data) < data.shape[0] and np.argmax(data) > 0:
        qt_interval = np.argmax(data[np.argmax(data):]) - np.argmin(data[:np.argmax(data)])
        features[signal_name + '_qt_interval'] = qt_interval

    # Compute the PR interval
    if(np.argmin(data) > 0):
        pr_interval = r_index - p_index
        features[signal_name + '_pr_interval'] = pr_interval

    # Compute the ST segment
    if np.argmax(data) < data.shape[0] and np.argmax(data) > 0:
        st_segment = t_index - s_index
        features[signal_name + '_st_segment'] = st_segment

    # Compute the P wave duration
    if np.argmin(data) > 0:
        p_duration = np.argmax(data) - np.argmax(data[:np.argmin(data)])
        features[signal_name + '_p_duration'] = p_duration

    # Compute the T wave duration
    if np.argmax(data) < data.shape[0] and np.argmax(data) > 0:
        t_duration = np.argmax(data[np.argmax(data):]) - np.argmax(data)
        features[signal_name + '_t_duration'] = t_duration

    # Return the extracted features
    return features


morphological_df = pd.DataFrame(index=ecg_df.index)

for ec in all_data.columns.to_list():
    mf = all_data[ec].apply(lambda x : extract_morphological_features(x, ec)).apply(pd.Series)
    morphological_df = pd.concat([morphological_df, mf], axis=1)
    print(ec, end='\r')
morphological_df.to_csv("features/morphological_features.csv")

# %%
import pandas as pd

time_df = pd.read_csv("features/time_features.csv", index_col="id")
frequency_df = pd.read_csv("features/frequency_features.csv", index_col="id")
poincarre_df = pd.read_csv("features/poincarre_features.csv", index_col="id")
wavelets_df = pd.read_csv("features/wavelet_features.csv", index_col="id")
morphological_df = pd.read_csv("features/morphological_features.csv", index_col="id")

# %%

X_features = pd.concat([time_df, frequency_df, poincarre_df, morphological_df, wavelets_df], axis=1)
print(X_features.shape)
X_features = X_features.replace([np.inf, -np.inf], np.nan)

X_features = X_features.loc[:, ((X_features != 0) & (X_features.notna())).any(axis=0)]
print(X_features.shape)

# Save the normalized dataframe to a CSV file
X_features.to_csv(f"features/{processing}_features.csv")

print(X_features.shape)

# %%
X_train_features = pd.read_csv("features/X_train_features.csv", index_col="id")
X_test_features = pd.read_csv("features/X_test_features.csv", index_col="id")
# _, y_train, _ = load_Xtrain_ytrain_Xtest()

# %%
X_train_features.shape, X_test_features.shape

# %%
# check which columns are missing in the test set
missing_cols = set(X_train_features.columns) - set(X_test_features.columns)
missing_cols

# %%
# check whether there are any missing values
y_train.value_counts()

# y
# 0    3030
# 2    1474
# 1     443
# 3     170
# Name: count, dtype: int64

# so imbalanced, we need to balance the training data to 3:2:1:1
# but there are NaN values in the features, we need to fill them first
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')
X_train_features = imputer.fit_transform(X_train_features)
X_test_features = imputer.fit_transform(X_test_features)



from imblearn.over_sampling import SMOTE

# set the ratio to 3:2:1:1
smote = SMOTE(sampling_strategy={0: 3030, 1: 3030, 2: 3030, 3: 3030})
X_upsampled, y_upsampled = smote.fit_resample(X_train_features, y_train)

# check the distribution of the target variable
y_upsampled.value_counts()

# train a lightgbm model
import lightgbm as lgb
from sklearn.model_selection import train_test_split
x1, x2, y1, y2 = train_test_split(X_upsampled, y_upsampled, test_size=0.2, random_state=42)

lgbm = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.05, num_leaves=32, colsample_bytree=0.2, reg_alpha=3, reg_lambda=1, min_split_gain=0.01, min_child_weight=40, silent=-1, verbose=-1)
from sklearn.metrics import f1_score

lgbm.fit(x1, y1, eval_set=[(x1, y1), (x2, y2)], eval_metric='multi_logloss')
y_pred = lgbm.predict(x2)
f1_score(y2, y_pred, average='weighted')

# predict the test data


# %%


# %%
y_pred = lgbm.predict(X_test_features)

# %%



