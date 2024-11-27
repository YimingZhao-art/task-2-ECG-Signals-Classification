import os
import sys

import biosppy.signals.ecg as ecg
import numpy as np
from numpy.fft import fft
from scipy.signal import welch
from scipy.stats import skew, kurtosis
from scipy.stats import gmean
from scipy.stats import entropy
import pywt
import pandas as pd
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")
from typing import List, Tuple, Union, Dict

ecg_features = [
    "ts",
    "filtered",
    "rpeaks",
    "templates_ts",
    "mean_templates",
    "heart_rate_ts",
    "heart_rate",
]

current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_dir, "../"))

save_dir = os.path.join(current_dir, "../Data/features/")
os.makedirs(save_dir, exist_ok=True)


# ECG features are features that describe the ECG signal
def get_ecg_df(X: pd.DataFrame, save: bool = False) -> pd.DataFrame:
    """Get the ECG features from the dataset.

    Args:
        X (pd.DataFrame): The time-series dataset.
        save (bool, optional): Whether to save the ECG features to a CSV file. Defaults to False.

    Returns:
        pd.DataFrame: A DataFrame containing the ECG features.
    """

    def extract_ecg_features(X: pd.DataFrame, i: int) -> tuple:
        """Extract ECG features from a single row of the dataset.

        Args:
            X (pd.DataFrame): The time-series dataset.
            i (int): The index of the row to extract features from.

        Returns:
            tuple: A tuple containing the extracted features:
                ts (np.array): The timestamp of the ECG signal.
                filtered (np.array): The filtered ECG signal.
                rpeaks (np.array): The R-peaks of the ECG signal.
                templates_ts (np.array): The timestamp of the templates.
                templates (np.array): The templates of the ECG signal.
                heart_rate_ts (np.array): The timestamp of the heart rate.
                heart_rate (np.array): The heart rate.
        """
        row = X.loc[i].dropna().to_numpy(dtype="float32")
        (
            ts,
            filtered,
            rpeaks,
            templates_ts,
            templates,
            heart_rate_ts,
            heart_rate,
        ) = ecg.ecg(signal=row, sampling_rate=300, show=False)

        return ts, filtered, rpeaks, templates_ts, templates, heart_rate_ts, heart_rate

    ecg_df = pd.DataFrame(columns=ecg_features)
    ecg_df.index.name = "id"
    for i in tqdm(range(X.shape[0]), desc="Extracting ECG features"):
        (
            ts,
            filtered,
            rpeaks,
            templates_ts,
            templates,
            heart_rate_ts,
            heart_rate,
        ) = extract_ecg_features(X, i)
        mean_templates = np.mean(templates, axis=0)
        ecg_df.loc[len(ecg_df)] = [
            ts,
            filtered,
            rpeaks,
            templates_ts,
            mean_templates,
            heart_rate_ts,
            heart_rate,
        ]

    if save:
        ecg_df.to_csv(os.path.join(save_dir, "ecg_features.csv"))

    return ecg_df


# Fast Fourier Transform (FFT) features are features that describe the signal in the frequency domain
def get_fft_df(ecg_df: pd.DataFrame, save: bool = False) -> pd.DataFrame:
    """Extract the Fast Fourier Transform (FFT) of the ECG features.

    Args:
        ecg_df (pd.DataFrame): The ECG features DataFrame.
        save (bool, optional): Whether to save the FFT features to a CSV file. Defaults to False.

    Returns:
        pd.DataFrame: A DataFrame containing the FFT of the ECG features.
    """

    def extract_fft(ecg_feature: np.array, fs: int = 300) -> np.array:
        """Extract the Fast Fourier Transform (FFT) of the ECG feature.

        Args:
            ecg_feature (np.array): The ECG feature to extract the FFT from.
            fs (int, optional): The sampling frequency of the ECG signal. Defaults to 300.

        Returns:
            np.array: The FFT of the ECG feature.
        """
        if ecg_feature.shape[0] == 0:
            return np.array([])
        fft_values = fft(ecg_feature)
        fft_values = 2.0 * np.abs(fft_values[: fs // 2]) / len(ecg_feature)

        return fft_values

    fft_features = ["fft_" + e for e in ecg_features]
    fft_df = pd.DataFrame(columns=fft_features, index=ecg_df.index)
    for ec in tqdm(ecg_features, desc="Extracting FFT features"):
        fft_df["fft_" + ec] = ecg_df[ec].map(extract_fft)

    if save:
        fft_df.to_csv(os.path.join(save_dir, "fft_features.csv"))

    return fft_df


# Power spectral density (PSD) features are features that describe the power of the signal at different frequencies
def get_psd_df(ecg_df: pd.DataFrame, save: bool = False) -> pd.DataFrame:
    def extract_psd(ecg_feature, fs=300):
        if ecg_feature.shape[0] == 0:
            return np.array([])
        freqs, psd_values = welch(ecg_feature, fs)
        return psd_values

    psd_features = ["psd_" + e for e in ecg_features]
    psd_df = pd.DataFrame(columns=psd_features, index=ecg_df.index)
    for ec in tqdm(ecg_features, desc="Extracting PSD features"):
        psd_df["psd_" + ec] = ecg_df[ec].map(extract_psd)

    if save:
        psd_df.to_csv(os.path.join(save_dir, "psd_features.csv"))

    return psd_df


def extract_time_features(
    data: np.ndarray, signal_name: str
) -> Dict[str, Union[int, float]]:
    """Extract time-domain features from the ECG signal.

    Args:
        data (np.ndarray): The ECG signal from the dataset.
        signal_name (str): The name of the ECG signal.

    Returns:
        Dict[str, Union[int, float]]: A dictionary containing the extracted features.
    """
    features = {}
    if type(data) != np.ndarray:
        return features
    if data.shape[0] == 0:
        return features

    # Initialize dictionary to hold features

    # Basic statistics
    features[signal_name + "_mean"] = np.mean(data)
    features[signal_name + "_median"] = np.median(data)
    features[signal_name + "_std"] = np.std(data)
    features[signal_name + "_var"] = np.var(data)
    features[signal_name + "_max"] = np.max(data)
    features[signal_name + "_min"] = np.min(data)
    features[signal_name + "_rms"] = np.sqrt(np.mean(np.square(data)))
    features[signal_name + "_peak_to_peak"] = (
        features[signal_name + "_max"] - features[signal_name + "_min"]
    )
    features[signal_name + "_skewness"] = skew(data)
    features[signal_name + "_kurtosis"] = kurtosis(data)

    # Zero Crossing Rate
    features[signal_name + "_zero_crossing_rate"] = ((data[:-1] * data[1:]) < 0).sum()

    # Signal Magnitude Area
    features[signal_name + "_sma"] = np.sum(np.abs(data))

    # Energy
    features[signal_name + "_energy"] = np.sum(np.square(data))

    # Entropy
    p_signal = data / np.sum(data)  # normalize signal
    features[signal_name + "_entropy"] = -np.sum(p_signal * np.log2(p_signal))

    # Crest Factor
    features[signal_name + "_crest_factor"] = (
        features[signal_name + "_max"] / features[signal_name + "_rms"]
    )

    # Impulse Factor
    features[signal_name + "_impulse_factor"] = (
        features[signal_name + "_max"] / features[signal_name + "_mean"]
    )

    # Shape Factor
    features[signal_name + "_shape_factor"] = features[signal_name + "_rms"] / (
        np.sum(np.abs(data)) / len(data)
    )

    # Clearance Factor
    features[signal_name + "_clearance_factor"] = features[
        signal_name + "_max"
    ] / np.sqrt(np.mean(np.square(np.abs(data))))

    # Return the extracted features
    return features


# Time-domain features are features that describe the signal in the time domain
def get_time_df(
    ecg_df: pd.DataFrame, fft_df: pd.DataFrame, psd_df: pd.DataFrame, save: bool = False
) -> pd.DataFrame:
    """Extract time-domain features from the ECG, FFT, and PSD features.

    Args:
        ecg_df (pd.DataFrame): The ECG features DataFrame.
        fft_df (pd.DataFrame): The FFT features DataFrame.
        psd_df (pd.DataFrame): The PSD features DataFrame.
        save (bool, optional): Whether to save the time-domain features to a CSV file. Defaults to False.

    Returns:
        pd.DataFrame: A DataFrame containing the time-domain features.
    """

    combine_df = pd.concat([ecg_df, fft_df, psd_df], axis=1)
    time_df = pd.DataFrame(index=ecg_df.index)
    for ec in tqdm(combine_df.columns.to_list(), desc="Extracting time features"):
        tf_features = (
            combine_df[ec]
            .apply(lambda x: extract_time_features(x, ec))  # Extract features
            .apply(pd.Series)  # Convert the dictionary to a DataFrame
        )
        time_df = pd.concat([time_df, tf_features], axis=1)

    if save:
        time_df.to_csv(os.path.join(save_dir, "time_features.csv"))

    return time_df


# Frequency-domain features are features that describe the signal in the frequency domain
def get_frequency_df(
    ecg_df: pd.DataFrame, fft_df: pd.DataFrame, psd_df: pd.DataFrame, save: bool = False
) -> pd.DataFrame:
    """Extract frequency-domain features from the ECG, FFT, and PSD features.

    Args:
        ecg_df (pd.DataFrame): The ECG features DataFrame.
        fft_df (pd.DataFrame): The FFT features DataFrame.
        psd_df (pd.DataFrame): The PSD features DataFrame.
        save (bool, optional): Whether to save the frequency-domain features to a CSV file. Defaults to False.

    Returns:
        pd.DataFrame: A DataFrame containing the frequency-domain features.
    """

    def extract_frequency_features(
        data: np.ndarray, signal_name: str, fs: int = 300
    ) -> Dict[str, Union[int, float]]:
        """Extract frequency-domain features from the ECG signal.

        Args:
            data (np.ndarray): The ECG signal from the dataset.
            signal_name (str): The name of the ECG signal.
            fs (int, optional): The sampling frequency of the ECG signal. Defaults to 300.

        Returns:
            Dict[str, Union[int, float]]: A dictionary containing the extracted features.
        """
        features = {}
        if type(data) != np.ndarray:
            return features
        if data.shape[0] == 0:
            return features

        # Compute the power spectral density of the signal
        freqs, psd_values = welch(data, fs=fs)

        # Compute the peak frequency
        features[signal_name + "_peak_freq"] = freqs[np.argmax(psd_values)]

        # Compute the median frequency
        features[signal_name + "_median_freq"] = freqs[len(freqs) // 2]

        # Compute the spectral entropy
        features[signal_name + "_spectral_entropy"] = entropy(psd_values)

        # Compute the spectral energy
        spectral_energy = np.sum(psd_values)
        features[signal_name + "_spectral_energy"] = spectral_energy

        # Compute the spectral centroid
        spectral_centroid = np.sum(freqs * psd_values) / np.sum(psd_values)
        features[signal_name + "_spectral_centroid"] = spectral_centroid

        # Compute the spectral spread
        spectral_spread = np.sqrt(
            np.sum((freqs - spectral_centroid) ** 2 * psd_values) / np.sum(psd_values)
        )
        features[signal_name + "_spectral_spread"] = spectral_spread

        # Compute the spectral flatness
        features[signal_name + "_spectral_flatness"] = gmean(psd_values) / np.mean(
            psd_values
        )

        # Return the extracted features
        return features

    combine_df = pd.concat([ecg_df, fft_df, psd_df], axis=1)
    frequency_df = pd.DataFrame(index=ecg_df.index)

    for ec in tqdm(combine_df.columns.to_list(), desc="Extracting frequency features"):
        ff = (
            combine_df[ec]
            .apply(lambda x: extract_frequency_features(x, ec))
            .apply(pd.Series)
        )
        frequency_df = pd.concat([frequency_df, ff], axis=1)

    if save:
        frequency_df.to_csv(os.path.join(save_dir, "frequency_features.csv"))

    return frequency_df


# Poincaré features are features that describe the variability of the heart rate
def get_poincarre_df(
    ecg_df: pd.DataFrame, fft_df: pd.DataFrame, psd_df: pd.DataFrame, save: bool = False
) -> pd.DataFrame:
    """Extract Poincaré features from the ECG, FFT, and PSD features. Pointcaré features are used to analyze the variability of the heart rate.

    Args:
        ecg_df (pd.DataFrame): The ECG features DataFrame.
        fft_df (pd.DataFrame): The FFT features DataFrame.
        psd_df (pd.DataFrame): The PSD features DataFrame.
        save (bool, optional): Whether to save the Poincaré features to a CSV file. Defaults to False.

    Returns:
        pd.DataFrame: A DataFrame containing the Poincaré features.
    """

    def calculate_poincare_descriptors(
        data: np.ndarray, signal_name: str
    ) -> Dict[str, Union[int, float]]:
        """Calculate the Poincaré descriptors of the ECG signal.

        Args:
            data (np.ndarray): The ECG signal from the dataset.
            signal_name (str): The name of the ECG signal.

        Returns:
            Dict[str, Union[int, float]]: A dictionary containing the Poincaré descriptors.
        """
        features = {}

        if type(data) != np.ndarray:
            return features
        if data.shape[0] == 0:
            return features

        # Calculate the differences between successive RR intervals
        rr_diff = np.diff(data)

        # Calculate SD1 and SD2
        features[signal_name + "_SD1"] = np.sqrt(np.std(rr_diff) ** 2 * 0.5)
        features[signal_name + "_SD2"] = np.sqrt(
            2 * np.std(data) ** 2 - 0.5 * np.std(rr_diff) ** 2
        )

        # Calculate SD1/SD2 ratio
        features[signal_name + "_SD1/SD2"] = (
            features[signal_name + "_SD1"] / features[signal_name + "_SD2"]
        )

        # Calculate the area of the Poincaré ellipse
        features[signal_name + "_ellipse_area"] = (
            np.pi * features[signal_name + "_SD1"] * features[signal_name + "_SD2"]
        )

        return features

    poincarre_df = pd.DataFrame(index=ecg_df.index)
    combine_df = pd.concat([ecg_df, fft_df, psd_df], axis=1)

    for ec in tqdm(combine_df.columns.to_list(), desc="Extracting Poincarre features"):
        ff = (
            combine_df[ec]
            .apply(lambda x: calculate_poincare_descriptors(x, ec))
            .apply(pd.Series)
        )
        poincarre_df = pd.concat([poincarre_df, ff], axis=1)

    if save:
        poincarre_df.to_csv(os.path.join(save_dir, "poincarre_features.csv"))

    return poincarre_df


# wavelet features are features that describe the signal at different scales
def get_wavelets_df(
    ecg_df: pd.DataFrame, fft_df: pd.DataFrame, psd_df: pd.DataFrame, save: bool = False
) -> pd.DataFrame:
    """Extract wavelet features from the ECG, FFT, and PSD features.

    Args:
        ecg_df (pd.DataFrame): The ECG features DataFrame.
        fft_df (pd.DataFrame): The FFT features DataFrame.
        psd_df (pd.DataFrame): The PSD features DataFrame.
        save (bool, optional): Whether to save the wavelet features to a CSV file. Defaults to False.

    Returns:
        pd.DataFrame: A DataFrame containing the wavelet features.
    """

    def extract_wavelet_features(
        signal: np.ndarray, signal_name: str, wavelet: str
    ) -> Dict[str, Union[int, float]]:
        """Extract wavelet features from the ECG signal.

        Args:
            signal (np.ndarray): The ECG signal from the dataset.
            signal_name (str): The name of the ECG signal.
            wavelet (str): The wavelet to use for feature extraction.

        Returns:
            Dict[str, Union[int, float]]: A dictionary containing the extracted features.
        """
        features = {}
        if type(signal) != np.ndarray:
            return features
        if signal.shape[0] == 0:
            return features

        # Compute wavelet coefficients
        coeffs = pywt.wavedec(signal, wavelet, level=3, mode="periodic")

        # Extract features from each level of coefficients
        for i, coeff in enumerate(coeffs):
            features.update(
                extract_time_features(coeff, f"{signal_name}_{wavelet}_coeff_{i}")
            )

        return features

    wavelets = ["db4"]  # Daubechies wavelet
    wavelets_df = pd.DataFrame(index=ecg_df.index)
    combine_df = pd.concat([ecg_df, fft_df, psd_df], axis=1)

    for signal_name in tqdm(
        combine_df.columns.to_list(), desc="Extracting wavelet features"
    ):
        for wavelet in wavelets:
            features = (
                combine_df[signal_name]
                .map(lambda s: extract_wavelet_features(s, signal_name, wavelet))
                .apply(pd.Series)
            )
            wavelets_df = pd.concat([wavelets_df, features], axis=1)

    if save:
        wavelets_df.to_csv(os.path.join(save_dir, "wavelet_features.csv"))

    return wavelets_df


# morphological features are features that describe the shape of the ECG signal
def get_morphological_df(
    ecg_df: pd.DataFrame, fft_df: pd.DataFrame, psd_df: pd.DataFrame, save: bool = False
) -> pd.DataFrame:
    """Extract morphological features from the ECG, FFT, and PSD features.

    Args:
        ecg_df (pd.DataFrame): The ECG features DataFrame.
        fft_df (pd.DataFrame): The FFT features DataFrame.
        psd_df (pd.DataFrame): The PSD features DataFrame.
        save (bool, optional): Whether to save the morphological features to a CSV file. Defaults to False.

    Returns:
        pd.DataFrame: A DataFrame containing the morphological features.
    """

    def extract_morphological_features(
        data: np.ndarray, signal_name: str
    ) -> Dict[str, Union[int, float]]:
        """Extract morphological features from the ECG signal.

        Args:
            data (np.ndarray): The ECG signal from the dataset.
            signal_name (str): The name of the ECG signal.

        Returns:
            Dict[str, Union[int, float]]: A dictionary containing the extracted features.
        """

        features = {}
        if type(data) != np.ndarray:
            return features
        if data.shape[0] == 0:
            return features

        # Compute the R-peak amplitude
        r_amplitude = np.max(data)
        features[signal_name + "_r_amplitude"] = r_amplitude
        r_index = np.argmax(data)
        features[signal_name + "_r_index"] = r_index

        # Compute the Q and S wave amplitudes
        if np.argmax(data) != 0:
            q_amplitude = np.min(data[: np.argmax(data)])
            features[signal_name + "_q_amplitude"] = q_amplitude
            q_index = np.argmin(data[: np.argmax(data)])
            features[signal_name + "_q_index"] = q_index

        if np.argmax(data) < data.shape[0]:
            s_amplitude = np.min(data[np.argmax(data) :])
            features[signal_name + "_s_amplitude"] = s_amplitude
            s_index = np.argmin(data[np.argmax(data) :])
            features[signal_name + "_s_index"] = s_index

        # Compute the T wave amplitude

        if np.argmax(data) < data.shape[0]:
            t_amplitude = np.max(data[np.argmax(data) :])
            features[signal_name + "_t_amplitude"] = t_amplitude
            t_index = np.argmax(data[np.argmax(data) :])
            features[signal_name + "_t_index"] = t_index

        # Compute the P wave amplitude
        if np.argmin(data) > 0:
            p_amplitude = np.max(data[: np.argmin(data)])
            features[signal_name + "_p_amplitude"] = p_amplitude
            p_index = np.argmax(data[: np.argmin(data)])
            features[signal_name + "_p_index"] = p_index

        # Compute the QRS complex duration
        if np.argmax(data) > 0:
            qrs_duration = s_index - q_index
            features[signal_name + "_qrs_duration"] = qrs_duration

        # Compute the QT interval
        if np.argmax(data) < data.shape[0] and np.argmax(data) > 0:
            qt_interval = np.argmax(data[np.argmax(data) :]) - np.argmin(
                data[: np.argmax(data)]
            )
            features[signal_name + "_qt_interval"] = qt_interval

        # Compute the PR interval
        if np.argmin(data) > 0:
            pr_interval = r_index - p_index
            features[signal_name + "_pr_interval"] = pr_interval

        # Compute the ST segment
        if np.argmax(data) < data.shape[0] and np.argmax(data) > 0:
            st_segment = t_index - s_index
            features[signal_name + "_st_segment"] = st_segment

        # Compute the P wave duration
        if np.argmin(data) > 0:
            p_duration = np.argmax(data) - np.argmax(data[: np.argmin(data)])
            features[signal_name + "_p_duration"] = p_duration

        # Compute the T wave duration
        if np.argmax(data) < data.shape[0] and np.argmax(data) > 0:
            t_duration = np.argmax(data[np.argmax(data) :]) - np.argmax(data)
            features[signal_name + "_t_duration"] = t_duration

        # Return the extracted features
        return features

    morphological_df = pd.DataFrame(index=ecg_df.index)
    combine_df = pd.concat([ecg_df, fft_df, psd_df], axis=1)

    for ec in tqdm(
        combine_df.columns.to_list(), desc="Extracting morphological features"
    ):
        mf = (
            combine_df[ec]
            .apply(lambda x: extract_morphological_features(x, ec))
            .apply(pd.Series)
        )
        morphological_df = pd.concat([morphological_df, mf], axis=1)

    if save:
        morphological_df.to_csv(os.path.join(save_dir, "morphological_features.csv"))

    return morphological_df
