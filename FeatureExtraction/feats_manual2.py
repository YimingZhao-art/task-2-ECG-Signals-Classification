import os
import neurokit2 as nk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_dir))
from func_extractor import *

sys.path.append(os.path.join(current_dir, "../"))
from utils import *

current_dir = os.path.dirname(os.path.realpath(__file__))
save_dir = os.path.join(current_dir, "../Data/features/")
os.makedirs(save_dir, exist_ok=True)


def main():
    X_train_raw, y_train_raw, X_test_raw = load_Xtrain_ytrain_Xtest()

    # Test Fourier
    signal = X_train_raw.iloc[0].dropna().to_numpy(dtype="float32")
    try:
        clean_signal = nk.ecg_clean(signal, sampling_rate=300, method="neurokit")
    except:
        try:
            clean_signal = nk.ecg_clean(
                signal, sampling_rate=300, method="hamilton2002"
            )
        except:
            try:
                clean_signal = nk.ecg_clean(
                    signal, sampling_rate=300, method="elgendi2010"
                )
            except:
                print(f"Fail")

    ff = extract_fft_heartbeat(heartbeat=clean_signal)
    test = extract_fft_feature(clean_signal=clean_signal)
    n = 8000
    fourier_specture = fft(clean_signal)
    fft_freq = sf.fftfreq(len(fourier_specture)) * 300
    fourier_specture = fourier_specture[fft_freq >= 0]
    fft_freq = fft_freq[fft_freq >= 0]
    fft_mask = fourier_specture > np.quantile(fourier_specture, 0.9)
    fourier_specture = np.multiply(fourier_specture, fft_mask)

    fig = plt.figure(figsize=(12, 6))
    grid = plt.GridSpec(2, 1, hspace=0.6)

    full_signal = fig.add_subplot(grid[0, 0])
    fft_comp = fig.add_subplot(grid[1, 0])

    full_signal.plot(np.arange(len(clean_signal)), clean_signal[0:], color="green")
    full_signal.set_xlim(0, 18000)
    full_signal.set_title("Full Signal")
    fft_comp.bar(fft_freq, list(fourier_specture), 0.1, color="purple")
    fft_comp.set_title("FFT of full signal")

    signal = X_train_raw.iloc[0].dropna().to_numpy(dtype="float32")
    meth = "hand"
    cleaned_signal = nk.ecg_clean(signal, sampling_rate=300, method="neurokit")
    ecg_data = extract_ecg_data(cleaned_signal, meth)
    fft_data = extract_fft_feature(clean_signal)

    features = multi_features(X_train_raw)
    # 改成存到Data/features/manual2_train_features.csv
    features.to_csv(os.path.join(save_dir, "manual2_train_features.csv"))
    features = pd.read_csv("train_features.csv", index_col=0)

    # Create test features
    test_features = multi_features(X_test_raw)
    test_features.to_csv(os.path.join(save_dir, "manual2_test_features.csv"))

    # y也可以存一个
    # Data/features/manual2_train_y.csv
    pd.DataFrame(y_train_raw).to_csv(os.path.join(save_dir, "manual2_train_y.csv"))


if __name__ == "__main__":
    main()
