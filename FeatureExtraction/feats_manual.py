import os
import sys

from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_dir))
from func_feats import *

sys.path.append(os.path.join(current_dir, "../"))
from utils import *

current_dir = os.path.dirname(os.path.realpath(__file__))
save_dir = os.path.join(current_dir, "../Data/features/")
os.makedirs(save_dir, exist_ok=True)


@time_wrapper
def extract_X_features(
    X: pd.DataFrame, save: bool = False, processing="X_train"
) -> pd.DataFrame:

    scaler = MinMaxScaler(feature_range=(-100, 100))

    X = X.apply(
        lambda row: pd.Series(scaler.fit_transform(row.values.reshape(-1, 1)).ravel()),
        axis=1,
    )
    print("Raw X shape:", X.shape)

    if processing == "X_test":
        if not os.path.exists(save_dir + "X_train_features.csv"):
            print("X_train_features.csv not found, it must be generated first")
            exit()

    ecg_df = get_ecg_df(X=X, save=save)
    fft_df = get_fft_df(ecg_df=ecg_df, save=save)
    psd_df = get_psd_df(ecg_df=ecg_df, save=save)

    time_df = get_time_df(ecg_df=ecg_df, fft_df=fft_df, psd_df=psd_df, save=save)
    frequency_df = get_frequency_df(
        ecg_df=ecg_df, fft_df=fft_df, psd_df=psd_df, save=save
    )
    poincarre_df = get_poincarre_df(
        ecg_df=ecg_df, fft_df=fft_df, psd_df=psd_df, save=save
    )
    wavelets_df = get_wavelets_df(
        ecg_df=ecg_df, fft_df=fft_df, psd_df=psd_df, save=save
    )
    morphological_df = get_morphological_df(
        ecg_df=ecg_df, fft_df=fft_df, psd_df=psd_df, save=save
    )

    X_features = pd.concat(
        [time_df, frequency_df, poincarre_df, morphological_df, wavelets_df], axis=1
    )

    print("X_features.shape", X_features.shape)

    X_features = X_features.replace([np.inf, -np.inf], np.nan)

    if processing == "X_test":

        X_train_features = pd.read_csv(save_dir + "X_train_features.csv", index_col=0)
        # keep same columns as X_train
        X_features = X_features[X_train_features.columns]

    else:
        X_features = X_features.loc[
            :, ((X_features != 0) & (X_features.notna())).any(axis=0)
        ]
    print("Final X_features.shape", X_features.shape)

    # Save the normalized dataframe to a CSV file
    X_features.to_csv(f"{save_dir}{processing}_features.csv")


if __name__ == "__main__":
    X_train, y_train, X_test = load_Xtrain_ytrain_Xtest()
    extract_X_features(X=X_train, save=True, processing="X_train")
    extract_X_features(X=X_test, save=True, processing="X_test")
