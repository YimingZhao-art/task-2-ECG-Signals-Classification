import numpy as np
import pandas as pd
import os
import sys
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline

current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_dir, "../"))
from utils import *

current_dir = os.path.dirname(os.path.realpath(__file__))
read_dir = os.path.join(current_dir, "../Data/features/")
final_dir = os.path.join(current_dir, "../Data/final/")
os.makedirs(read_dir, exist_ok=True)


def main():

    X_train_hc_unprocessed = pd.read_csv(
        read_dir + "manual2_train_features.csv", index_col=0
    ).to_numpy()
    X_test_hc_unprocessed = pd.read_csv(
        read_dir + "manual2_test_features.csv", index_col=0
    ).to_numpy()
    y_tot = (
        pd.read_csv(read_dir + "manual2_train_y.csv", index_col=0).to_numpy().ravel()
    )

    X_train_hc_no_nan = np.nan_to_num(X_train_hc_unprocessed, nan=0)
    X_test_hc_no_nan = np.nan_to_num(X_test_hc_unprocessed, nan=0)
    pipe = Pipeline(
        [
            ("variance", VarianceThreshold()),
        ]
    )

    X_tot_hc = pipe.fit_transform(X_train_hc_no_nan, y_tot)
    X_test_hc = pipe.transform(X_test_hc_no_nan)

    columns_names = []

    def get_prediction_estimator(fold):

        X_tot_ml1 = np.loadtxt(
            os.path.join(read_dir, "resnet_training_features" + str(fold) + ".txt"),
            delimiter=",",
        )

        X_test_ml1 = np.loadtxt(
            os.path.join(read_dir, "resnet_test_features" + str(fold) + ".txt"),
            delimiter=",",
        )

        X_tot_ml2 = np.loadtxt(
            os.path.join(read_dir, "fella_ml_training_features" + str(fold) + ".txt"),
            delimiter=",",
        )

        X_test_ml2 = np.loadtxt(
            os.path.join(read_dir, "fella_ml_test_features" + str(fold) + ".txt"),
            delimiter=",",
        )

        X_all = np.concatenate((X_tot_hc, X_tot_ml1, X_tot_ml2), axis=1)
        X_test_combined = np.concatenate((X_test_hc, X_test_ml1, X_test_ml2), axis=1)
        y_all = y_tot

        # set columns names
        # for hc, use hc_[i]
        # for ml1, use ml1_[i]
        # for ml2, use ml2_[i]
        nonlocal columns_names
        if len(columns_names) == 0:
            for i in range(X_tot_hc.shape[1]):
                columns_names.append("hc_" + str(i))
            for i in range(X_tot_ml1.shape[1]):
                columns_names.append("ml1_" + str(i))
            for i in range(X_tot_ml2.shape[1]):
                columns_names.append("ml2_" + str(i))
            columns_names = np.array(columns_names)

        # save X_all, X_test_combined, y_all
        pd.DataFrame(X_all).to_csv(
            os.path.join(read_dir, f"X_all_{fold}.csv"), header=columns_names
        )
        pd.DataFrame(X_test_combined).to_csv(
            os.path.join(read_dir, f"X_test_combined_{fold}.csv"), header=columns_names
        )
        pd.DataFrame(y_all).to_csv(os.path.join(read_dir, f"y_all_{fold}.csv"))

    for i in range(5):
        get_prediction_estimator(fold=i)

    for i in range(1):
        X_all = pd.read_csv(
            os.path.join(read_dir, f"X_all_{i}.csv"), index_col=0
        ).to_numpy()
        X_test_combined = pd.read_csv(
            os.path.join(read_dir, f"X_test_combined_{i}.csv"), index_col=0
        ).to_numpy()
        y_all = (
            pd.read_csv(os.path.join(read_dir, f"y_all_{i}.csv"), index_col=0)
            .to_numpy()
            .ravel()
        )

    # merge by averaging 5 folds
    X_all = np.zeros(X_all.shape)
    X_test_combined = np.zeros(X_test_combined.shape)
    y_all = np.zeros(y_all.shape)

    for i in range(5):
        X_all += pd.read_csv(
            os.path.join(read_dir, f"X_all_{i}.csv"), index_col=0
        ).to_numpy()
        X_test_combined += pd.read_csv(
            os.path.join(read_dir, f"X_test_combined_{i}.csv"), index_col=0
        ).to_numpy()
        y_all += (
            pd.read_csv(os.path.join(read_dir, f"y_all_{i}.csv"), index_col=0)
            .to_numpy()
            .ravel()
        )

    X_all /= 5
    X_test_combined /= 5
    y_all /= 5
    # set y_all back to int
    y_all = y_all.astype(int)

    pd.DataFrame(X_all).to_csv(
        os.path.join(final_dir, "p1_X_train.csv"), header=columns_names
    )
    pd.DataFrame(X_test_combined).to_csv(
        os.path.join(final_dir, "p1_X_test.csv"), header=columns_names
    )
    pd.DataFrame(y_all).to_csv(os.path.join(final_dir, "p1_y_train.csv"))

    print_boundary("Finished merging the features", fill_char="*")
    print("The shape of the final X_train:", X_all.shape)
    print("The shape of the final X_test:", X_test_combined.shape)
    print("The shape of the final y_train:", y_all.shape)
    print_boundary(fill_char="*")


if __name__ == "__main__":
    main()
