import os
import sys
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import f1_score
import lightgbm as lgb

from typing import List, Tuple, Dict, Any

current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current_dir)
sys.path.append(current_dir + "/../")
print(current_dir)
data_folder = os.path.join(current_dir, "../Data")
os.makedirs(data_folder, exist_ok=True)  # create the data folder if it doesn't exist
final_data_folder = os.path.join(data_folder, "final")
os.makedirs(final_data_folder, exist_ok=True)

from utils import *

try:
    y_pred = np.zeros((3411))
    # test submission functionality
    submit_pred(y_pred, "test_submission.csv")
except Exception as e:
    print("Unable to submit the file. Error:", e)
    exit(1)


def imputeOrScale(
    processor: object,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    use_all: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """imputeOrScale

    Args:
        processor (object): Imputer or Scaler object
        X_train (pd.DataFrame): The training data
        X_test (pd.DataFrame): The testing data
        use_all (bool, optional): Whether to use all data for fitting. Defaults to False.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: The transformed training and testing data
    """
    if use_all:
        X_all = pd.concat([X_train, X_test], axis=0)
        X_all = pd.DataFrame(processor.fit_transform(X_all), columns=X_all.columns)
        processor.fit(X_all)
        X_train = pd.DataFrame(processor.transform(X_train), columns=X_train.columns)
        X_test = pd.DataFrame(processor.transform(X_test), columns=X_test.columns)

    else:
        X_train = pd.DataFrame(
            processor.fit_transform(X_train), columns=X_train.columns
        )
        X_test = pd.DataFrame(processor.transform(X_test), columns=X_test.columns)

    return X_train, X_test


def handleImbalance(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    skip: bool = False,
    method: str = "over",
    target_ratio_dict: dict = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """handleImbalance

    Args:
        X_train (pd.DataFrame): The training data
        y_train (pd.DataFrame): The training labels
        skip (bool, optional): Whether to skip the imbalance handling. Defaults to False.
        method (str, optional): The method to use for handling imbalance. Defaults to "over".
        target_ratio_dict (dict, optional): The target ratio dictionary. Defaults to None.


    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: The transformed training data and labels
    """
    if skip:
        return X_train, y_train

    if method == "over":
        smote = SMOTE(
            sampling_strategy=target_ratio_dict if target_ratio_dict else "auto",
            random_state=42,
        )
        X_train, y_train = smote.fit_resample(X_train, y_train)

    elif method == "under":
        rus = RandomUnderSampler(
            sampling_strategy=target_ratio_dict if target_ratio_dict else "auto",
            random_state=42,
        )
        X_train, y_train = rus.fit_resample(X_train, y_train)

    return X_train, y_train


@function_runtime_tracker
def processWithStrategy(
    warmstart: bool = True, save: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if warmstart:
        # directly read the processed data in the final folder
        print("Reading the processed data from the final folder", final_data_folder)
        X_train = pd.read_csv(
            os.path.join(final_data_folder, "p2_X_train.csv"), index_col=0
        )
        X_test = pd.read_csv(
            os.path.join(final_data_folder, "p2_X_test.csv"), index_col=0
        )
        y_train = pd.read_csv(
            os.path.join(final_data_folder, "p2_y_train.csv"), index_col=0
        )
        return X_train, y_train, X_test

    strategy = elicitStrategy()
    if not strategy:
        print("Exiting the process.")
        exit(0)

    print_boundary("Data Processing Started")
    # Load the data
    X_train = pd.read_csv(
        os.path.join(data_folder, "features/X_train_features.csv"), index_col=0
    )
    X_test = pd.read_csv(
        os.path.join(data_folder, "features/X_test_features.csv"), index_col=0
    )
    y_train = pd.read_csv(os.path.join(data_folder, "Input/y_train.csv"), index_col=0)

    # Process the data
    X_train, X_test = imputeOrScale(
        strategy["impute"]["processor"],
        X_train,
        X_test,
        strategy["impute"]["use_all"],
    )

    X_train, X_test = imputeOrScale(
        strategy["scale"]["processor"],
        X_train,
        X_test,
        strategy["scale"]["use_all"],
    )

    X_train, y_train = handleImbalance(
        X_train,
        y_train,
        strategy["imbalance"]["skip"],
        strategy["imbalance"]["method"],
        strategy["imbalance"]["target_ratio_dict"],
    )

    if save:
        print("Saving the processed data to the final folder", final_data_folder)
        X_train.to_csv(os.path.join(final_data_folder, "p2_X_train.csv"))
        X_test.to_csv(os.path.join(final_data_folder, "p2_X_test.csv"))
        y_train.to_csv(os.path.join(final_data_folder, "p2_y_train.csv"))

    print("Shape of X_train:", X_train.shape)
    print("Shape of X_test:", X_test.shape)
    print("Shape of y_train:", y_train.shape)
    print_boundary("Data Processing Completed")

    return X_train, y_train, X_test


def main():
    X_train, y_train, X_test = processWithStrategy(save=True, warmstart=False)
    return

    # split while keep the imbalance ratio
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train.values.ravel(),
        test_size=0.2,
        stratify=y_train,
        random_state=42,
    )

    # it is multi-class classification
    # use multi_logloss as objective

    lgb_params = {
        "objective": "multiclass",
        "num_class": 4,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "max_depth": 7,
        "n_estimators": 1000,
        "n_jobs": -1,
        "seed": 42,
        "early_stopping_rounds": 100,
    }

    lgbm = lgb.LGBMClassifier(**lgb_params)

    lgbm.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])

    y_pred = lgbm.predict(X_val)
    print(f"F1 Score: {f1_score(y_val, y_pred, average='micro')}")


if __name__ == "__main__":
    main()
