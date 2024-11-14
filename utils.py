import pandas as pd
import numpy as np
from typing import List, Tuple, Dict

import os
import sys
from tqdm import tqdm
from itertools import product
from sklearn.model_selection import cross_val_score
from typing import Any
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

currentdir = os.path.dirname(os.path.realpath(__file__))
datadir = os.path.join(currentdir, "Data")


def print_boundary(center_information="", fill_char="="):
    raw_boundary = fill_char * 100
    # put the center information in the middle
    boundary = (
        raw_boundary[: len(raw_boundary) // 2 - len(center_information) // 2]
        + center_information
        + raw_boundary[len(raw_boundary) // 2 + len(center_information) // 2 :]
    )
    print(boundary[:100])


def time_wrapper(func):
    """
    Wrapper function to measure the time taken by a function.

    Args:
    func: function, function to be wrapped.

    Returns:
    Wrapped function.
    """

    def wrapper(*args, **kwargs):
        import time

        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Time taken by {func.__name__}: {end-start:.2f} seconds")
        return result

    return wrapper


# ================== Data Loading ==================


@time_wrapper
def load_rawdata() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load raw data.

    Returns:
    Tuple of three DataFrames: train, test, and sample_submission.
    """

    traindata = pd.read_csv(datadir + "/train.csv", index_col="id")
    testdata = pd.read_csv(datadir + "/test.csv", index_col="id")

    return traindata, testdata


def load_Xtrain_ytrain_Xtest() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Load X_train, y_train, and X_test.

    Returns:
    Tuple of three DataFrames: X_train, y_train, and X_test.
    """
    train, test = load_rawdata()
    X_train = train.drop("y", axis=1)
    y_train = train["y"]
    X_test = test

    return X_train, y_train, X_test


# ================== Feature Preprocessing ==================


def print_strategy(strategy: Dict[str, Any]) -> None:
    """print_strategy

    Args:
        strategy (Dict[str, Any]): The strategy to print
    """
    print_boundary("Strategy", fill_char="*")
    for key, value in strategy.items():
        print(key)
        for k, v in value.items():
            print(f"\t{k}: {v}")
    print_boundary("End of Strategy", fill_char="*")
    return


def elicitStrategy() -> Dict[str, Any]:
    """elicitStrategy

    Returns:
        Dict[str, Any]: The elicited strategy
    """
    print_boundary("Elicit Strategy")
    strategy = {
        "impute": {
            "processor": SimpleImputer(strategy="mean"),
            "use_all": False,
        },
        "scale": {
            "processor": StandardScaler(),
            "use_all": False,
        },
        "imbalance": {
            "skip": True,
            "method": "over",
            "target_ratio_dict": None,
        },
    }

    print("The sample strategy is:")
    print_strategy(strategy)

    conti = input("Do you want to set the strategy? Enter 'yes' or 'no': ")
    if conti.lower() == "no":
        return strategy

    print("Do you want to impute the data?")
    impute = input("Enter 'yes' or 'no': ")
    if impute.lower() == "yes":
        strategy["impute"]["processor"] = SimpleImputer(strategy="mean")
        choices = [
            SimpleImputer(strategy="mean"),
            SimpleImputer(strategy="median"),
            SimpleImputer(strategy="most_frequent"),
            KNNImputer(),
        ]
        print("Choose the imputer:")
        for i, choice in enumerate(choices):
            print(f"{i+1}. {choice}")
        choice = int(input("Enter the number: "))
        strategy["impute"]["processor"] = choices[choice - 1]

        use_all = input(
            "Do you want to use all data for fitting the imputer? Enter 'yes' or 'no': "
        )
        strategy["impute"]["use_all"] = True if use_all.lower() == "yes" else False

    print("Do you want to scale the data?")
    scale = input("Enter 'yes' or 'no': ")
    if scale.lower() == "yes":
        strategy["scale"]["processor"] = StandardScaler()
        choices = [StandardScaler(), MinMaxScaler()]
        print("Choose the scaler:")
        for i, choice in enumerate(choices):
            print(f"{i+1}. {choice}")
        choice = int(input("Enter the number: "))
        strategy["scale"]["processor"] = choices[choice - 1]

        use_all = input(
            "Do you want to use all data for fitting the scaler? Enter 'yes' or 'no': "
        )
        strategy["scale"]["use_all"] = True if use_all.lower() == "yes" else False

    print("Do you want to handle imbalance?")
    imbalance = input("Enter 'yes' or 'no': ")
    if imbalance.lower() == "yes":
        strategy["imbalance"]["skip"] = False
        method = input("Enter the method to handle imbalance (over/under): ")
        strategy["imbalance"]["method"] = method

        target_ratio = input(
            "Do you want to specify the target ratio? Enter 'yes' or 'no': "
        )
        if target_ratio.lower() == "yes":
            target_ratio_dict = {}
            for i in range(4):
                target_ratio_dict[i] = float(
                    input(f"Enter the target ratio for class {i}: ")
                )
            strategy["imbalance"]["target_ratio_dict"] = target_ratio_dict

    print("The final strategy is:")
    print_strategy(strategy)

    print_boundary("Elicit Strategy Completed")
    return strategy


# ================== Model Tuning ==================


def get_best_parameters(X_train, y_train, estimator, parameters, verbose=0):

    best_parameters = {}
    best_score = 0
    best_estimator = None

    i = 0

    if verbose == 1:
        print("Searching the best parameters for the estimator:", estimator)

    for p in tqdm(
        list(product(*parameters.values())), desc="Searching the best parameters"
    ):
        estimator.set_params(**dict(zip(parameters.keys(), p)))
        # use f1 score for classification
        from sklearn.metrics import make_scorer
        from sklearn.metrics import f1_score

        score = cross_val_score(
            estimator,
            X_train,
            y_train,
            cv=5,
            scoring=make_scorer(f1_score, average="weighted"),
        )

        if verbose == 2:
            print(
                "No:",
                i,
                "params:",
                p,
                "score:",
                np.round(score.mean(), 4),
                "best:",
                np.round(best_score, 4),
            )

        if score.mean() > best_score:
            best_score = score.mean()
            best_parameters = dict(zip(parameters.keys(), p))
            best_estimator = estimator
        i += 1

    print("Best params:", best_parameters)
    print("score:", best_score)
    print("best:", best_estimator)

    return best_estimator


# ================== Model Evaluation ==================


def submit_pred(y_pred: np.ndarray, filename: str = "submission.csv") -> None:
    """
    Save predictions to a CSV file.

    Args:
    y_pred: np.ndarray, predictions.
    filename: str, name of the file to save.

    Returns:
    None
    """
    submission = pd.read_csv(os.path.join(datadir, "Input", "sample_submission.csv"))
    submission["y"] = y_pred
    submission.to_csv(os.path.join(datadir, "Output", filename), index=False)

    return None
