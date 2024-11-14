import pandas as pd
import numpy as np
from typing import List, Tuple, Dict

import os
import sys
from tqdm import tqdm
from itertools import product
from sklearn.model_selection import cross_val_score

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


def get_best_parameters(X_train, y_train, estimator, parameters, verbose=0):

    # search = GridSearchCV(estimator=estimator, param_grid=parameters, scoring='r2', n_jobs=-1, cv=5, verbose=1)
    # search.fit(X_train, y_train)

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
        score = cross_val_score(estimator, X_train, y_train, cv=5, scoring="r2")

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
