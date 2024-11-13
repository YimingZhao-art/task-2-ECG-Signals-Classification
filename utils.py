import pandas as pd
import numpy as np
from typing import List, Tuple, Dict

import os
import sys

currentdir = os.path.dirname(os.path.realpath(__file__))
datadir = currentdir + "/Data/Input"

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

def load_rawdata() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load raw data.

    Returns:
    Tuple of three DataFrames: train, test, and sample_submission.
    """
    
    traindata = pd.read_csv(datadir + "/train.csv", index_col="id")
    testdata = pd.read_csv(datadir + "/test.csv", index_col="id")

    return traindata, testdata

@time_wrapper
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
    submission = pd.read_csv("Data/Input/sample_submission.csv")
    submission["y"] = y_pred
    submission.to_csv(f"Data/Output/{filename}", index=False)

    return None
