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
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score

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


def function_runtime_tracker(func):
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
        print_boundary(
            f"Time taken by {func.__name__}: {end-start:.2f} seconds", fill_char="%"
        )
        return result

    return wrapper


def check_inf(X: pd.DataFrame) -> List[str]:
    """check the columns with inf values

    Args:
        X (pd.DataFrame): The input DataFrame

    Returns:
        List[str]: The columns with inf values
    """
    inf_col = set()
    for col in X.columns:
        if np.isinf(X[col]).sum() > 0:
            # print(col, np.isinf(X[col]).sum())
            inf_col.add(col)
    return inf_col


# ================== Data Loading ==================


@function_runtime_tracker
def load_rawdata() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load raw data.

    Returns:
    Tuple of three DataFrames: train, test, and sample_submission.
    """

    traindata = pd.read_csv(datadir + "/Input/train.csv", index_col="id")
    testdata = pd.read_csv(datadir + "/Input/test.csv", index_col="id")

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


@function_runtime_tracker
def load_final_data(
    warm_start: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Load final data.

    Args:
    warm_start: bool, whether to use the existing final data.

    Returns:
    Tuple of three DataFrames: X_train, X_test, and y_train.
    """
    print_boundary("Loading Final Data", fill_char="=")

    # check whether the final data is already generated
    if (
        os.path.exists(datadir + "/final/X_train_total.csv")
        and os.path.exists(datadir + "/final/X_test_total.csv")
        and os.path.exists(datadir + "/final/y_train_total.csv")
    ) and warm_start:
        X_train_total = pd.read_csv(datadir + "/final/X_train_total.csv", index_col=0)
        X_test_total = pd.read_csv(datadir + "/final/X_test_total.csv", index_col=0)
        y_train_total = pd.read_csv(datadir + "/final/y_train_total.csv", index_col=0)
        print_boundary("Final Data Shape", fill_char="-")
        print(X_train_total.shape, X_test_total.shape, y_train_total.shape)
        print_boundary()
        return X_train_total, X_test_total, y_train_total.values.ravel()

    X_train_p1 = pd.read_csv(datadir + "/final/p1_X_train.csv", index_col=0)
    X_test_p1 = pd.read_csv(datadir + "/final/p1_X_test.csv", index_col=0)
    y_train_p1 = pd.read_csv(datadir + "/final/p1_y_train.csv", index_col=0)
    X_train_p2 = pd.read_csv(datadir + "/final/p2_X_train.csv", index_col=0)
    X_test_p2 = pd.read_csv(datadir + "/final/p2_X_test.csv", index_col=0)
    y_train_p2 = pd.read_csv(datadir + "/final/p2_y_train.csv", index_col=0)
    print(X_train_p1.shape, X_test_p1.shape, y_train_p1.shape)
    print(X_train_p2.shape, X_test_p2.shape, y_train_p2.shape)
    assert (
        X_train_p1.shape[0] == y_train_p1.shape[0]
    ), "X_train_p1 and y_train_p1 not equal"
    assert (
        X_train_p2.shape[0] == y_train_p2.shape[0]
    ), "X_train_p2 and y_train_p2 not equal"
    assert (
        y_train_p2.values.ravel() == y_train_p2.values.ravel()
    ).all() == True, "y_train_p2 not equal to y_train_p2"

    X_train_total = pd.concat(
        [X_train_p1, X_train_p2], axis=1
    )  # concat along columns, column-wise
    X_test_total = pd.concat([X_test_p1, X_test_p2], axis=1)
    y_train_total = y_train_p1

    inf_col = check_inf(X_train_total) | check_inf(X_test_total)
    inf_col = list(inf_col)
    print("Columns with inf values: ", inf_col)
    # drop the columns with inf values
    X_train_total = X_train_total.drop(inf_col, axis=1)
    X_test_total = X_test_total.drop(inf_col, axis=1)
    X_train_total.to_csv(datadir + "/final/X_train_total.csv")
    X_test_total.to_csv(datadir + "/final/X_test_total.csv")
    y_train_total.to_csv(datadir + "/final/y_train_total.csv")

    print_boundary("Final Data Shape", fill_char="-")
    print(X_train_total.shape, X_test_total.shape, y_train_total.shape)
    print_boundary()

    return X_train_total, X_test_total, y_train_total.values.ravel()


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
@function_runtime_tracker
def evaluate_model(X_train, y_train, model, cv=True):
    """
    Evaluate a model using cross-validation or train-test split.

    Args:
    X_train: pd.DataFrame, training data.
    y_train: pd.Series, training target.
    model: Any, model to evaluate.
    cv: bool, whether to use cross-validation.

    Returns:
    float, mean F1 score.
    """
    if cv:
        score = cross_val_score(
            model,
            X_train,
            y_train,
            cv=5,
            scoring=make_scorer(f1_score, average="micro"),
            n_jobs=-1,
            verbose=1,
        )
    else:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        score = f1_score(y_val, y_pred, average="micro")

    return score.mean()


@function_runtime_tracker
def get_best_parameters(X_train, y_train, estimator, parameters, verbose=0, cv=True):

    best_parameters = {}
    best_score = 0
    best_estimator = None
    scores = []
    i = 0

    if verbose == 1:
        print("Searching the best parameters for the estimator:", estimator)

    for p in tqdm(
        list(product(*parameters.values())), desc="Searching the best parameters"
    ):
        estimator.set_params(**dict(zip(parameters.keys(), p)))

        score = evaluate_model(X_train, y_train, estimator, cv=cv)

        if verbose == 1:
            # print every 20 iterations
            if i % 20 == 0:
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

        elif verbose == 2:
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


def calculate_weights(scores, base=3.0, delta=0.02):
    min_score = min(scores)
    weights = np.array([base ** ((s - min_score) / delta) for s in scores])
    weights /= np.sum(weights)
    return weights
