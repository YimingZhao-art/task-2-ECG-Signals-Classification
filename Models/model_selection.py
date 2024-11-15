# try usual models and select the best one
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
    VotingClassifier,
    BaggingClassifier,
    StackingClassifier,
)
from sklearn.linear_model import (
    LogisticRegression,
    RidgeClassifier,
    SGDClassifier,
    PassiveAggressiveClassifier,
)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

current_dir = os.path.dirname(os.path.realpath(__file__))
print(current_dir)
sys.path.append(current_dir)
sys.path.append(current_dir + "/../")
sys.path.append(current_dir + "/../FeaturePreprocessing/")

from utils import *
from FeaturePreprocessing.process import *


current_dir = os.path.dirname(os.path.realpath(__file__))

models = {
    # "GaussianProcess": GaussianProcessClassifier(kernel=RBF(), n_jobs=-1),
    "LGBM": LGBMClassifier(n_jobs=-1, verbose=-1),
    "XGB": XGBClassifier(n_jobs=-1),
    "CatBoost": CatBoostClassifier(verbose=0),
    "GradientBoosting": GradientBoostingClassifier(verbose=0),  # too slow
    "AdaBoost": AdaBoostClassifier(),
    "ExtraTrees": ExtraTreesClassifier(),
    "Bagging": BaggingClassifier(),
    "RandomForest": RandomForestClassifier(),
    "LogisticRegression": LogisticRegression(),
    "Ridge": RidgeClassifier(),
    "SGD": SGDClassifier(),
    "PassiveAggressive": PassiveAggressiveClassifier(),
    "SVC": SVC(),
    "DecisionTree": DecisionTreeClassifier(),
    "KNeighbors": KNeighborsClassifier(),
    # "GaussianNB": GaussianNB(),
    "MLP": MLPClassifier(),
}


@time_wrapper
def try_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    models: Dict[str, Any],
) -> Dict[str, float]:
    """try_models

    Args:
        X_train (pd.DataFrame): The training data
        y_train (pd.Series): The training target
        X_test (pd.DataFrame): The testing data
        y_test (pd.Series): The testing target
        models (Dict[str, Any]): The models to try

    Returns:
        Dict[str, float]: The model scores
    """
    scores = {}
    for name, model in models.items():
        start_time = time.time()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        scores[name] = f1_score(
            y_test, y_pred, average="micro"
        )  # since it's a multiclass classification problem
        end_time = time.time()
        print(f"{name}: {scores[name]} in {end_time-start_time:.2f} seconds")

        # clear resources, if possible
        del model

    with open(os.path.join(current_dir, "scores.txt"), "w") as f:
        for name, score in scores.items():
            f.write(f"{name}: {score:.4f}\n")

    return scores


def main():
    X_train, X_test, y_train = load_final_data()

    assert (
        X_test.shape[0] == 3411 and X_train.shape[1] == X_test.shape[1]
    ), "Data shape error"

    y_train = y_train.values.ravel()

    try_models(X_train, y_train, X_test, y_train, models)


if __name__ == "__main__":
    main()
