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
    "GaussianProcess": GaussianProcessClassifier(kernel=RBF(), n_jobs=-1),
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
    "GaussianNB": GaussianNB(),
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
    X_train = pd.read_csv('Data/final/final_X_train.csv', header=None).iloc[1:, 1:]
    X_test = pd.read_csv('Data/final/final_X_test.csv', header=None).iloc[1:, 1:]
    y_train = pd.read_csv('Data/final/final_y_train.csv', header=None).iloc[1:, 1:]

    
    assert X_test.shape[0] == 3411 and X_train.shape[1] == X_test.shape[1], "Data shape error"

    y_train = y_train.values.ravel()

    with open(os.path.join(current_dir, "scores.txt"), "r") as f:
        scores = f.read()
        scores = [score.split(": ") for score in scores.split("\n") if score != ""]
        scores = {
            score[0]: float(score[1]) for score in scores if float(score[1]) > 0.86
        }

    def calculate_weights(scores, base=3.0, delta=0.02):
        min_score = min(scores)
        weights = np.array([base ** ((s - min_score) / delta) for s in scores])
        weights /= np.sum(weights)
        return weights

    print("Scores: ", scores)
    s = [scores[name] for name in scores.keys()]
    weights = calculate_weights(s)
    print("Weights: ", weights)

    # 确保 estimators 和 weights 的模型名称一致
    estimators = [(name, models[name]) for name in scores.keys()]
    stack = StackingClassifier(estimators=estimators)
    voting = VotingClassifier(estimators=estimators, voting="soft", weights=weights)

    # print_boundary("cv for voting", fill_char="*")
    # start_time = time.time()
    # # make f1 scorer
    # f1_scorer = make_scorer(f1_score, average="micro")
    # cv_score = cross_val_score(
    #     voting, X_train, y_train, cv=5, scoring=f1_scorer, n_jobs=-1
    # )
    # print("Cross validation score: ", cv_score)
    # print("Mean cross validation score: ", np.mean(cv_score))
    # end_time = time.time()
    # print_boundary(f"used {start_time-end_time}", fill_char="*")

    # print_boundary("cv for stacking", fill_char="*")
    # start_time = time.time()
    # cv_score2 = cross_val_score(
    #     stack, X_train, y_train, cv=5, scoring=f1_scorer, n_jobs=-1
    # )
    # print("Cross validation score: ", cv_score2)
    # print("Mean cross validation score: ", np.mean(cv_score2))
    # end_time = time.time()
    # print_boundary(f"used {start_time-end_time}", fill_char="*")
    
    # exit()

    print("Train the model...", estimators)

    @time_wrapper
    def train(model):
        # train on the whole dataset
        model.fit(X_train, y_train)
        return model

    voting = train(voting)
    # predict on the test set
    y_pred = voting.predict(X_test)
    submit_pred(y_pred, "softvote.csv")

    print("Trying models...")


if __name__ == "__main__":
    main()
