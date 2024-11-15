from sklearn.ensemble import VotingClassifier
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
from MODELS import finetuned_models as models

sys.path.append(current_dir + "/../")
from utils import *

sys.path.append(current_dir + "/../FeaturePreprocessing/")
from FeaturePreprocessing.process import *


current_dir = os.path.dirname(os.path.realpath(__file__))


def main():
    X_train, X_test, y_train = load_final_data()

    with open(os.path.join(current_dir, "selected_models.txt"), "r") as f:
        scores = f.read()
        scores = [score.split("|") for score in scores.split("\n") if score != ""]
        scores = {
            score[0]: float(score[1]) for score in scores if float(score[1]) > 0.86
        }

    print("Scores: ", scores)
    s = [scores[name] for name in scores.keys()]
    weights = calculate_weights(s)
    print("Weights: ", weights)

    # 确保 estimators 和 weights 的模型名称一致
    estimators = [(name, models[name]) for name in scores.keys()]
    voting = VotingClassifier(estimators=estimators, voting="soft", weights=weights)

    print("Train the model...", estimators)

    @function_runtime_tracker
    def train(model):
        # train on the whole dataset
        model.fit(X_train, y_train)
        return model

    voting = train(voting)
    # predict on the test set
    y_pred = voting.predict(X_test)
    submit_pred(y_pred, "finetune_softvote.csv")

    print("Trying models...")


if __name__ == "__main__":
    main()
