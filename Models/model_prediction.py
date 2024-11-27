from sklearn.ensemble import VotingClassifier
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import torch

if torch.cuda.is_available():
    print("Using GPU")
else:
    print("Using CPU")
    print("Consider using GPU for faster training!!!")

current_dir = os.path.dirname(os.path.realpath(__file__))
print(current_dir)
sys.path.append(current_dir)
from MODELS import finetuned_models as models

sys.path.append(current_dir + "/../")
from utils import *

sys.path.append(current_dir + "/../FeaturePreprocessing/")
from FeaturePreprocessing.process import *


current_dir = os.path.dirname(os.path.realpath(__file__))


def main(filename: str = "final_result"):
    X_train, X_test, y_train = load_final_data()

    with open(os.path.join(current_dir, "selected_models.txt"), "r") as f:
        scores = f.read()
        scores = [score.split("|") for score in scores.split("\n") if score != ""]
        excluded = ["GradientBoosting", "Bagging"]  # time_consuming
        scores = {
            score[0]: float(score[1])
            for score in scores
            if float(score[1]) > 0.86 and score[0] not in excluded
        }

    print_boundary("Selected models", fill_char="-")
    print("Scores: ", scores)
    s = [scores[name] for name in scores.keys()]
    weights = calculate_weights(s)
    print("Weights: ", weights)
    print_boundary(fill_char="-")

    print_boundary("Using XGB to evaluate the model", fill_char="*")
    xgb = models["XGB"]
    score = evaluate_model(X_train, y_train, xgb, cv=True)
    print("XGB score: ", score)
    print_boundary(fill_char="*")

    # 确保 estimators 和 weights 的模型名称一致
    estimators = [(name, models[name]) for name in scores.keys()]
    voting = VotingClassifier(estimators=estimators, voting="soft", weights=weights)

    print_boundary("Using Voting to train the model", fill_char="*")
    print("Train the model...", [name for name in scores.keys()])
    print_boundary(fill_char="*")

    @function_runtime_tracker
    def train(model):
        # train on the whole dataset
        model.fit(X_train, y_train)
        return model

    voting = train(voting)
    y_pred = voting.predict(X_test)
    
    print_boundary(f"Generate the prediction file: {filename}.csv", fill_char="=")
    submit_pred(y_pred, f"{filename}.csv")


if __name__ == "__main__":
    main()
