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
from MODELS import vanilla_models

sys.path.append(current_dir + "/../")
from utils import *

sys.path.append(current_dir + "/../FeaturePreprocessing/")
from FeaturePreprocessing.process import *


current_dir = os.path.dirname(os.path.realpath(__file__))


@function_runtime_tracker
def try_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
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
        scores[name] = evaluate_model(
            X_train=X_train, y_train=y_train, model=model, cv=False
        )
        end_time = time.time()
        print(f"{name}: {scores[name]:.2f} in {end_time-start_time:.2f} seconds")

    with open(os.path.join(current_dir, "scores.txt"), "w") as f:
        for name, score in scores.items():
            f.write(f"{name}: {score:.4f}\n")

    return scores


def main():
    X_train, X_test, y_train = load_final_data()
    scores = try_models(X_train, y_train, vanilla_models)
    with open(os.path.join(current_dir, "scores.txt"), "r") as f:
        scores = f.read()
        scores = [score.split(": ") for score in scores.split("\n") if score != ""]
        scores = {score[0]: float(score[1]) for score in scores}
        with open(os.path.join(current_dir, "selected_models.txt"), "w") as f:
            for name, score in scores.items():
                if score > 0.86:
                    f.write(f"{name}|{score}\n")


if __name__ == "__main__":
    main()
