from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
    BaggingClassifier,
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

vanilla_models = {
    # "GaussianProcess": GaussianProcessClassifier(kernel=RBF(), n_jobs=-1),
    "LGBM": LGBMClassifier(n_jobs=-1, verbose=-1, random_state=42),
    "XGB": XGBClassifier(n_jobs=-1, random_state=42),
    "CatBoost": CatBoostClassifier(verbose=0, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(
        verbose=0, random_state=42
    ),  # too slow
    "HistGradientBoosting": HistGradientBoostingClassifier(verbose=0, random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "ExtraTrees": ExtraTreesClassifier(random_state=42, n_jobs=-1),
    "Bagging": BaggingClassifier(random_state=42, n_jobs=-1),
    "RandomForest": RandomForestClassifier(random_state=42, n_jobs=-1),
    "LogisticRegression": LogisticRegression(random_state=42, n_jobs=-1),
    "Ridge": RidgeClassifier(random_state=42),
    "SGD": SGDClassifier(random_state=42),
    "PassiveAggressive": PassiveAggressiveClassifier(random_state=42),
    "SVC": SVC(random_state=42),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "KNeighbors": KNeighborsClassifier(n_jobs=-1),
    # "GaussianNB": GaussianNB(),
    "MLP": MLPClassifier(random_state=42),
}

finetuned_models = {
    "GaussianProcess": GaussianProcessClassifier(kernel=RBF(), n_jobs=-1),
    "LGBM": LGBMClassifier(
        n_jobs=-1,
        verbose=-1,
        max_depth=5,
        learning_rate=0.07,
        n_estimators=1000,
        subsample=0.9,
        random_state=42,
    ),
    "XGB": XGBClassifier(
        n_jobs=-1,
        n_estimators=800,
        learning_rate=0.03,
        verbosity=0,
        min_child_weight=5,
        max_depth=7,
        random_state=42,
    ),
    "CatBoost": CatBoostClassifier(
        verbose=0, auto_class_weights="SqrtBalanced", random_state=42
    ),
    "GradientBoosting": GradientBoostingClassifier(verbose=0, random_state=42),
    "HistGradientBoosting": HistGradientBoostingClassifier(
        learning_rate=0.01, max_iter=500, class_weight="balanced", random_state=42
    ),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "ExtraTrees": ExtraTreesClassifier(
        n_estimators=500,
        max_depth=15,
        criterion="entropy",
        class_weight="balanced_subsample",
        random_state=42,
    ),
    "Bagging": BaggingClassifier(random_state=42, n_estimators=100, n_jobs=-1),
    "RandomForest": RandomForestClassifier(random_state=42, n_jobs=-1, n_estimators=700, class_weight="balanced"),
    "LogisticRegression": LogisticRegression(random_state=42),
    "Ridge": RidgeClassifier(random_state=42),
    "SGD": SGDClassifier(random_state=42),
    "PassiveAggressive": PassiveAggressiveClassifier(random_state=42),
    "SVC": SVC(random_state=42),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "KNeighbors": KNeighborsClassifier(n_jobs=-1),
    "GaussianNB": GaussianNB(),
    "MLP": MLPClassifier(random_state=42),
}

if __name__ == "__main__":
    import numpy as np
    # create a fake dataset
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 3, 100)
    
    # check the models are working
    for name, model in vanilla_models.items():
        model.fit(X, y)
        print(f"Model {name} is trained.")
        
    for name, model in finetuned_models.items():
        model.fit(X, y)
        print(f"Model {name} is trained.")