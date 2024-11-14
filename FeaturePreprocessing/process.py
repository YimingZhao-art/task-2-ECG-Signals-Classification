import os
import sys
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import f1_score
import lightgbm as lgb

current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current_dir)
sys.path.append(current_dir + "/../")
print(current_dir)
data_folder = os.path.join(current_dir, "../Data")
os.makedirs(data_folder, exist_ok=True)  # create the data folder if it doesn't exist
final_data_folder = os.path.join(data_folder, "final")
os.makedirs(final_data_folder, exist_ok=True)

from utils import *

try:
    y_pred = np.zeros((3411))
    # test submission functionality
    submit_pred(y_pred, "test_submission.csv")
except Exception as e:
    print("Unable to submit the file. Error:", e)
    exit(1)


def imputeOrScale(
    processor: object,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    use_all: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """imputeOrScale

    Args:
        processor (object): Imputer or Scaler object
        X_train (pd.DataFrame): The training data
        X_test (pd.DataFrame): The testing data
        use_all (bool, optional): Whether to use all data for fitting. Defaults to False.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: The transformed training and testing data
    """
    if use_all:
        X_all = pd.concat([X_train, X_test], axis=0)
        X_all = pd.DataFrame(processor.fit_transform(X_all), columns=X_all.columns)
        processor.fit(X_all)
        X_train = pd.DataFrame(processor.transform(X_train), columns=X_train.columns)
        X_test = pd.DataFrame(processor.transform(X_test), columns=X_test.columns)

    else:
        X_train = pd.DataFrame(
            processor.fit_transform(X_train), columns=X_train.columns
        )
        X_test = pd.DataFrame(processor.transform(X_test), columns=X_test.columns)

    return X_train, X_test


def handleImbalance(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    skip: bool = False,
    method: str = "over",
    target_ratio_dict: dict = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """handleImbalance

    Args:
        X_train (pd.DataFrame): The training data
        y_train (pd.DataFrame): The training labels
        skip (bool, optional): Whether to skip the imbalance handling. Defaults to False.
        method (str, optional): The method to use for handling imbalance. Defaults to "over".
        target_ratio_dict (dict, optional): The target ratio dictionary. Defaults to None.


    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: The transformed training data and labels
    """
    if skip:
        return X_train, y_train

    if method == "over":
        smote = SMOTE(
            sampling_strategy=target_ratio_dict if target_ratio_dict else "auto",
            random_state=42,
        )
        X_train, y_train = smote.fit_resample(X_train, y_train)

    elif method == "under":
        rus = RandomUnderSampler(
            sampling_strategy=target_ratio_dict if target_ratio_dict else "auto",
            random_state=42,
        )
        X_train, y_train = rus.fit_resample(X_train, y_train)

    return X_train, y_train

@time_wrapper
def processData(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    imputer: object = SimpleImputer(strategy="mean"),
    scaler: object = StandardScaler(),
    save: bool = False,
    imbalance: bool = False,
    imbalance_method: str = "over",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """processData

    Args:
        X_train (pd.DataFrame): The training data
        X_test (pd.DataFrame): The testing data
        y_train (pd.DataFrame): The training labels
        imputer (object, optional): The imputer object. Defaults to SimpleImputer(strategy='mean').
        scaler (object, optional): The scaler object. Defaults to StandardScaler().
        save (bool, optional): Whether to save the processed data. Defaults to False

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: The transformed training and testing data and labels
    """
    # Impute missing values
    X_train, X_test = imputeOrScale(imputer, X_train, X_test, use_all=True)

    # Scale the data
    X_train, X_test = imputeOrScale(scaler, X_train, X_test)

    # Handle imbalance
    X_train, y_train = handleImbalance(
        X_train=X_train, y_train=y_train, skip=not imbalance, method=imbalance_method
    )

    if save:
        X_train.to_csv(os.path.join(final_data_folder, "X_train.csv"), index=False)
        X_test.to_csv(os.path.join(final_data_folder, "X_test.csv"), index=False)
        y_train.to_csv(os.path.join(final_data_folder, "y_train.csv"), index=False)

    print_boundary("Data Processing Complete")
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print_boundary("")
    return X_train, X_test, y_train


def main():
    # Load the data
    X_train_features = pd.read_csv(
        os.path.join(data_folder, "features/X_train_features.csv"), index_col=0
    )
    X_test_features = pd.read_csv(
        os.path.join(data_folder, "features/X_test_features.csv"), index_col=0
    )
    y_train = pd.read_csv(os.path.join(data_folder, "Input/y_train.csv"), index_col=0)

    # Process the data
    X_train, X_test, y_train = processData(
        X_train_features,
        X_test_features,
        y_train,
        save=True,
        scaler=StandardScaler(),
        imputer=SimpleImputer(strategy="mean"),
        imbalance=False,
        imbalance_method="over",
    )
    
    # split while keep the imbalance ratio
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train.values.ravel(),
        test_size=0.2,
        stratify=y_train,
        random_state=42,
    )

    # it is multi-class classification
    # use multi_logloss as objective

    lgb_params = {
        'objective': 'multiclass',
        'num_class': 4,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'max_depth': 7,
        'n_estimators': 1000,
        'n_jobs': -1,
        'seed': 42,
        'early_stopping_rounds': 100,
    }

    lgbm = lgb.LGBMClassifier(**lgb_params)

    lgbm.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])

    y_pred = lgbm.predict(X_val)
    print(f"F1 Score: {f1_score(y_val, y_pred, average='weighted')}")

if __name__ == "__main__":
    main()
