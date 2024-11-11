import pandas as pd
import numpy as np

from typing import List, Tuple, Dict


def load_rawdata() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load raw data.
    
    Returns:
    Tuple of three DataFrames: train, test, and sample_submission.
    """
    traindata = pd.read_csv('Data/Input/train.csv', index_col='id')
    testdata = pd.read_csv('Data/Input/test.csv', index_col='id')
    
    return traindata, testdata
    
def submit_pred(y_pred: np.ndarray, filename: str = 'submission.csv') -> None:
    """
    Save predictions to a CSV file.
    
    Args:
    y_pred: np.ndarray, predictions.
    filename: str, name of the file to save.
    
    Returns:
    None
    """
    submission = pd.read_csv('Data/Input/sample_submission.csv')
    submission['y'] = y_pred
    submission.to_csv(f"Data/Output/{filename}", index=False)
    
    return None