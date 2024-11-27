from FeatureExtraction.feats_manual import main as extract_features_manual
from FeaturePreprocessing.process import main as process_data
from NNModels.DeepNet.trainmodel import main as extract_features_deepnet
from NNModels.ENCASE.trainmodel import main as extract_features_encase
from NNModels.gather_NN_features import main as combine_features
from Models.model_prediction import main as model_prediction

import os
import sys

current_dir = os.path.dirname(os.path.realpath(__file__))
print(current_dir)
sys.path.append(current_dir)
from utils import *


def main(have_extracted=True):
    print_boundary("Whole Process", fill_char="#")

    print_boundary("Extracting features", fill_char="=")

    print_boundary("Using different packages to extract features", fill_char="*")

    if not have_extracted:
        extract_features_manual()  # this save the features/X_train_features.csv, features/X_test_features.csv
    process_data()  # this save the final/p2_X_train.csv, final/p2_X_test.csv, final/p2_y_train.csv

    print_boundary("Using the Neural Network Models to extract features", fill_char="*")
    if not have_extracted:
        # manual_2
        extract_features_deepnet(test=True)  # test the functionality of the function
        extract_features_encase(test=True)  # test the functionality of the function

    combine_features()  # this save the final/p1_X_train.csv, final/p1_X_test.csv, final/p1_y_train.csv

    print_boundary("Finished extracting features", fill_char="=")

    print_boundary("Training the model", fill_char="=")
    model_prediction(filename="test_result")

if __name__ == "__main__":
    main(have_extracted=True)