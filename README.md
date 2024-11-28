# Advanced Machine Learning - Project 2 - ECG Signals Classification

This project focuses on classifying time-series data from ECG signals. The authors reference various resources, re-implementing their methods while also making significant improvements (see Reference). The two primary repositories referred to in this work achieve F1 scores of 0.847 and 0.825, respectively. However, through thorough analysis of the strengths and weaknesses of these approaches, the authors have developed a novel method that achieves an F1 score of **0.864** on the public leaderboard. This represents a notable improvement over the existing implementations, highlighting the impact of the authors' contributions.

## Table of Contents

- [Device Requirements](#device-requirements)
  - [Feature Extraction](#feature-extraction)
  - [Neural Network Models](#neural-network-models)
  - [Model Prediction](#model-prediction)
- [Run the Project](#run-the-project)
- [Main Process](#main-process)
  - [Feature Extraction](#feature-extraction-1)
    - [Manual Features](#manual-features)
    - [Neural Network](#neural-network-1)
  - [Feature Process](#feature-process)
  - [Model Prediction](#model-prediction-1)
- [Reference](#reference)

## Device Requirements

### Feature Extraction

- `FeatureExtraction/feats_manual1.py`: Approximately 20 minutes on Intel i7.
- `FeatureExtraction/feats_manual2.py`: **Must be run on Ubuntu.**

### Neural Network Models

- `NNModels/ENCASE/trainmodel.py`: Runtime is approximately 120 minutes on A100 (Colab, 30 computing units)
- `NNModels/DeepNet/trainmodel.py`: Runtime is approximately 300 minutes on 3080Ti (Ubuntu).

### Model Prediction

- **Recommendation**: Strongly recommended to use GPU for acceleration (e.g., XGBoost, LightGBM).
- **Generate Predictions**: Takes around 7 minutes.

## Run the Project

1. Install the requirements

   > pip install -r requirements.txt
   >
2. Run the `main.py`, which will execute the entire process from raw data to final prediction

   > python main.py --have_extracted=False
   >

   For testing the project, you can contact the authors via [mail](zym0303@connect.hku.hk) for the extracted data. Then you can start with the extracted features and directly predict the output.

## Main Process

### Feature Extraction

The feature extraction uses manually extracted features and the last layers of neural networks as features.

#### Manual Features

Manual extracted features include:

* RR Interval
* R Amplitude
* Q Amplitude
* QRS Duration
* Heart Rate Variability
* Frequency Domain Features

In the implementation of the [repo1](https://github.com/ImaGoodFella/aml_projects/tree/main/project2), it lacks the statistics of those features. Therefore, the authors concatenated those features as well, which showed improvements compared to the [repo1](https://github.com/ImaGoodFella/aml_projects/tree/main/project2). Hence, even though the [repo2](https://github.com/mchami02/ECG-Signal-Classification-Task) only achieved a score of about 0.82, it mentioned important features.

#### Neural Network

This is the part that consumes the most time in the projects.

1. ENCASE: Extracts the final layer with 4 features (different from the referred [repo1](https://github.com/ImaGoodFella/aml_projects/tree/main/project2)).
2. Goodfellow: Extracts the final layer with 64 features.

### Feature Process

1. Abnormal values, including: `nan`, `inf`, and `-inf`. The authors impute `nan` with the mean and delete the column with infinity.
2. Feature selection: Originally ~2700 features. The authors tested ~2700, 500, and 300, which showed that 500 and 2700 are both better. When considering time complexity, 500 may be preferred, but for the final submission, 2700 features are used.

### Model Prediction

Similar to Project 1, the authors still use some base models and ensemble them. However, there are main problems with [repo1](https://github.com/ImaGoodFella/aml_projects/tree/main/project2https://). The [repo1](https://github.com/ImaGoodFella/aml_projects/tree/main/project2https://) still uses Stacking instead of Majority Voting. According to the authors' testing, Majority Voting (soft) shows a significant improvement in both time (1/5 time) and performance (0.05 higher).

Since the authors use Majority Voting (soft), only base models with `predict_proba` can be used. This time, the authors use:

* LGBM | 0.8801
* XGB | 0.8891
* CatBoost | 0.8701
* ExtraTrees | 0.8701
* Bagging | 0.8633
* RandomForest | 0.8701

The score after the model name is the validation score, which is used to weight the model in the Voting. Note that Gradient Boosting is excluded since it is too slow.

The weight is calculated by score using the formula:

$$
w_i = \frac{e^{\frac{s_i - \min(s)}{\delta}}}{\sum_{j} e^{\frac{s_j - \min(s)}{\delta}}}

$$

## Reference

[Repo1](https://github.com/ImaGoodFella/aml_projects/tree/main/project2)

[Repo2](https://github.com/mchami02/ECG-Signal-Classification-Task)
