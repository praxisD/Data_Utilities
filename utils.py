"""
utils.py — common data-prep and modelling helpers

Status: under active development.
TODO:
  * generalize cross-validation to arbitrary scoring functions
  * add more data preprocessing functions
  * add more model evaluation metrics
  * add more model hyperparameter tuning functions
"""



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import mean_poisson_deviance
from sklearn import linear_model


####################################################
####### --- Data Preprocessing Functions --- #######
####################################################



def add_outlier_label(df, col, p_up=1.5, p_low=1.5):
    """
    Adds a binary column indicating outliers in the specified column using the IQR method.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.
        col (str): The name of the column to analyze.
        p_up, p_low (float): Parameters to multiply IQR in the formula, default to 1.5
        
    Returns:
        pd.DataFrame: A copy of the DataFrame with an additional column 
        indicating outliers (1 for outlier, 0 otherwise).
    """
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not included in DataFrame")

    col_data = df[col]
    IQR = col_data.quantile(0.75) - col_data.quantile(0.25)
    upper = col_data.quantile(0.75) + p_up * IQR
    lower = col_data.quantile(0.25) - p_low * IQR

    X = df.copy()
    X['Extr_' + col] = 0
    X.loc[X[col] >= upper, 'Extr_' + col] = 1
    X.loc[X[col] <= lower, 'Extr_' + col] = 1
    
    return X


def sets_mutually_exclusive(sets):
    """
    Determine if a collection of sets are pairwise disjoint (i.e., no element appears in more than one set).

    Args:
        sets (iterable of set): A sequence of sets to check for mutual exclusivity.

    Returns:
        bool: True if no two sets share an element, False otherwise.
    """
    seen = set()  # Accumulates elements we've already encountered

    for s in sets:
        # If current set shares any element with seen, they're not exclusive
        if seen.intersection(s):
            return False
        # Add elements of this set to seen for future checks
        seen.update(s)

    return True




####################################################
######## --- Poisson Deviance Functions --- ########
####################################################


def poisson_deviance(y_true, y_pred):
    """
    Compute Poisson deviance metrics between true and predicted values.

    Returns:
        dict:
            - 'standard': Poisson deviance between y_true and y_pred.
            - 'compared_to_mean': Poisson deviance between y_true and its mean prediction.
            - 'relative': Ratio of deviance with y_pred to deviance with mean prediction.
              Values < 1 imply that y_pred improves over simply predicting the mean.
    """
    # Compute deviance between predictions and true values
    dev = mean_poisson_deviance(y_true, y_pred)

    # Compute deviance between true values and predicting their mean
    dev_mean = mean_poisson_deviance(y_true, np.full(len(y_true), y_true.mean()))

    # Relative deviance: how much better (or worse) y_pred is compared to mean prediction
    rel = dev / dev_mean

    return {
        'standard': dev,
        'compared_to_mean': dev_mean,
        'relative': rel
    }


def train_and_validate(full_df, features, model, train_start_year, target):
    """
    Fit `model` on data from `train_start_year`, validate on the following year, 
    and compute train/validation scores along with Poisson deviance metrics.

    Parameters:
    - full_df: pd.DataFrame containing data with a 'start_year' column
    - features: list of column names to use as predictors
    - model: scikit-learn estimator implementing fit(), score(), and predict()
    - train_start_year: int, the year to train on
    - target: string, name of the target column

    Returns:
    dict with:
      - 'train_score': model.score on training set (e.g. R^2 or D^2)
      - 'val_score': model.score on validation set (e.g. R^2 or D^2)
      - 'predictions': array of predicted values for validation year
      - 'Poisson_deviance_predictions': standard Poisson deviance of preds
      - 'Poisson_deviance_mean': Poisson deviance compared to mean model
      - 'relative_Poisson_deviance': ratio of model deviance to mean deviance
    """
    # Make a copy to avoid modifying original DataFrame
    df = full_df.copy()

    # Split into training and validation sets by year
    X_train = df[df['start_year'] == train_start_year][features]
    y_train = df[df['start_year'] == train_start_year][target]
    X_val = df[df['start_year'] == train_start_year + 1][features]
    y_val = df[df['start_year'] == train_start_year + 1][target]

    # Fit the model
    model.fit(X_train, y_train)

    # Compute scores
    train_score = model.score(X_train, y_train)
    val_score = model.score(X_val, y_val)

    # Generate predictions on validation set
    preds = model.predict(X_val)

    # Compute Poisson deviance metrics
    dev = poisson_deviance(y_val, preds)

    return {
        'train_score': train_score,
        'val_score': val_score,
        'predictions': preds,
        'Poisson_deviance_predictions': dev['standard'],  # model vs truth
        'Poisson_deviance_mean': dev['compared_to_mean'],  # mean model baseline
        'relative_Poisson_deviance': dev['relative']       # ratio of above two
    }


def average_relative_Poisson_deviance(full_df, features, model, valid_years, target):
    """
    Compute weighted average of relative Poisson deviance across years.
    """
    lengths = []  # sample counts per year
    scores = []   # deviance scores per year

    for year in valid_years:
        lengths.append(len(full_df[full_df['start_year'] == year]))
        scores.append(train_and_validate(full_df, features, model, year, target)['relative_Poisson_deviance'])

    # return overall weighted average
    return np.average(scores, weights=lengths)


def fine_tuned_1d_search_poisson_deviance(full_df, features, model_class, valid_years, target, param_name, log_param_values, fine_tune_steps=21):
    """
    Perform a grid search with an initial logarithmic scale search followed by a fine-tuning linear search, minimizing relative Poisson deviance.
    """
    # First step: Coarse search on a logarithmic scale
    best_param = None
    best_score = float('inf')

    for value in log_param_values:

        model = model_class(**{param_name: value})

        current_score = average_relative_Poisson_deviance(full_df, features, model, valid_years, target)
        
        if current_score < best_score:
            best_score = current_score
            best_param = value

    # Second step: Fine-tune on a linear scale around the best_param
    lower_bound = best_param / 10
    upper_bound = best_param * 10
    linear_param_values = np.linspace(lower_bound, upper_bound, fine_tune_steps)

    for value in linear_param_values:

        model = model_class(**{param_name: value})

        current_score = average_relative_Poisson_deviance(full_df, features, model, valid_years, target)
        
        if current_score < best_score:
            best_score = current_score
            best_param = value

    return best_param, best_score