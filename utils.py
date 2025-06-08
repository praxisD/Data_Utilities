import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

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