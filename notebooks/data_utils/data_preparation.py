# --------------------------------------------
# ----------- Importing libraries ------------
# --------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split



# --------------------------------------------
# ----------- Function definition ------------
# --------------------------------------------

def train_val_test_split_by_date(data, train_end_date, val_end_date):
    """
    Split time series by specific dates.

    Parameters:
    -----------
    data : pandas DataFrame with DatetimeIndex
    train_end_date : str or datetime
        Last date for training set
    val_end_date : str or datetime
        Last date for validation set

    Returns:
    --------
    train, val, test : pandas DataFrame
    """
    train = data[:train_end_date]
    val = data[train_end_date:val_end_date]
    test = data[val_end_date:]

    return train, val, test


def scale(train, val, test, method='minmax'):
    """
    Scale data using only training set statistics.

    Parameters:
    -----------
    train, val, test : pandas DataFrame
    method : str
        'minmax' or 'standard'

    Returns:
    --------
    train_scaled, val_scaled, test_scaled, scaler
    """
    # Choose scaler
    if method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError("method must be 'minmax' or 'standard'")

    # Fit on train, transform all (keeping DataFrame structure)
    train_scaled = pd.DataFrame(
        scaler.fit_transform(train),
        index=train.index,
        columns=train.columns
    )

    val_scaled = pd.DataFrame(
        scaler.transform(val),
        index=val.index,
        columns=val.columns
    )

    test_scaled = pd.DataFrame(
        scaler.transform(test),
        index=test.index,
        columns=test.columns
    )

    return train_scaled, val_scaled, test_scaled, scaler


def inverse_scale(data, scaler):
    """
    Inverse transform scaled data back to original scale.

    Parameters:
    -----------
    data : numpy array or pandas DataFrame
        Scaled data
    scaler : sklearn scaler object
        Fitted scaler from scale_train_only()

    Returns:
    --------
    numpy array or pandas DataFrame in original scale
    """
    return scaler.inverse_transform(data)


def split_features_target(data, target_column):
    """
    Split DataFrame into features (X) and target (y).

    Parameters:
    -----------
    data : pandas DataFrame
        The dataset to split
    target_column : str or list of str
        Name(s) of the target column(s)

    Returns:
    --------
    X : pandas DataFrame
        Features (all columns except target)
    y : pandas DataFrame or Series
        Target column(s)
    """
    if isinstance(target_column, str):
        # Single target column
        y = data[[target_column]]  # Keep as DataFrame
        X = data.drop(columns=[target_column])
    else:
        # Multiple target columns
        y = data[target_column]
        X = data.drop(columns=target_column)

    return X, y


def create_sequences(X_data, y_data, lookback, horizon):
    """
    Create sequences for multivariate multi-step time series forecasting.

    Parameters:
    -----------
    X_data : numpy array or pandas DataFrame
        Feature data (all input variables)
    y_data : numpy array or pandas DataFrame
        Target data (variable(s) to predict)
    lookback : int
        Number of past time steps to use as input (window size)
    horizon : int
        Number of future time steps to predict

    Returns:
    --------
    X : numpy array, shape (samples, lookback, n_features)
        Input sequences (past features)
    y : numpy array, shape (samples, horizon, n_targets)
        Target sequences (future target values)
    """
    X, y = [], []

    # Convert to numpy if DataFrame
    if hasattr(X_data, 'values'):
        X_data = X_data.values
    if hasattr(y_data, 'values'):
        y_data = y_data.values

    for i in range(len(X_data) - lookback - horizon + 1):
        # Input: past 'lookback' time steps of features
        X.append(X_data[i:i + lookback])

        # Output: next 'horizon' time steps of target
        y.append(y_data[i + lookback:i + lookback + horizon])

    X = np.array(X)
    y = np.array(y)

    return X, y


def save_sequences(X_train_seq, y_train_seq, X_val_seq, y_val_seq, X_test_seq, y_test_seq, scaler_train, save_dir):
    # Save sequences
    np.save(os.path.join(save_dir, "X_train_seq.npy"), X_train_seq)
    np.save(os.path.join(save_dir, "y_train_seq.npy"), y_train_seq)
    np.save(os.path.join(save_dir, "X_val_seq.npy"), X_val_seq)
    np.save(os.path.join(save_dir, "y_val_seq.npy"), y_val_seq)
    np.save(os.path.join(save_dir, "X_test_seq.npy"), X_test_seq)
    np.save(os.path.join(save_dir, "y_test_seq.npy"), y_test_seq)

    # Save scalers
    joblib.dump(scaler_train, os.path.join(save_dir, "scaler_train.pkl"))