#!/usr/bin/env python3
"""
Data Preprocessing Module for Air Quality Analysis
Contains functions for data preparation, scaling, sequence creation, and dataloader setup
"""
import warnings
from typing import Dict, Tuple, List, Optional, Any

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

from aq_edge.utils.logging import LoggerHandler

warnings.filterwarnings('ignore')
logger = LoggerHandler('preprocessing')

def load_station_data(
        stations: List[str] = ['APLAN', 'MHH', 'PFM', 'PGB', 'PLIB', 'USAM', 'UTEC'],
        data_dir: str = '../data/air/'
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Load air quality data for specified stations from CSV files.

    Args:
        stations (List[str]): List of station codes to load.
        data_dir (str): Directory containing station CSV files.

    Returns:
        Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
            - Dictionary mapping station code to its DataFrame.
            - Combined DataFrame of all stations.
    """

    station_data = {}
    all_data = []

    logger.info("Loading station data...")
    for station in stations:
        try:
            df = pd.read_csv(f'{data_dir}{station}.csv', sep=';')
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

            # Remove rows with invalid timestamps (NaT values)
            initial_count = len(df)
            df = df.dropna(subset=['Timestamp'])
            if len(df) < initial_count:
                logger.warning(f"  Removed {initial_count - len(df)} rows with invalid timestamps")

            # Set 'Timestamp' as the index
            df.set_index('Timestamp', inplace=True)

            df['Station'] = station
            station_data[station] = df
            all_data.append(df)
            logger.info(f"[OK] Loaded {station}: {len(df)} records")
        except FileNotFoundError:
            logger.error(f"File not found: {data_dir}{station}.csv")
        except Exception as e:
            logger.error(f"Error loading {station}: {e}")

    # Combine all data
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"\n[OK] Total combined records: {len(combined_df)}")
        logger.info(f"[OK] Columns: {combined_df.columns.tolist()}")
    else:
        combined_df = pd.DataFrame()
        logger.error("\n[ERROR] No data loaded successfully")

    return station_data, combined_df

def split_data(
    data: pd.DataFrame,
    train_end_timestamp: pd.Timestamp,
    val_end_timestamp: pd.Timestamp
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    (docstring unchanged)
    """
    def _match_tz(ts):
        # Make the cutoff timezone-compatible with the index (naive cutoffs are taken as UTC)
        ts = pd.Timestamp(ts)
        if data.index.tz is not None:
            return ts.tz_localize(data.index.tz) if ts.tzinfo is None else ts.tz_convert(data.index.tz)
        return ts.tz_convert(None) if ts.tzinfo is not None else ts

    train_end = _match_tz(train_end_timestamp)
    val_end = _match_tz(val_end_timestamp)
    train = data[data.index <= train_end]
    val = data[(data.index > train_end) & (data.index <= val_end)]
    test = data[data.index > val_end]
    return train, val, test

def _make_scaler(name: str):
    scalers = {"minmax": MinMaxScaler, "standard": StandardScaler, "robust": RobustScaler}
    if name.lower() not in scalers:
        raise ValueError(f"Unsupported scaler type: {name}. Use 'minmax', 'standard', or 'robust'.")
    return scalers[name.lower()]()

def scale_data(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    test_data: pd.DataFrame,
    scaler: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
    """Fit the scaler on train only, transform all three splits."""
    train_scaler = _make_scaler(scaler)
    logger.info(f"[OK] scaler: {scaler}")
    return (
        train_scaler.fit_transform(train_data),
        train_scaler.transform(val_data),
        train_scaler.transform(test_data),
        train_scaler,
    )

def split_train_val_test_X_y(
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        test_data: pd.DataFrame,
        features: List[str],
        target: str
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Split train, validation, and test data into features (X) and target (y).

    Args:
        train_data (pd.DataFrame): Training data
        val_data (pd.DataFrame): Validation data
        test_data (pd.DataFrame): Test data
        features (List[str]): Feature column names
        target (str): Target column name

    Returns:
        Tuple: X_train, y_train, X_val, y_val, X_test, y_test
    """
    # Split training data
    X_train = train_data[features].copy()
    y_train = train_data[target].copy()

    # Split validation data
    X_val = val_data[features].copy()
    y_val = val_data[target].copy()

    # Split test data
    X_test = test_data[features].copy()
    y_test = test_data[target].copy()

    return X_train, y_train, X_val, y_val, X_test, y_test

def create_sequences(
    X_scaled: np.ndarray,
    y_scaled: np.ndarray,
    timestamps: pd.Index,
    input_sequence_length: int,
    output_sequence_length: int = 1
) -> Tuple[np.ndarray, np.ndarray, List[pd.Index], List[pd.Index]]:
    """
    Create sequences for time series prediction with corresponding timestamps.

    Args:
        X_scaled (np.ndarray): Scaled feature data
        y_scaled (np.ndarray): Scaled target data
        timestamps (pd.Index): Corresponding timestamps
        input_sequence_length (int): Length of input sequences
        output_sequence_length (int): Length of output sequences (default: 1)

    Returns:
        Tuple: X_sequences, y_sequences, input_timestamps, output_timestamps
    """
    X_sequences = []
    y_sequences = []
    input_timestamps = []
    output_timestamps = []

    for i in range(len(X_scaled) - input_sequence_length - output_sequence_length + 1):
        X_seq = X_scaled[i:i + input_sequence_length]
        y_seq = y_scaled[i + input_sequence_length:i + input_sequence_length + output_sequence_length]

        # Get corresponding timestamps
        input_ts = timestamps[i:i + input_sequence_length]
        output_ts = timestamps[i + input_sequence_length:i + input_sequence_length + output_sequence_length]

        X_sequences.append(X_seq)
        y_sequences.append(y_seq)
        input_timestamps.append(input_ts)
        output_timestamps.append(output_ts)

    return np.array(X_sequences), np.array(y_sequences), input_timestamps, output_timestamps


def create_datasets(
    X_train_seq: np.ndarray,
    y_train_seq: np.ndarray,
    X_val_seq: np.ndarray,
    y_val_seq: np.ndarray,
    X_test_seq: np.ndarray,
    y_test_seq: np.ndarray
) -> Tuple[TensorDataset, TensorDataset, TensorDataset]:
    """
    Create PyTorch TensorDatasets from sequence arrays.

    Args:
        X_train_seq, y_train_seq: Training sequences
        X_val_seq, y_val_seq: Validation sequences
        X_test_seq, y_test_seq: Test sequences

    Returns:
        Tuple: train_dataset, validation_dataset, test_dataset
    """
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_seq)
    y_train_tensor = torch.FloatTensor(y_train_seq)
    X_val_tensor = torch.FloatTensor(X_val_seq)
    y_val_tensor = torch.FloatTensor(y_val_seq)
    X_test_tensor = torch.FloatTensor(X_test_seq)
    y_test_tensor = torch.FloatTensor(y_test_seq)

    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    validation_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    return train_dataset, validation_dataset, test_dataset

def create_dataloaders(
    train_dataset: TensorDataset,
    validation_dataset: TensorDataset,
    test_dataset: TensorDataset,
    batch_size: int = 32,
    shuffle_train: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders from datasets.

    Args:
        train_dataset, validation_dataset, test_dataset: TensorDatasets
        batch_size (int): Batch size for dataloaders (default: 32)
        shuffle_train (bool): Whether to shuffle training data (default: True)

    Returns:
        Tuple: train_dataloader, validation_dataloader, test_dataloader
    """
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        drop_last=False
    )

    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )

    return train_dataloader, validation_dataloader, test_dataloader

def prepare_data(
    data: pd.DataFrame,
    features: List[str],
    target: str,
    train_end_timestamp: pd.Timestamp,
    val_end_timestamp: pd.Timestamp,
    input_sequence_length: int = 24,
    output_sequence_length: int = 1,
    batch_size: int = 32,
    scaler: str = 'standard'
) -> Dict[str, Any]:
    """
    Data preparation using fixed timestamp splits and a single scaler.
    Args:
        data (pd.DataFrame): Input DataFrame with datetime index
        features (List[str]): List of feature column names
        target (str): Target column name
        train_end_timestamp (pd.Timestamp): Last timestamp for training set (inclusive)
        val_end_timestamp (pd.Timestamp): Last timestamp for validation set (inclusive)
        input_sequence_length (int): Length of input sequences (default: 24)
        output_sequence_length (int): Length of output sequences (default: 1)
        batch_size (int): Batch size for dataloaders (default: 32)
    Returns:
        Dict[str, Any]: Dictionary containing all prepared data components
    """
    # Step 1: Split data into train/val/test using timestamps
    all_cols = features + [target]
    train_data, val_data, test_data = split_data(
        data[all_cols], train_end_timestamp, val_end_timestamp
    )

    # Step 2: Scale all columns using a single scaler
    train_scaled, val_scaled, test_scaled, scaler = scale_data(
        train_data, val_data, test_data, scaler=scaler
    )

    train_scaled = pd.DataFrame(train_scaled, columns=all_cols, index=train_data.index)
    val_scaled = pd.DataFrame(val_scaled, columns=all_cols, index=val_data.index)
    test_scaled = pd.DataFrame(test_scaled, columns=all_cols, index=test_data.index)

    # Step 3: Split X and y from scaled arrays
    X_train, y_train, X_val, y_val, X_test, y_test = split_train_val_test_X_y(
        train_scaled, val_scaled, test_scaled, features, target
    )

    # Step 4: Get timestamps
    train_timestamps = train_data.index
    val_timestamps = val_data.index
    test_timestamps = test_data.index

    # Step 5: Create sequences
    X_train_seq, y_train_seq, train_input_timestamps, train_output_timestamps = create_sequences(
        X_train.values, y_train.values, train_data.index, input_sequence_length, output_sequence_length
    )
    X_val_seq, y_val_seq, val_input_timestamps, val_output_timestamps = create_sequences(
        X_val.values, y_val.values, val_data.index, input_sequence_length, output_sequence_length
    )
    X_test_seq, y_test_seq, test_input_timestamps, test_output_timestamps = create_sequences(
        X_test.values, y_test.values, test_data.index, input_sequence_length, output_sequence_length
    )

    # Step 6: Create datasets and dataloaders
    train_dataset, validation_dataset, test_dataset = create_datasets(
        X_train_seq, y_train_seq, X_val_seq, y_val_seq, X_test_seq, y_test_seq
    )
    train_dataloader, validation_dataloader, test_dataloader = create_dataloaders(
        train_dataset, validation_dataset, test_dataset, batch_size
    )

    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,

        'X_test': X_test, 'y_test': y_test,
        'train_timestamps': train_timestamps,
        'val_timestamps': val_timestamps,
        'test_timestamps': test_timestamps,
        'scaler': scaler,
        'X_train_seq': X_train_seq, 'y_train_seq': y_train_seq,
        'X_val_seq': X_val_seq, 'y_val_seq': y_val_seq,
        'X_test_seq': X_test_seq, 'y_test_seq': y_test_seq,
        'train_input_timestamps': train_input_timestamps,
        'train_output_timestamps': train_output_timestamps,
        'val_input_timestamps': val_input_timestamps,
        'val_output_timestamps': val_output_timestamps,
        'test_input_timestamps': test_input_timestamps,
        'test_output_timestamps': test_output_timestamps,
        'train_dataset': train_dataset,
        'validation_dataset': validation_dataset,
        'test_dataset': test_dataset,
        'train_dataloader': train_dataloader,
        'validation_dataloader': validation_dataloader,
        'test_dataloader': test_dataloader,
        'input_sequence_length': input_sequence_length,
        'output_sequence_length': output_sequence_length,
        'batch_size': batch_size,
        'num_features': len(features),
        'feature_names': features,
        'target_name': target,
    }
