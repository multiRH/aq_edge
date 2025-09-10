#!/usr/bin/env python3
"""
Data Preprocessing Module for Air Quality Analysis
Contains functions for data preparation, scaling, sequence creation, and dataloader setup
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, Tuple, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Import logger from utils
from aq_edge.utils.logging import LoggerHandler

# Initialize logger with the actual module name
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
    X: pd.DataFrame,
    y: pd.Series,
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Index, pd.Index, pd.Index]:
    """
    Split data into train, validation, and test sets.

    Args:
        X (pd.DataFrame): Feature data
        y (pd.Series): Target data
        train_ratio (float): Ratio for training data (default: 0.7)
        validation_ratio (float): Ratio for validation data (default: 0.15)

    Returns:
        Tuple: X_train, X_val, X_test, y_train, y_val, y_test, train_timestamps, val_timestamps, test_timestamps
    """
    total_samples = len(X)
    train_end = int(total_samples * train_ratio)
    val_end = int(total_samples * (train_ratio + validation_ratio))

    X_train = X.iloc[:train_end]
    X_val = X.iloc[train_end:val_end]
    X_test = X.iloc[val_end:]

    y_train = y.iloc[:train_end]
    y_val = y.iloc[train_end:val_end]
    y_test = y.iloc[val_end:]

    # Extract timestamps
    train_timestamps = X.index[:train_end]
    val_timestamps = X.index[train_end:val_end]
    test_timestamps = X.index[val_end:]

    return X_train, X_val, X_test, y_train, y_val, y_test, train_timestamps, val_timestamps, test_timestamps

def scaling_data(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler, MinMaxScaler]:
    """
    Scale the split data using MinMaxScaler.

    Args:
        X_train, X_val, X_test (pd.DataFrame): Feature datasets
        y_train, y_val, y_test (pd.Series): Target datasets

    Returns:
        Tuple: X_train_scaled, X_val_scaled, X_test_scaled, y_train_scaled, y_val_scaled, y_test_scaled, X_scaler, y_scaler
    """
    # Initialize scalers
    X_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    # Fit and transform training data
    X_train_scaled = X_scaler.fit_transform(X_train)
    y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()

    # Transform validation and test data using fitted scalers
    X_val_scaled = X_scaler.transform(X_val)
    X_test_scaled = X_scaler.transform(X_test)

    y_val_scaled = y_scaler.transform(y_val.values.reshape(-1, 1)).flatten()
    y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1)).flatten()

    return X_train_scaled, X_val_scaled, X_test_scaled, y_train_scaled, y_val_scaled, y_test_scaled, X_scaler, y_scaler

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

def get_prediction_timestamps(
    timestamps: pd.Index,
    input_sequence_length: int,
    output_sequence_length: int = 1
) -> pd.Index:
    """
    Generate timestamps for future predictions based on the last available timestamp.

    Args:
        timestamps (pd.Index): Original timestamps
        input_sequence_length (int): Length of input sequences used for prediction
        output_sequence_length (int): Number of future predictions

    Returns:
        pd.Index: Timestamps for future predictions
    """
    if len(timestamps) < input_sequence_length:
        raise ValueError("Not enough timestamps for the required sequence length")

    # Get the frequency of the time series
    freq = pd.infer_freq(timestamps)
    if freq is None:
        # Try to infer from the most common time difference
        time_diffs = timestamps[1:] - timestamps[:-1]
        most_common_diff = pd.Series(time_diffs).mode()[0]
        freq = most_common_diff

    # Generate future timestamps
    last_timestamp = timestamps[-1]
    future_timestamps = pd.date_range(
        start=last_timestamp + pd.Timedelta(freq),
        periods=output_sequence_length,
        freq=freq
    )

    return future_timestamps

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
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
    input_sequence_length: int = 24,
    output_sequence_length: int = 1,
    batch_size: int = 32
) -> Dict[str, Any]:
    """
    Comprehensive data preparation function for time series forecasting.

    Args:
        data (pd.DataFrame): Input DataFrame with datetime index
        features (List[str]): List of feature column names
        target (str): Target column name
        train_ratio (float): Ratio for training data (default: 0.7)
        validation_ratio (float): Ratio for validation data (default: 0.15)
        input_sequence_length (int): Length of input sequences (default: 24)
        output_sequence_length (int): Length of output sequences (default: 1)
        batch_size (int): Batch size for dataloaders (default: 32)

    Returns:
        Dict[str, Any]: Dictionary containing all prepared data components
    """
    # Step 1: Split into features (X) and target (y)
    X = data[features].copy()
    y = data[target].copy()

    print(f"Data shape: {data.shape}")
    print(f"Features: {features}")
    print(f"Target: {target}")
    print(f"X shape: {X.shape}, y shape: {y.shape}")

    # Step 2: Split data into train/validation/test
    X_train, X_val, X_test, y_train, y_val, y_test, train_timestamps, val_timestamps, test_timestamps = split_data(
        X, y, train_ratio, validation_ratio
    )

    print(f"\nData splits:")
    print(f"Train: X {X_train.shape}, y {y_train.shape}")
    print(f"Validation: X {X_val.shape}, y {y_val.shape}")
    print(f"Test: X {X_test.shape}, y {y_test.shape}")

    # Step 3: Scale the data
    X_train_scaled, X_val_scaled, X_test_scaled, y_train_scaled, y_val_scaled, y_test_scaled, X_scaler, y_scaler = scaling_data(
        X_train, X_val, X_test, y_train, y_val, y_test
    )

    print(f"\nScaled data shapes:")
    print(f"X_train_scaled: {X_train_scaled.shape}, y_train_scaled: {y_train_scaled.shape}")

    # Step 4: Create sequences
    X_train_seq, y_train_seq, train_input_timestamps, train_output_timestamps = create_sequences(
        X_train_scaled, y_train_scaled, train_timestamps, input_sequence_length, output_sequence_length
    )
    X_val_seq, y_val_seq, val_input_timestamps, val_output_timestamps = create_sequences(
        X_val_scaled, y_val_scaled, val_timestamps, input_sequence_length, output_sequence_length
    )
    X_test_seq, y_test_seq, test_input_timestamps, test_output_timestamps = create_sequences(
        X_test_scaled, y_test_scaled, test_timestamps, input_sequence_length, output_sequence_length
    )

    print(f"\nSequence shapes:")
    print(f"X_train_seq: {X_train_seq.shape}, y_train_seq: {y_train_seq.shape}")
    print(f"X_val_seq: {X_val_seq.shape}, y_val_seq: {y_val_seq.shape}")
    print(f"X_test_seq: {X_test_seq.shape}, y_test_seq: {y_test_seq.shape}")

    # Step 5: Create datasets
    train_dataset, validation_dataset, test_dataset = create_datasets(
        X_train_seq, y_train_seq, X_val_seq, y_val_seq, X_test_seq, y_test_seq
    )

    print(f"\nDatasets created:")
    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Validation dataset: {len(validation_dataset)} samples")
    print(f"Test dataset: {len(test_dataset)} samples")

    # Step 6: Create dataloaders
    train_dataloader, validation_dataloader, test_dataloader = create_dataloaders(
        train_dataset, validation_dataset, test_dataset, batch_size
    )

    print(f"\nDataloaders created with batch size: {batch_size}")
    print(f"Train batches: {len(train_dataloader)}")
    print(f"Validation batches: {len(validation_dataloader)}")
    print(f"Test batches: {len(test_dataloader)}")

    # Return comprehensive dictionary with all components
    return {
        # Original data
        'X': X,
        'y': y,

        # Split data
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,

        # Timestamps for splits
        'train_timestamps': train_timestamps,
        'val_timestamps': val_timestamps,
        'test_timestamps': test_timestamps,

        # Scaled data
        'X_train_scaled': X_train_scaled,
        'X_val_scaled': X_val_scaled,
        'X_test_scaled': X_test_scaled,
        'y_train_scaled': y_train_scaled,
        'y_val_scaled': y_val_scaled,
        'y_test_scaled': y_test_scaled,

        # Scalers
        'X_scaler': X_scaler,
        'y_scaler': y_scaler,

        # Sequences
        'X_train_seq': X_train_seq,
        'y_train_seq': y_train_seq,
        'X_val_seq': X_val_seq,
        'y_val_seq': y_val_seq,
        'X_test_seq': X_test_seq,
        'y_test_seq': y_test_seq,

        # Sequence timestamps
        'train_input_timestamps': train_input_timestamps,
        'train_output_timestamps': train_output_timestamps,
        'val_input_timestamps': val_input_timestamps,
        'val_output_timestamps': val_output_timestamps,
        'test_input_timestamps': test_input_timestamps,
        'test_output_timestamps': test_output_timestamps,

        # Datasets
        'train_dataset': train_dataset,
        'validation_dataset': validation_dataset,
        'test_dataset': test_dataset,

        # DataLoaders
        'train_dataloader': train_dataloader,
        'validation_dataloader': validation_dataloader,
        'test_dataloader': test_dataloader,

        # Metadata
        'input_sequence_length': input_sequence_length,
        'output_sequence_length': output_sequence_length,
        'batch_size': batch_size,
        'num_features': len(features),
        'feature_names': features,
        'target_name': target,

        # Utility function for future predictions
        'get_prediction_timestamps': lambda: get_prediction_timestamps(
            data.index, input_sequence_length, output_sequence_length
        )
    }


