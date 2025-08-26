#!/usr/bin/env python3
"""
Air Quality Station Data Consolidation Module

This module provides functionality to consolidate air quality monitoring data
from multiple file formats (CSV and XLSX) for various monitoring stations.

"""

import os
import pandas as pd
from typing import List, Tuple, Optional

# Import logger from utils
from aq_edge.utils.logging import LoggerHandler

# Initialize logger with the actual module name
logger = LoggerHandler('consolidate_air_station_files')


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names across all air quality monitoring stations.

    This function normalizes column names to ensure consistency across different
    data sources by renaming columns with variations in naming conventions to
    standard names.

    Args:
        df (pd.DataFrame): Input DataFrame with potentially non-standardized column names.
                          Expected to contain air quality measurement columns such as
                          CO2 and VOC data with varying naming formats.

    Returns:
        pd.DataFrame: A copy of the input DataFrame with standardized column names:
                     - CO2 columns: 'CO2 (ppm)' or 'CO2_ppm' → 'CO2'
                     - VOC columns: 'VOC (ppb)' or 'VOC_ppb' → 'VOC'
    """
    # Create a copy to avoid modifying the original
    df = df.copy()

    # Standardize CO2 column name
    if 'CO2 (ppm)' in df.columns:
        df = df.rename(columns={'CO2 (ppm)': 'CO2'})
    elif 'CO2_ppm' in df.columns:
        df = df.rename(columns={'CO2_ppm': 'CO2'})

    # Standardize VOC column name if needed
    if 'VOC (ppb)' in df.columns:
        df = df.rename(columns={'VOC (ppb)': 'VOC'})
    elif 'VOC_ppb' in df.columns:
        df = df.rename(columns={'VOC_ppb': 'VOC'})

    return df


def find_station_files(data_dir: str, station: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Find CSV and XLSX files for a given station in the data directory.

    Args:
        data_dir (str): Path to the directory containing station data files
        station (str): Station identifier/name to search for (e.g., 'APLAN', 'MHH')

    Returns:
        Tuple[Optional[str], Optional[str]]: A tuple containing (csv_filename, xlsx_filename).
                                           Returns None for either if file is not found.

    Raises:
        OSError: If the data directory cannot be accessed
    """
    try:
        files = [f for f in os.listdir(data_dir)
                if f.startswith(station) and (f.endswith('.csv') or f.endswith('.xlsx'))]

        csv_file = next((f for f in files if f.endswith('.csv')), None)
        xlsx_file = next((f for f in files if f.endswith('.xlsx')), None)

        return csv_file, xlsx_file
    except OSError as e:
        logger.error(f"Error accessing directory {data_dir}: {e}")
        return None, None


def load_station_data(data_dir: str, csv_file: str, xlsx_file: str, station: str) -> List[pd.DataFrame]:
    """
    Load and standardize data from CSV and XLSX files for a station.

    Args:
        data_dir (str): Path to the directory containing the data files
        csv_file (str): Name of the CSV file to load
        xlsx_file (str): Name of the XLSX file to load
        station (str): Station identifier for error reporting

    Returns:
        List[pd.DataFrame]: List of successfully loaded and standardized DataFrames.
                           May contain 0, 1, or 2 DataFrames depending on loading success.

    """
    dfs: List[pd.DataFrame] = []

    # Read CSV
    if csv_file:
        try:
            df_csv = pd.read_csv(os.path.join(data_dir, csv_file), sep=';')
            df_csv = standardize_columns(df_csv)
            dfs.append(df_csv)
            logger.info(f"CSV loaded: {len(df_csv)} rows, columns: {list(df_csv.columns)}")
        except Exception as e:
            logger.error(f"Error reading CSV for {station}: {e}")

    # Read XLSX
    if xlsx_file:
        try:
            df_xlsx = pd.read_excel(os.path.join(data_dir, xlsx_file))
            df_xlsx = standardize_columns(df_xlsx)
            dfs.append(df_xlsx)
            logger.info(f"XLSX loaded: {len(df_xlsx)} rows, columns: {list(df_xlsx.columns)}")
        except Exception as e:
            logger.error(f"Error reading XLSX for {station}: {e}")

    return dfs


def process_timestamps(df: pd.DataFrame, station: str) -> pd.DataFrame:
    """
    Parse and standardize timestamp columns in the DataFrame.

    This function attempts to parse timestamps using multiple formats and converts
    them to UTC timezone. It handles both timezone-aware and timezone-naive timestamps.

    Args:
        df (pd.DataFrame): Input DataFrame that may contain a 'Timestamp' column
        station (str): Station identifier for error reporting

    Returns:
        pd.DataFrame: DataFrame with processed timestamps in UTC timezone.
                     If no 'Timestamp' column exists or parsing fails,
                     returns the original DataFrame unchanged.

    """
    if 'Timestamp' not in df.columns:
        return df

    try:
        # Try different timestamp formats
        if df['Timestamp'].dtype == 'object':
            # Try parsing different formats
            try:
                df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce', format='%b %d %Y %H:%M')
            except:
                try:
                    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
                except:
                    logger.warning(f"Could not parse timestamps for {station}")
                    return df

        # Convert to UTC if not already timezone-aware
        if df['Timestamp'].dt.tz is None:
            df['Timestamp'] = df['Timestamp'].dt.tz_localize('UTC')
        else:
            df['Timestamp'] = df['Timestamp'].dt.tz_convert('UTC')
        logger.info("Converting Timestamp to UTC")

    except Exception as e:
        logger.error(f"Error parsing Timestamp for {station}: {e}")

    return df


def remove_duplicate_timestamps(df: pd.DataFrame, station: str) -> pd.DataFrame:
    """
    Remove duplicate timestamps and sort DataFrame by timestamp.

    Args:
        df (pd.DataFrame): Input DataFrame that may contain duplicate timestamps
        station (str): Station identifier for logging purposes

    Returns:
        pd.DataFrame: DataFrame with duplicates removed and sorted by timestamp.
                     If no 'Timestamp' column exists, returns original DataFrame.

    """
    if 'Timestamp' not in df.columns:
        return df

    initial_count = len(df)
    df = df.drop_duplicates(subset=['Timestamp'], keep='first')

    if len(df) < initial_count:
        logger.info(f"Removed {initial_count - len(df)} duplicate timestamps")

    # Sort by Timestamp
    df = df.sort_values('Timestamp')
    logger.info(f"Date range: {df['Timestamp'].min()} to {df['Timestamp'].max()}")

    return df


def consolidate_station_data(station: str, data_dir: str, output_dir: str) -> bool:
    """
    Consolidate CSV and XLSX data files for a single air quality monitoring station.

    This function processes all data for a given station by:
    1. Finding the corresponding CSV and XLSX files
    2. Loading and standardizing the data
    3. Processing timestamps and removing duplicates
    4. Saving the consolidated data to a single CSV file

    Args:
        station (str): Station identifier (e.g., 'APLAN', 'MHH', 'PFM')
        data_dir (str): Path to directory containing raw station data files
        output_dir (str): Path to directory where consolidated files will be saved

    Returns:
        bool: True if consolidation was successful, False otherwise

    """
    logger.info(f"Processing station: {station}")

    # Find the CSV and XLSX files for the station
    csv_file, xlsx_file = find_station_files(data_dir, station)

    if not csv_file or not xlsx_file:
        files = [f for f in os.listdir(data_dir) if f.startswith(station)]
        logger.warning(f"Expected both CSV and XLSX files for {station}, found: {files}")
        return False

    logger.info(f"Found files: {csv_file}, {xlsx_file}")

    # Load data from both files
    dfs = load_station_data(data_dir, csv_file, xlsx_file, station)

    if not dfs:
        logger.error(f"No data loaded for {station}")
        return False

    # Concatenate all data
    df_all = pd.concat(dfs, ignore_index=True)
    logger.info(f"Combined data: {len(df_all)} rows")

    # Process timestamps
    df_all = process_timestamps(df_all, station)
    df_all = remove_duplicate_timestamps(df_all, station)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save consolidated data
    out_path = os.path.join(output_dir, f"{station}.csv")
    df_all.to_csv(out_path, index=False, sep=';')

    logger.info(f"Saved consolidated file: {out_path}")
    logger.info(f"Final columns: {list(df_all.columns)}")

    return True


def main() -> None:
    """
    Main function to consolidate air quality data from multiple stations.

    This function processes data from all configured air quality monitoring stations,
    consolidating CSV and XLSX files for each station into single, standardized CSV files.

    The function:
    1. Sets up input and output directories
    2. Iterates through all configured stations
    3. Consolidates data for each station
    4. Reports overall success/failure statistics

    """

    data_dir = r'..\..\data\air\data Estaciones CHSS'#os.path.join(os.path.dirname(__file__), '..', 'data', 'air', 'data Estaciones CHSS')
    output_dir = r'..\..\data\air'#os.path.join(os.path.dirname(__file__), '..', 'data', 'air')
    stations: List[str] = ['APLAN', 'MHH', 'PFM', 'PGB', 'PLIB', 'USAM', 'UTEC']

    logger.info("Starting consolidation of air station files...")
    logger.info(f"Source directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")

    successful_stations = 0
    total_stations = len(stations)

    for station in stations:
        if consolidate_station_data(station, data_dir, output_dir):
            successful_stations += 1

    logger.info(f"Consolidation complete!")
    logger.info(f"Successfully processed {successful_stations}/{total_stations} stations")


if __name__ == "__main__":
    main()
