#!/usr/bin/env python3
"""
Air Quality Data Analysis and Forecasting Script
Comprehensive analysis of air quality stations: APLAN, MHH, PFM, PGB, PLIB, USAM, UTEC
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
from typing import Dict
from datetime import datetime
import traceback

# Import modules
from aq_edge.utils.logging import LoggerHandler
from aq_edge.datautils.reporting import generate_pdf_report, generate_executive_summary_pdf
from aq_edge.datautils.preprocessing import load_station_data

warnings.filterwarnings('ignore')

# Initialize logger
logger = LoggerHandler('air_quality_analysis')

def setup_plotting() -> None:
    """
    Configure matplotlib and seaborn for better visualizations.
    No parameters. No return value. Sets global plotting style.
    """
    # Fix deprecated seaborn style
    plt.style.use('default')  # Use default instead of deprecated seaborn-v0_8
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 8)
    logger.info("[OK] Plotting configuration complete")


def show_data_overview(combined_df: pd.DataFrame) -> None:
    """
    Display basic overview of the combined air quality DataFrame.

    Args:
        combined_df (pd.DataFrame): Combined DataFrame of all stations.
    """
    logger.info("\n" + "="*60)
    logger.info("DATA OVERVIEW")
    logger.info("="*60)

    logger.info("\nFirst few rows:")
    logger.info(combined_df.head())

    logger.info("\nData types:")
    logger.info(combined_df.dtypes)

    logger.info("\nBasic info:")
    combined_df.info()
    
    logger.info("\nBasic statistics:")
    logger.info(combined_df.describe())

def calculate_station_statistics(station_data: Dict[str, pd.DataFrame]) -> None:
    """
    Calculate and print summary statistics for each station's data.

    Args:
        station_data (Dict[str, pd.DataFrame]): Dictionary of station DataFrames.
    """
    logger.info("\n" + "="*60)
    logger.info("SUMMARY STATISTICS PER STATION")
    logger.info("="*60)

    variables = ['Temp.', 'Hum.', 'CO2', 'VOC', 'ICA']
    
    for station, df in station_data.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"STATION: {station}")
        logger.info(f"{'='*50}")

        logger.info(f"Total records: {len(df)}")
        logger.info(f"Date range: {df['Timestamp'].min()} to {df['Timestamp'].max()}")

        for var in variables:
            if var in df.columns:
                logger.info(f"\n--- {var} ---")
                logger.info(f"Missing values: {df[var].isna().sum()} ({df[var].isna().sum()/len(df)*100:.2f}%)")
                logger.info(f"Mean: {df[var].mean():.2f}")
                logger.info(f"Std: {df[var].std():.2f}")
                logger.info(f"Min: {df[var].min():.2f}")
                logger.info(f"Max: {df[var].max():.2f}")

                # Outliers using IQR method
                Q1 = df[var].quantile(0.25)
                Q3 = df[var].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = ((df[var] < lower_bound) | (df[var] > upper_bound)).sum()
                logger.info(f"Outliers (IQR method): {outliers} ({outliers/len(df)*100:.2f}%)")
                logger.info(f"Q1: {Q1:.2f}, Q3: {Q3:.2f}")

def plot_time_series(station_data: Dict[str, pd.DataFrame]) -> None:
    """
    Generate time series plots for each variable and station.

    Args:
        station_data (Dict[str, pd.DataFrame): Dictionary of station DataFrames.
    """
    logger.info("\n" + "="*60)
    logger.info("GENERATING TIME SERIES PLOTS")
    logger.info("="*60)

    # Check if we have any data
    if not station_data:
        logger.error("No station data available for plotting")
        return

    # Updated variable mapping to handle different column names
    variable_mapping = {
        'Temp.': ['Temp.', 'Temperature'],
        'Hum.': ['Hum.', 'Humidity'],
        'CO2': ['CO2', 'CO2 (ppm)', 'CO2_ppm'],
        'VOC': ['VOC', 'VOC (ppb)', 'VOC_ppb'],
        'ICA': ['ICA']
    }

    stations = list(station_data.keys())
    
    for var_name, var_columns in variable_mapping.items():
        # Create subplots - 2 rows, 4 columns to accommodate 7 stations
        fig, axes = plt.subplots(2, 4, figsize=(20, 12))
        axes = axes.flatten()

        station_count = 0
        for i, station in enumerate(stations):
            df = station_data[station]

            # Find which column exists for this variable
            actual_column = None
            for col_name in var_columns:
                if col_name in df.columns:
                    actual_column = col_name
                    break

            if actual_column is not None and not df[actual_column].dropna().empty:
                # Plot time series for this station
                axes[station_count].plot(df['Timestamp'], df[actual_column], alpha=0.7, linewidth=1.5,
                           color=f'C{station_count}', label=station)

                axes[station_count].set_title(f'{station} - {var_name}', fontsize=12, fontweight='bold')
                axes[station_count].set_xlabel('Time', fontsize=10)
                axes[station_count].set_ylabel(var_name, fontsize=10)
                axes[station_count].grid(True, alpha=0.3)
                axes[station_count].tick_params(axis='x', rotation=45)

                # Add basic statistics as text - check for valid data
                valid_data = df[actual_column].dropna()
                if len(valid_data) > 0:
                    mean_val = valid_data.mean()
                    std_val = valid_data.std()
                    axes[station_count].text(0.02, 0.98, f'Mean: {mean_val:.1f}\nStd: {std_val:.1f}',
                               transform=axes[station_count].transAxes, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                               fontsize=9)

                station_count += 1
            else:
                logger.warning(f"{station} does not have {var_name} data or data is empty")

        # Remove empty subplots
        for i in range(station_count, len(axes)):
            fig.delaxes(axes[i])

        if station_count > 0:
            plt.suptitle(f'{var_name} Time Series - All Stations ({station_count} stations)', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.show()
            logger.info(f"[OK] Generated time series subplots for {var_name} ({station_count} stations)")
        else:
            plt.close(fig)
            logger.warning(f"No data available for {var_name} - skipping plot")

    # Also create an overview plot with all variables for one station
    logger.info("\nGenerating overview plots per station...")

    for station in stations:
        df = station_data[station]
        available_vars = []
        actual_columns = []

        # Find available variables for this station
        for var_name, var_columns in variable_mapping.items():
            for col_name in var_columns:
                if col_name in df.columns and not df[col_name].dropna().empty:
                    available_vars.append(var_name)
                    actual_columns.append(col_name)
                    break

        if len(available_vars) > 0:
            fig, axes = plt.subplots(len(available_vars), 1, figsize=(15, 3*len(available_vars)))

            # Handle case where there's only one variable
            if len(available_vars) == 1:
                axes = [axes]

            for i, (var_name, actual_col) in enumerate(zip(available_vars, actual_columns)):
                axes[i].plot(df['Timestamp'], df[actual_col], alpha=0.7, linewidth=1.5,
                           color=f'C{i}')
                axes[i].set_title(f'{var_name} ({actual_col})', fontsize=12, fontweight='bold')
                axes[i].set_ylabel(var_name, fontsize=10)
                axes[i].grid(True, alpha=0.3)
                axes[i].tick_params(axis='x', rotation=45)

                # Add statistics - ensure we have valid data
                valid_data = df[actual_col].dropna()
                if len(valid_data) > 0:
                    mean_val = valid_data.mean()
                    std_val = valid_data.std()
                    min_val = valid_data.min()
                    max_val = valid_data.max()
                    axes[i].text(0.02, 0.98,
                               f'Mean: {mean_val:.1f} | Std: {std_val:.1f}\nMin: {min_val:.1f} | Max: {max_val:.1f}',
                               transform=axes[i].transAxes, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                               fontsize=9)

            # Only add xlabel to the bottom plot
            axes[-1].set_xlabel('Time', fontsize=12)

            plt.suptitle(f'{station} - All Variables Overview', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.show()

            logger.info(f"[OK] Generated overview plot for {station}")
        else:
            logger.warning(f"{station} has no recognizable variables with valid data")

def plot_distributions(station_data: Dict[str, pd.DataFrame]) -> None:
    """
    Create box plots and distribution histograms for each variable across stations.

    Args:
        station_data (Dict[str, pd.DataFrame]): Dictionary of station DataFrames.
    """
    logger.info("\n" + "="*60)
    logger.info("GENERATING DISTRIBUTION PLOTS")
    logger.info("="*60)

    variables = ['Temp.', 'Hum.', 'CO2', 'VOC', 'ICA']
    stations = list(station_data.keys())
    
    # Box plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, var in enumerate(variables):
        data_for_plot = []
        labels_for_plot = []
        
        for station in stations:
            if var in station_data[station].columns:
                data_for_plot.append(station_data[station][var].dropna())
                labels_for_plot.append(station)
        
        if data_for_plot:
            axes[i].boxplot(data_for_plot, labels=labels_for_plot)
            axes[i].set_title(f'{var} Distribution Across Stations', fontweight='bold')
            axes[i].set_ylabel(var)
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].grid(True, alpha=0.3)
    
    # Remove empty subplot
    axes[-1].remove()
    plt.tight_layout()
    plt.show()
    logger.info("[OK] Generated box plots")

    # Distribution histograms
    for var in variables:
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for i, station in enumerate(stations):
            if var in station_data[station].columns:
                data = station_data[station][var].dropna()
                
                # Histogram with KDE
                axes[i].hist(data, bins=30, alpha=0.7, density=True, color=f'C{i}')
                axes[i].axvline(data.mean(), color='red', linestyle='--', 
                               label=f'Mean: {data.mean():.2f}')
                axes[i].axvline(data.median(), color='orange', linestyle='--', 
                               label=f'Median: {data.median():.2f}')
                
                # Add KDE
                try:
                    from scipy.stats import gaussian_kde
                    kde = gaussian_kde(data)
                    x_range = np.linspace(data.min(), data.max(), 100)
                    axes[i].plot(x_range, kde(x_range), 'k-', alpha=0.5)
                except:
                    pass
                
                axes[i].set_title(f'{station} - {var}')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
        
        # Remove empty subplot
        axes[-1].remove()
        
        plt.suptitle(f'{var} Distribution by Station', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        logger.info(f"[OK] Generated distribution plots for {var}")

def analyze_correlations(station_data: Dict[str, pd.DataFrame]) -> None:
    """
    Analyze and visualize Pearson correlations within and across stations.

    Args:
        station_data (Dict[str, pd.DataFrame]): Dictionary of station DataFrames.
    """
    logger.info("\n" + "="*60)
    logger.info("CORRELATION ANALYSIS - SEPARATED PLOTS")
    logger.info("="*60)

    # Check if we have any data
    if not station_data:
        logger.error("No station data available for correlation analysis")
        return

    # Updated variable mapping to handle different column names
    variable_mapping = {
        'Temp.': ['Temp.', 'Temperature'],
        'Hum.': ['Hum.', 'Humidity'],
        'CO2': ['CO2', 'CO2 (ppm)', 'CO2_ppm'],
        'VOC': ['VOC', 'VOC (ppb)', 'VOC_ppb'],
        'ICA': ['ICA']
    }

    stations = list(station_data.keys())
    
    # Within-station correlations - SEPARATED PLOTS
    logger.info("Generating individual correlation matrices for each station...")

    for station in stations:
        df = station_data[station]
        
        # Find available variables for this station using mapping
        available_vars = []
        actual_columns = []
        for var_name, var_columns in variable_mapping.items():
            for col_name in var_columns:
                if col_name in df.columns and not df[col_name].dropna().empty:
                    available_vars.append(var_name)
                    actual_columns.append(col_name)
                    break

        if len(available_vars) > 1:  # Need at least 2 variables for correlation
            # Create correlation matrix using actual column names
            correlation_data = df[actual_columns].corr()

            # Check if correlation matrix has valid values
            if correlation_data.isna().all().all():
                logger.warning(f"{station} has no valid correlations")
                continue

            # Rename columns to standardized variable names for display
            correlation_data.columns = available_vars
            correlation_data.index = available_vars

            # Create individual plot for this station
            plt.figure(figsize=(10, 8))

            # Create heatmap with seaborn for better styling
            mask = np.triu(np.ones_like(correlation_data, dtype=bool))
            sns.heatmap(correlation_data,
                       mask=mask,
                       annot=True,
                       cmap='RdBu_r',
                       center=0,
                       square=True,
                       fmt='.3f',
                       cbar_kws={"shrink": .8},
                       vmin=-1,
                       vmax=1)

            plt.title(f'{station} - Variable Correlation Matrix',
                     fontsize=16, fontweight='bold', pad=20)
            plt.tight_layout()
            plt.show()

            # Print correlation summary
            logger.info(f"\n{station} - Correlation Summary:")
            logger.info(f"Variables analyzed: {', '.join(available_vars)}")
            logger.info(f"Data points: {len(df):,}")
            logger.info("Strong correlations (|r| > 0.7):")

            strong_corr = []
            for i in range(len(available_vars)):
                for j in range(i+1, len(available_vars)):
                    corr_val = correlation_data.iloc[i, j]
                    if not np.isnan(corr_val) and abs(corr_val) > 0.7:
                        strong_corr.append(f"  {available_vars[i]} ↔ {available_vars[j]}: {corr_val:.3f}")

            if strong_corr:
                for corr in strong_corr:
                    logger.info(corr)
            else:
                logger.info("  No strong correlations found")

            logger.info("-" * 50)
        else:
            logger.warning(f"{station} has insufficient variables for correlation analysis")

    logger.info("\nGenerating cross-station correlations for each variable...")

    # Cross-station correlations - SEPARATED PLOTS FOR EACH VARIABLE
    for var_name, var_columns in variable_mapping.items():
        # Create a dataframe with each station as a column
        station_var_data = pd.DataFrame()
        
        for station in stations:
            df = station_data[station]
            # Find which column exists for this variable
            actual_column = None
            for col_name in var_columns:
                if col_name in df.columns:
                    actual_column = col_name
                    break

            if actual_column is not None and not df[actual_column].dropna().empty:
                # Create a copy of the data with timestamp as index
                df_temp = df[['Timestamp', actual_column]].copy()

                # Remove duplicates by keeping the first occurrence of each timestamp
                df_temp = df_temp.drop_duplicates(subset=['Timestamp'], keep='first')

                # Set timestamp as index
                df_station = df_temp.set_index('Timestamp')[actual_column]

                # Add to combined dataframe
                if station_var_data.empty:
                    station_var_data = pd.DataFrame({station: df_station})
                else:
                    station_var_data = station_var_data.join(df_station.rename(station), how='outer')

        if len(station_var_data.columns) > 1:
            # Drop rows where all values are NaN
            station_var_data = station_var_data.dropna(how='all')

            # Calculate correlation matrix (this will handle NaN values automatically)
            corr_matrix = station_var_data.corr()
            
            # Only show correlation if we have enough data and valid correlations
            if not corr_matrix.empty and corr_matrix.shape[0] > 1 and not corr_matrix.isna().all().all():
                # Create individual plot for this variable
                plt.figure(figsize=(12, 10))

                # Create heatmap
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                sns.heatmap(corr_matrix,
                           mask=mask,
                           annot=True,
                           cmap='RdBu_r',
                           center=0,
                           square=True,
                           fmt='.3f',
                           cbar_kws={"shrink": .8},
                           vmin=-1,
                           vmax=1)

                plt.title(f'{var_name} - Cross-Station Correlation Matrix',
                         fontsize=16, fontweight='bold', pad=20)
                plt.ylabel('Stations', fontsize=12)
                plt.xlabel('Stations', fontsize=12)
                plt.tight_layout()
                plt.show()

                # Print cross-station correlation summary
                logger.info(f"\n{var_name} - Cross-Station Correlation Summary:")
                logger.info(f"Data points used: {len(station_var_data):,} timestamps")
                logger.info(f"Stations with data: {', '.join(station_var_data.columns)}")

                # Find highest correlations
                logger.info("Highest correlations between stations:")
                correlations = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        if not np.isnan(corr_matrix.iloc[i, j]):
                            correlations.append((
                                corr_matrix.columns[i],
                                corr_matrix.columns[j],
                                corr_matrix.iloc[i, j]
                            ))

                # Sort by absolute correlation value
                correlations.sort(key=lambda x: abs(x[2]), reverse=True)

                # Show top 3 correlations
                for i, (station1, station2, corr_val) in enumerate(correlations[:3]):
                    logger.info(f"  {i+1}. {station1} ↔ {station2}: {corr_val:.3f}")

                logger.info("-" * 60)
            else:
                logger.warning(f"{var_name} has insufficient overlapping data for cross-correlation")
        else:
            logger.warning(f"{var_name} has insufficient stations for cross-correlation analysis")

    logger.info(f"[OK] Generated separated correlation matrices for all stations and variables")

def analyze_missing_data(station_data: Dict[str, pd.DataFrame]) -> None:
    """
    Analyze and visualize missing data patterns for all stations and variables.

    Args:
        station_data (Dict[str, pd.DataFrame]): Dictionary of station DataFrames.
    """
    logger.info("\n" + "="*60)
    logger.info("MISSING DATA ANALYSIS")
    logger.info("="*60)

    stations = list(station_data.keys())

    # Create missing data summary
    missing_summary = pd.DataFrame()

    for station in stations:
        df = station_data[station]
        missing_count = df.isnull().sum()
        missing_percent = (missing_count / len(df)) * 100

        station_missing = pd.DataFrame({
            f'{station}_count': missing_count,
            f'{station}_percent': missing_percent
        })

        if missing_summary.empty:
            missing_summary = station_missing
        else:
            missing_summary = missing_summary.join(station_missing, how='outer')

    logger.info("Missing data summary:")
    logger.info(missing_summary.round(2))

    # Visualize missing data patterns
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # Missing data heatmap
    missing_matrix = pd.DataFrame()
    for station in stations:
        df = station_data[station]
        for col in ['Temp.', 'Hum.', 'CO2', 'VOC', 'ICA']:
            if col in df.columns:
                missing_matrix[f'{station}_{col}'] = df[col].isnull()

    if not missing_matrix.empty:
        # Convert boolean matrix to numeric, handling NaN values properly
        # First fill NaN with False (meaning data is present), then convert to int
        missing_matrix_clean = missing_matrix.fillna(False)
        missing_matrix_numeric = missing_matrix_clean.astype(int)

        im = axes[0].imshow(missing_matrix_numeric.T, cmap='RdYlBu', aspect='auto')
        axes[0].set_title('Missing Data Pattern\n(Yellow = Missing, Blue = Present)')
        axes[0].set_xlabel('Time Index')
        axes[0].set_ylabel('Station_Variable')
        axes[0].set_yticks(range(len(missing_matrix.columns)))
        axes[0].set_yticklabels(missing_matrix.columns, fontsize=8)

        # Add colorbar
        plt.colorbar(im, ax=axes[0], label='Missing (1) vs Present (0)')

    # Missing data percentage by station-variable
    if not missing_matrix.empty:
        missing_pct = missing_matrix.mean() * 100
        missing_pct.plot(kind='bar', ax=axes[1])
        axes[1].set_title('Missing Data Percentage by Station-Variable')
        axes[1].set_ylabel('Missing Percentage (%)')
        axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()
    logger.info("[OK] Generated missing data visualizations")

def analyze_outliers(station_data: Dict[str, pd.DataFrame]) -> None:
    """
    Detect and summarize outliers in each variable for all stations using IQR and Z-score methods.

    Args:
        station_data (Dict[str, pd.DataFrame]): Dictionary of station DataFrames.
    """
    logger.info("\n" + "="*60)
    logger.info("OUTLIER ANALYSIS")
    logger.info("="*60)

    variables = ['Temp.', 'Hum.', 'CO2', 'VOC', 'ICA']
    stations = list(station_data.keys())
    
    for var in variables:
        logger.info(f"\n{'='*30} {var} {'='*30}")

        outlier_summary = pd.DataFrame()
        
        for station in stations:
            if var in station_data[station].columns:
                data = station_data[station][var].dropna()
                
                if len(data) > 0:
                    # IQR method
                    Q1, Q3 = data.quantile([0.25, 0.75])
                    IQR = Q3 - Q1
                    iqr_outliers = ((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).sum()
                    
                    # Z-score method (|z| > 3)
                    z_scores = np.abs(stats.zscore(data))
                    z_outliers = (z_scores > 3).sum()
                    
                    # Modified Z-score method (using median)
                    median = np.median(data)
                    mad = np.median(np.abs(data - median))
                    if mad > 0:
                        modified_z_scores = 0.6745 * (data - median) / mad
                        modified_z_outliers = (np.abs(modified_z_scores) > 3.5).sum()
                    else:
                        modified_z_outliers = 0
                    
                    outlier_summary[station] = [
                        len(data),
                        iqr_outliers,
                        f"{iqr_outliers/len(data)*100:.2f}%",
                        z_outliers,
                        f"{z_outliers/len(data)*100:.2f}%",
                        modified_z_outliers,
                        f"{modified_z_outliers/len(data)*100:.2f}%"
                    ]
        
        if not outlier_summary.empty:
            outlier_summary.index = ['Total Records', 'IQR Outliers', 'IQR %', 
                                   'Z-Score Outliers', 'Z-Score %', 
                                   'Modified Z Outliers', 'Modified Z %']
            logger.info(outlier_summary)

def generate_final_report(station_data: Dict[str, pd.DataFrame], combined_df: pd.DataFrame) -> None:
    """
    Print a comprehensive summary report of the air quality analysis.

    Args:
        station_data (Dict[str, pd.DataFrame]): Dictionary of station DataFrames.
        combined_df (pd.DataFrame): Combined DataFrame of all stations.
    """
    logger.info("\n" + "="*70)
    logger.info("COMPREHENSIVE DATA SUMMARY REPORT")
    logger.info("="*70)

    stations = list(station_data.keys())
    
    logger.info(f"Analysis generated on: {pd.Timestamp.now()}")
    logger.info(f"Stations analyzed: {', '.join(stations)}")

    total_records = sum(len(station_data[station]) for station in stations)
    logger.info(f"Total records across all stations: {total_records:,}")

    if station_data:
        date_range_start = min(station_data[station]['Timestamp'].min() for station in stations)
        date_range_end = max(station_data[station]['Timestamp'].max() for station in stations)
        logger.info(f"Date range: {date_range_start} to {date_range_end}")

    # Variable summary
    variables = ['Temp.', 'Hum.', 'CO2', 'VOC', 'ICA']
    logger.info(f"\nVariables analyzed: {', '.join(variables)}")

    # Station-wise record counts
    logger.info("\nRecord counts by station:")
    for station in stations:
        count = len(station_data[station])
        logger.info(f"  {station}: {count:,} records")

    logger.info("\n" + "="*70)
    logger.info("ANALYSIS COMPLETE")
    logger.info("Review the plots and statistics above for detailed insights.")
    logger.info("="*70)



def main() -> None:
    """
    Main execution function for air quality analysis workflow.
    Loads data, performs analysis, generates reports and visualizations.
    No parameters. No return value.
    """
    logger.info("="*70)
    logger.info("AIR QUALITY STATIONS COMPREHENSIVE ANALYSIS")
    logger.info("="*70)
    logger.info("Analyzing data from stations: APLAN, MHH, PFM, PGB, PLIB, USAM, UTEC")
    logger.info("="*70)

    now = datetime.now()
    formatted = now.strftime("%Y%m%d_%H%M%S")

    try:
        # Setup
        setup_plotting()

        # Load data
        station_data, combined_df = load_station_data(
            stations = ['APLAN', 'MHH', 'PFM', 'PGB', 'PLIB', 'USAM', 'UTEC'],
            data_dir= '../../data/air/'
        )

        # Check if we have any data before proceeding
        if not station_data or combined_df.empty:
            logger.error("No data loaded. Please check data files and paths.")
            return

        # Data overview
        show_data_overview(combined_df)

        # Statistical analysis
        calculate_station_statistics(station_data)

        # Visualizations
        plot_time_series(station_data)
        plot_distributions(station_data)

        # Correlation analysis
        analyze_correlations(station_data)

        # Data quality analysis
        analyze_missing_data(station_data)
        analyze_outliers(station_data)

        # Generate comprehensive PDF report
        generate_pdf_report(station_data, combined_df, filename=f'../../artifacts/data_report_{formatted}.pdf')

        # Generate executive summary PDF
        generate_executive_summary_pdf(station_data, combined_df,
                                       filename=f'../../artifacts/executive_summary_{formatted}.pdf'
                                       )

        # Generate final summary
        generate_final_report(station_data, combined_df)

    except Exception as e:
        logger.error(f"An error occurred during analysis: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
