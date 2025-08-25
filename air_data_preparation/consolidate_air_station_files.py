import os
import pandas as pd
from datetime import datetime

def standardize_columns(df):
    """Standardize column names across all stations"""
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

data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'air', 'data Estaciones CHSS')
output_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'air')
stations = ['APLAN', 'MHH', 'PFM', 'PGB', 'PLIB', 'USAM', 'UTEC']

print("Starting consolidation of air station files...")
print(f"Source directory: {data_dir}")
print(f"Output directory: {output_dir}")

for station in stations:
    print(f"\nProcessing station: {station}")

    # Find the CSV and XLSX files for the station
    files = [f for f in os.listdir(data_dir) if f.startswith(station) and (f.endswith('.csv') or f.endswith('.xlsx'))]
    csv_file = next((f for f in files if f.endswith('.csv')), None)
    xlsx_file = next((f for f in files if f.endswith('.xlsx')), None)

    if not csv_file or not xlsx_file:
        print(f"Warning: Expected both CSV and XLSX files for {station}, found: {files}")
        continue

    print(f"  Found files: {csv_file}, {xlsx_file}")

    dfs = []

    # Read CSV
    try:
        df_csv = pd.read_csv(os.path.join(data_dir, csv_file), sep=';')
        df_csv = standardize_columns(df_csv)
        dfs.append(df_csv)
        print(f"  CSV loaded: {len(df_csv)} rows, columns: {list(df_csv.columns)}")
    except Exception as e:
        print(f"Error reading CSV for {station}: {e}")

    # Read XLSX
    try:
        df_xlsx = pd.read_excel(os.path.join(data_dir, xlsx_file))
        df_xlsx = standardize_columns(df_xlsx)
        dfs.append(df_xlsx)
        print(f"  XLSX loaded: {len(df_xlsx)} rows, columns: {list(df_xlsx.columns)}")
    except Exception as e:
        print(f"Error reading XLSX for {station}: {e}")

    # Concatenate and process
    if not dfs:
        print(f"  No data loaded for {station}")
        continue

    df_all = pd.concat(dfs, ignore_index=True)
    print(f"  Combined data: {len(df_all)} rows")

    # Parse Timestamp to UTC
    if 'Timestamp' in df_all.columns:
        try:
            # Try different timestamp formats
            if df_all['Timestamp'].dtype == 'object':
                # Try parsing different formats
                try:
                    df_all['Timestamp'] = pd.to_datetime(df_all['Timestamp'], errors='coerce', format='%b %d %Y %H:%M')
                except:
                    try:
                        df_all['Timestamp'] = pd.to_datetime(df_all['Timestamp'], errors='coerce')
                    except:
                        print(f"  Warning: Could not parse timestamps for {station}")

            # Convert to UTC if not already timezone-aware
            if df_all['Timestamp'].dt.tz is None:
                df_all['Timestamp'] = df_all['Timestamp'].dt.tz_localize('UTC')
            else:
                df_all['Timestamp'] = df_all['Timestamp'].dt.tz_convert('UTC')

        except Exception as e:
            print(f"Error parsing Timestamp for {station}: {e}")

    # Remove duplicate timestamps if any
    if 'Timestamp' in df_all.columns:
        initial_count = len(df_all)
        df_all = df_all.drop_duplicates(subset=['Timestamp'], keep='first')
        if len(df_all) < initial_count:
            print(f"  Removed {initial_count - len(df_all)} duplicate timestamps")

        # Sort by Timestamp
        df_all = df_all.sort_values('Timestamp')
        print(f"  Date range: {df_all['Timestamp'].min()} to {df_all['Timestamp'].max()}")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save to {station}.csv in the main air data directory
    out_path = os.path.join(output_dir, f"{station}.csv")
    df_all.to_csv(out_path, index=False, sep=';')

    print(f"  ✓ Saved consolidated file: {out_path}")
    print(f"  Final columns: {list(df_all.columns)}")

print("\n✓ Consolidation complete!")
