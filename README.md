# AQ Edge - Air Quality Monitoring and Analysis Platform

A Python-based system for processing and analyzing air quality data from multiple environmental monitoring stations.

## Overview

AQ Edge consolidates, analyzes, and visualizes air quality data with automated processing, statistical analysis, and comprehensive reporting capabilities.

## Features

### 🔄 Data Processing
- Multi-format support (CSV/XLSX)
- Automated data consolidation and standardization
- UTC timestamp conversion with duplicate removal

### 📊 Analysis Capabilities
- Time series and statistical analysis
- Correlation analysis (within/cross-station)
- Outlier detection (IQR, Z-score methods)
- Data quality assessment

### 📈 Visualization
- Time series plots and distribution analysis
- Correlation heatmaps
- Missing data visualization

### 📄 Reporting
- Comprehensive PDF reports
- Executive summaries
- Timestamped logs with performance metrics

## Project Structure

```
aq_edge/
├── datautils/                 # Data processing modules
│   ├── consolidate_air_station_files.py
│   └── air_quality_analysis.py
├── utils/
│   └── logging.py            # Custom logging
└── __init__.py

artifacts/                    # Reports and logs
├── data_report.pdf
├── executive_summary.pdf
└── *.log

data/
└── air/
    ├── *.csv               # Consolidated data
    └── data Estaciones CHSS/  # Raw files
```

## Supported Monitoring Stations

7 air quality monitoring stations: **APLAN**, **MHH**, **PFM**, **PGB**, **PLIB**, **USAM**, **UTEC**

## Monitored Variables

- **Temperature** (°C) | **Humidity** (%) | **CO2** (ppm) | **VOC** (ppb) | **ICA** (Air Quality Index)

## Installation

### Prerequisites
- Python 3.8+

### Dependencies
```bash
pip install pandas numpy matplotlib seaborn scipy openpyxl
```

## Usage

### 1. Data Consolidation
```python
from aq_edge.datautils.consolidate_air_station_files import main
main()
```

### 2. Comprehensive Analysis
```python
from aq_edge.datautils.air_quality_analysis import main
main()
```

### 3. Individual Functions
```python
from aq_edge.datautils.air_quality_analysis import load_station_data, analyze_correlations

station_data, combined_df = load_station_data()
analyze_correlations(station_data)
```

## Data Format Requirements

### Input Data Structure
- **CSV**: Semicolon-separated (`;`)
- **XLSX**: Standard Excel format
- **Required**: `Timestamp` column
- **Optional**: `Temp.`, `Hum.`, `CO2`, `VOC`, `ICA`

### File Naming Convention
```
{STATION_NAME}_*.csv
{STATION_NAME}_*.xlsx
```

## Output Files

### Generated Reports
1. **Comprehensive PDF Report** - Complete analysis with visualizations
2. **Executive Summary** - 2-page overview with key findings

### Log Files
- Format: `{module_name}_YYYYMMDD_HHMMSS.log`
- Location: `artifacts/` directory

## Logging Features

- Structured logging with timestamps
- Performance tracking for each phase
- Module-specific logs with error handling
- File and console output

## API Reference

### Core Functions
```python
# Data consolidation
consolidate_station_data(station: str, data_dir: str, output_dir: str) -> bool

# Data loading
load_station_data(stations: List[str], data_dir: str) -> Tuple[Dict, pd.DataFrame]

# Analysis
analyze_correlations(station_data: Dict[str, pd.DataFrame]) -> None
plot_time_series(station_data: Dict[str, pd.DataFrame]) -> None

# Reports
generate_pdf_report(station_data, combined_df, filename: str) -> None
```

## Configuration

### Default Directories
- **Input**: `data/air/data Estaciones CHSS/`
- **Output**: `data/air/`
- **Reports/Logs**: `artifacts/`

## Performance Metrics

Tracks execution times, success rates, and provides detailed performance breakdown for each analysis phase.

## Data Quality Features

### Automated Quality Checks
- Missing data detection and visualization
- Multi-method outlier identification
- Duplicate timestamp detection
- Data completeness metrics

## Troubleshooting

### Common Issues
1. **File Not Found**: Check naming conventions
2. **Import Errors**: Verify dependencies installed
3. **Memory Issues**: Process stations individually
4. **Timestamp Parsing**: Verify datetime formats

## Contributing

1. Fork repository
2. Create feature branch
3. Add tests and documentation
4. Submit pull request

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contact

Refer to project repository for questions and contributions.

## Changelog

- **v1.0.0**: Initial release
- **v1.1.0**: PDF reporting
- **v1.2.0**: Enhanced logging
- **v1.3.0**: Executive summaries

---

*Last Updated: August 26, 2025*
