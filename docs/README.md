# Chicago Crimes Analysis - Documentation Index

Welcome to the comprehensive documentation for the Chicago Crimes Analysis project. This project implements advanced data engineering and machine learning techniques for crime prediction and spatial analysis.

## 📚 Documentation Structure

### 🏗️ Architecture & Design
- **[Medallion Architecture](MEDALLION_ARCHITECTURE.md)** - Complete guide to Bronze → Silver → Gold data pipeline
- **[Project Overview](../README.md)** - Main project documentation and getting started guide

### 🔬 Advanced Techniques
- **[Cyclical Feature Encoding](CYCLICAL_FEATURES.md)** - Sin/Cos encoding for temporal patterns in ML models
- **[H3 Hexagonal Binning](H3_HEXAGONAL_BINNING.md)** - Uber's spatial indexing for uniform geographic analysis

### 💾 Data Layers

#### Bronze Layer
- **Purpose**: Raw data standardization and validation
- **Input**: CSV files from Chicago Open Data Portal
- **Output**: Standardized Parquet files with year partitioning
- **Implementation**: `src/data/bronze_processor.py`

#### L1 Layer (current)
- **Purpose**: Standardize raw CSVs to partitioned Parquet
- **Input**: `data/raw/chicago_crimes_YYYY.csv`
- **Output**: `data/l1/year=YYYY/month=MM/*.parquet`
- **Run**: `python src/l1_build.py 2018`
- **Docs**: [L1 – Standardized Clean Layer](L1.md)

#### Silver Layer  
- **Purpose**: Advanced feature engineering for ML
- **Key Features**: H3 spatial binning, cyclical time encoding, 60+ features
- **Output**: ML-ready dataset with spatial and temporal features
- **Implementation**: `src/data/preprocessing.py`

#### L2 Layer (current)
- **Purpose**: Feature engineering (H3, street, temporal, cyclical)
- **Input**: `data/l1/*`
- **Output**: `data/l2/*`
- **Run**: `python src/l2_features.py 9`
- **Docs**: [L2 – Feature Engineering Layer](L2.md)

#### Gold Layer
- **Purpose**: Business-ready aggregated datasets
- **Output**: Monthly aggregations, time series datasets, KPIs
- **Implementation**: `src/gold/crime_monthly.py`

## 🚀 Quick Start Guide

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/Ntropy86/Chicago-Crimes.git
cd Chicago-Crimes

# Create conda environment
conda create -n chicago python=3.10
conda activate chicago

# Install dependencies
pip install -r requirements.txt
pip install h3 holidays geopy hdbscan
```

### 2. Data Pipeline Execution
```bash
# Step 1: Download raw data
python -m src.data.download_data 2020

# Step 2: Create Bronze layer
python -m src.data.bronze_processor 2020 2024

# Step 3: Create Silver layer with H3 features
python -m src.data.preprocessing 2020 2024 9

# Step 4: Create Gold layer aggregations
python -m src.gold.crime_monthly
```

### 3. Key Features Created

**Spatial Features (H3)**:
- `h3_hex` - Primary hexagon (~174m)
- `h3_hex_l8` - Parent hexagon (~461m)
- `h3_hex_l7` - Grandparent hexagon (~1.22km)

**Temporal Features (Cyclical)**:
- `hour_sin/cos` - Hour of day cyclically encoded
- `day_of_week_sin/cos` - Day of week cyclically encoded
- `month_sin/cos` - Month cyclically encoded

**Crime Intelligence**:
- `crime_severity` - Numerical severity score (1-4)
- `is_violent_crime`, `is_property_crime` - Crime type flags
- `data_completeness_score` - Data quality metric

## 🧠 Key Concepts Explained

### Why Medallion Architecture?
- **Separation of Concerns**: Each layer has distinct responsibility
- **Data Quality**: Progressive refinement and validation
- **Scalability**: Can easily integrate with Spark/EMR
- **Maintainability**: Clear data lineage and debugging

### Why H3 Hexagonal Binning?
- **Uniform Analysis**: Eliminates administrative boundary bias
- **Better ML Features**: Consistent spatial inputs for algorithms
- **Multi-Resolution**: City-wide to block-level analysis
- **Visualization**: Beautiful hexagonal heatmaps

### Why Cyclical Feature Encoding?
- **Temporal Continuity**: 11 PM and 1 AM are neighbors, not 22 hours apart
- **Better ML Performance**: Neural networks learn time patterns correctly
- **Seasonal Patterns**: Captures cyclical crime trends across time boundaries

## 🎯 Use Cases

### Machine Learning Models
- **HDBScan Clustering**: H3 hexagons provide uniform spatial input
- **LSTM Forecasting**: Cyclical features improve temporal pattern learning
- **Spatial-Temporal Analysis**: Combined H3 + time features

### Business Intelligence
- **Crime Hotspot Mapping**: Hexagonal heatmaps show true spatial patterns
- **Resource Allocation**: Optimize police patrol routes using H3 grids
- **Trend Analysis**: Time series with proper cyclical handling

### Interactive Dashboards
- **Multi-Scale Visualization**: Zoom from city-wide to block-level
- **Time Animation**: Crime patterns over time with cyclical continuity
- **Real-Time Updates**: H3 grid enables efficient spatial queries

## 📊 Data Quality & Validation

### Spatial Validation
- **Coordinate Bounds**: Chicago city limits (41.0-42.5°N, -88.5 to -87.0°W)
- **H3 Assignment Rate**: >99% successful hexagon assignments
- **Multi-Resolution Consistency**: Parent-child hexagon relationships

### Temporal Validation
- **Date Range Validation**: 2001-present data coverage
- **Cyclical Continuity**: Sin/Cos encoding preserves time boundaries
- **Holiday Detection**: US/Illinois holidays properly flagged

### Feature Quality
- **Completeness Scoring**: 0.0-1.0 data quality metric per record
- **Missing Data Handling**: Proper null value treatment
- **Feature Correlation**: Cyclical features maintain temporal relationships

## 🔧 Performance Optimization

### Storage Efficiency
- **Parquet Format**: 70% size reduction vs CSV
- **Column Storage**: Optimized for analytical queries
- **Partitioning**: Year-based partitioning for time-range queries

### Processing Speed
- **Vectorized Operations**: Pandas/NumPy for H3 calculations
- **Efficient Joins**: H3 indexing enables fast spatial queries
- **Memory Management**: Chunked processing for large datasets

### Scalability
- **Modular Design**: Easy Spark/EMR integration
- **H3 Global System**: Scales to any geographic region
- **Feature Pipeline**: Reusable for other cities/datasets

## 🚀 Next Steps

### Machine Learning Implementation
1. **HDBScan Spatial Clustering** - Identify crime hotspots using H3 features
2. **ARIMA+LSTM Hybrid Model** - Time series forecasting with cyclical features
3. **Interactive Dashboard** - Plotly visualizations with H3 hexagon maps

### Advanced Analytics
1. **Spatial Autocorrelation** - Measure spatial clustering patterns
2. **Anomaly Detection** - Identify unusual crime patterns
3. **Predictive Policing** - Resource allocation optimization

### Deployment
1. **AWS EMR Integration** - Scalable cloud processing
2. **Real-Time Pipeline** - Stream processing for live data
3. **API Development** - RESTful API for model serving

## 📝 Contributing

When contributing to this project:

1. **Follow Architecture**: Maintain Bronze → Silver → Gold separation
2. **Document Features**: Update docs for new feature engineering
3. **Test Cyclical Encoding**: Validate temporal feature continuity
4. **Validate H3 Features**: Ensure proper spatial indexing

## 📞 Support

For questions about:
- **Medallion Architecture**: See [MEDALLION_ARCHITECTURE.md](MEDALLION_ARCHITECTURE.md)
- **Cyclical Features**: See [CYCLICAL_FEATURES.md](CYCLICAL_FEATURES.md)  
- **H3 Binning**: See [H3_HEXAGONAL_BINNING.md](H3_HEXAGONAL_BINNING.md)
- **General Issues**: Check the main [README.md](../README.md)

This documentation provides the foundation for understanding and extending the Chicago Crimes analysis project with advanced spatial and temporal analytics.