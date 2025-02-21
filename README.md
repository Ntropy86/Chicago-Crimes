# Chicago Crime Data Forecasting & Hotspot Analysis

## Project Overview
This graduate-level Data Science project focuses on developing an advanced crime analysis and prediction system for Chicago using hybrid modeling approaches and machine learning techniques. The project combines time series forecasting with spatial clustering to provide actionable insights for law enforcement resource allocation.

## Objectives
1. Develop a hybrid ARIMA+LSTM model for accurate crime incident forecasting
2. Implement spatial clustering using DBSCAN to identify high-risk zones
3. Create interactive visualizations for crime hotspot analysis
4. Deploy the solution using AWS EMR for scalable processing
5. Provide actionable insights for law enforcement resource allocation

## Technical Architecture
- **Data Processing**: PySpark for large-scale data processing
- **Time Series Forecasting**: Hybrid ARIMA+LSTM model
- **Spatial Analysis**: DBSCAN clustering
- **Cloud Infrastructure**: AWS EMR
- **Visualization**: Plotly
- **Version Control**: Git/GitHub

## Data Sources
- Chicago Police Department's CLEAR (Citizen Law Enforcement Analysis and Reporting) system
- Chicago Data Portal (https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2)
- Census demographic data for contextual analysis

## Methodology
1. **Data Collection and Preprocessing**
   - Historical crime data extraction
   - Data cleaning and feature engineering
   - Temporal and spatial feature creation

2. **Time Series Analysis**
   - ARIMA modeling for linear patterns
   - LSTM implementation for non-linear patterns
   - Hybrid model development and optimization

3. **Spatial Analysis**
   - DBSCAN clustering for hotspot detection
   - Geographic boundary consideration
   - Temporal-spatial correlation analysis

4. **Visualization and Reporting**
   - Interactive dashboards using Plotly
   - Temporal trend analysis
   - Spatial distribution mapping

## Expected Outcomes
- Monthly crime incident forecasts with 85%+ accuracy
- Identified high-risk zones with temporal patterns
- Interactive visualization dashboard
- Automated reporting system
- 30% reduction in manual analysis time

## Project Timeline
- Week 1-2: Data collection and preprocessing
- Week 3-4: ARIMA+LSTM model development
- Week 5-6: DBSCAN implementation and spatial analysis
- Week 7-8: AWS EMR setup and deployment
- Week 9-10: Dashboard development and testing
- Week 11-12: Documentation and final presentation

## Required Technologies
- Python 3.8+
- PySpark
- TensorFlow/Keras
- Scikit-learn
- Plotly
- AWS EMR
- Git

## Installation and Setup

### Prerequisites
```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### AWS EMR Setup
1. Configure AWS CLI
2. Set up EMR cluster
3. Configure security groups and IAM roles

## Project Structure
```
chicago-crime-analysis/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_time_series_analysis.ipynb
│   └── 03_spatial_analysis.ipynb
├── src/
│   ├── data/
│   ├── models/
│   └── visualization/
├── tests/
├── requirements.txt
├── README.md
└── .gitignore
```

## Getting Started
1. Clone the repository
```bash
git clone https://github.com/[username]/chicago-crime-analysis.git
cd chicago-crime-analysis
```

2. Set up the environment
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. Download the dataset
```bash
python src/data/download_data.py
```

## Contributing
Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.

## License
This project is licensed under the MIT License - see the LICENSE.md file for details

## Contact
[Your Name] - [Your Email]
Project Link: https://github.com/[username]/chicago-crime-analysis