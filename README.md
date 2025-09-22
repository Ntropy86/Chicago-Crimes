# Chicago Crime Data Forecasting & Hotspot Analysis with AI Agent

## Project Overview
This graduate-level Data Science project focuses on developing an advanced crime analysis and prediction system for Chicago using hybrid modeling approaches, machine learning techniques, and an intelligent AI agent. The project combines time series forecasting with spatial clustering and implements an agentic approach that allows users to interact with crime data through natural language queries, automatically generating SQL queries and visualizations.

## Objectives
1. Implement Medallion Architecture (Bronze → Silver → Gold) for robust data engineering
2. Develop a hybrid ARIMA+LSTM model for accurate crime incident forecasting
3. Implement spatial clustering using DBSCAN to identify high-risk zones
4. Build an AI Agent with natural language to SQL capabilities using semantic search
5. Create interactive visualizations with agent integration for crime hotspot analysis
6. Deploy the solution using AWS EMR for scalable processing
7. Provide actionable insights for law enforcement resource allocation through conversational AI

## Technical Architecture
- **Data Engineering**: Medallion Architecture (Bronze/Silver/Gold layers) with PySpark
- **Time Series Forecasting**: Hybrid ARIMA+LSTM model
- **Spatial Analysis**: DBSCAN clustering for hotspot detection
- **AI Agent**: LangChain + OpenAI for natural language processing
- **Vector Database**: ChromaDB for semantic search capabilities
- **Graph Database**: Neo4j for knowledge representation and relationships
- **SQL Generation**: Text-to-SQL with semantic understanding
- **Cloud Infrastructure**: AWS EMR with automated deployment
- **Visualization**: Interactive Plotly dashboards with agent integration
- **Version Control**: Git/GitHub with CI/CD pipeline

## Data Sources
- Chicago Police Department's CLEAR (Citizen Law Enforcement Analysis and Reporting) system
- Chicago Data Portal (https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2)
- Census demographic data for contextual analysis

## Methodology

### Phase 1: Data Engineering (Medallion Architecture)
1. **Bronze Layer**: Raw data ingestion with basic validation
   - Historical crime data extraction from Chicago Open Data Portal
   - Data quality checks and schema validation
   - Incremental data loading capabilities

2. **Silver Layer**: Cleaned and standardized data
   - Data cleaning and feature engineering
   - Temporal and spatial feature creation
   - Data type standardization and null handling

3. **Gold Layer**: Business-ready aggregated datasets
   - Monthly crime aggregations for forecasting
   - Spatial clustering results for hotspot analysis
   - Feature-rich datasets optimized for ML and analytics

### Phase 2: Machine Learning & AI Agent
4. **Time Series Analysis**
   - ARIMA modeling for linear patterns
   - LSTM implementation for non-linear patterns
   - Hybrid model development and optimization

5. **Spatial Analysis**
   - DBSCAN clustering for hotspot detection
   - Geographic boundary consideration
   - Temporal-spatial correlation analysis

6. **AI Agent Development**
   - Vector database setup for semantic search
   - Graph database for knowledge representation
   - Natural language to SQL query generation
   - Conversational interface for data exploration

### Phase 3: Visualization & Deployment
7. **Interactive Dashboards**
   - Agent-integrated Plotly visualizations
   - Natural language query interface
   - Automated chart generation based on user questions

8. **Cloud Deployment**
   - AWS EMR cluster setup
   - CI/CD pipeline implementation
   - Scalable infrastructure for production use

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

### Core Data & ML Stack
- Python 3.10+
- PySpark for distributed processing
- TensorFlow/Keras for deep learning
- Scikit-learn for traditional ML
- Statsmodels for time series analysis

### AI Agent & Database Stack
- LangChain for AI agent orchestration
- OpenAI API for natural language processing
- ChromaDB for vector storage and semantic search
- Neo4j for graph database and knowledge representation
- SQLAlchemy for database ORM

### Visualization & Deployment
- Plotly for interactive visualizations
- Streamlit for web interface
- AWS EMR for cloud processing
- Docker for containerization
- GitHub Actions for CI/CD
- Git for version control

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

## Project Structure (Medallion Architecture)
```
chicago-crime-analysis/
├── data/
│   ├── bronze/          # Raw data layer
│   ├── silver/          # Cleaned data layer
│   └── gold/            # Business-ready aggregated data
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_arima_modeling.ipynb
│   ├── 03_lstm_modeling.ipynb
│   └── 04_spatial_analysis.ipynb
├── src/
│   ├── data/            # Data engineering pipelines
│   ├── models/          # ML models (ARIMA, LSTM, DBSCAN)
│   ├── agent/           # AI agent components
│   ├── visualization/   # Dashboard and plotting utilities
│   ├── database/        # Vector & Graph DB utilities
│   └── utils/           # Shared utilities
├── tests/               # Unit and integration tests
├── deployment/          # AWS EMR and Docker configs
├── requirements.txt
├── docker-compose.yml
├── README.md
└── .gitignore
```

## Getting Started
1. Clone the repository
```bash
git clone https://github.com/[Ntropy86]/chicago-crime-analysis.git
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
python -m src.data.download_data.py
```

## Contributing
Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.

## License
This project is licensed under the MIT License - see the LICENSE.md file for details

## Contact
[Neat](<ADD Eastegg here>) - [kargeti@wisc.edu](mailto:kargeti@wisc.edu)
Project Link: https://github.com/[Ntropy86]/chicago-crime-analysis