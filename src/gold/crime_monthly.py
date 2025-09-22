import pandas as pd
import numpy as np
import glob
from pathlib import Path
from typing import Iterable, List, Optional, Dict, Any
from datetime import datetime, timedelta
import warnings

try:
    from src.utils.logger_config import setup_logger
    from src.utils.exceptions import DataProcessingError
except Exception:  # pragma: no cover - fallback for ad-hoc runs
    import logging

    def setup_logger(name: str) -> logging.Logger:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )
        return logging.getLogger(name)

    class DataProcessingError(Exception):
        pass


logger = setup_logger(__name__)

class CrimeMonthlyAggregator:
    """Builder for the Phase-1 Gold dataset: `crime_monthly.parquet`.

    Design:
        - Orchestrates a pure, ordered set of steps (load → filter → aggregate →
          backfill → windows → enforce schema → write).
        - Keeps config, logging, and error handling in one place.
        - Easy to extend with Spark later by swapping step internals.

    Public API:
        - build(): runs the full pipeline and writes the Parquet.

    """
    def __init__(
        self,
        raw_data_path: str = "data/raw", 
        processed_data_path: str = "data/processed",
        output_path: str = "data/gold",
        start_year: Optional[int] = None,
        end_year: Optional[int] = None
    ):
        """
        Initialize the CrimeMonthlyAggregator.
        
        Args:
            raw_data_path: Path to raw crime data files
            processed_data_path: Path to processed crime data files  
            output_path: Path where gold datasets will be saved
            start_year: Optional start year for processing (default: all available)
            end_year: Optional end year for processing (default: current year)
        """
        self.raw_data_path = Path(raw_data_path)
        self.processed_data_path = Path(processed_data_path)
        self.output_path = Path(output_path)
        
        # Ensure output directory exists
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Year range configuration
        self.start_year = start_year
        self.end_year = end_year or datetime.now().year
        
        # Schema definition for output consistency
        self.gold_schema = {
            'year_month': 'datetime64[ns]',
            'total_crimes': 'int64',
            'violent_crimes': 'int64', 
            'property_crimes': 'int64',
            'drug_crimes': 'int64',
            'arrest_rate': 'float64',
            'domestic_rate': 'float64',
            'avg_latitude': 'float64',
            'avg_longitude': 'float64',
            'unique_locations': 'int64',
            'top_crime_type': 'object',
            'total_districts': 'int64'
        }
        
        # Crime type categorization for business logic
        self.violent_crimes = {
            'ASSAULT', 'BATTERY', 'HOMICIDE', 'KIDNAPPING', 
            'ROBBERY', 'SEX OFFENSE', 'CRIM SEXUAL ASSAULT'
        }
        
        self.property_crimes = {
            'BURGLARY', 'THEFT', 'MOTOR VEHICLE THEFT', 'ARSON',
            'CRIMINAL DAMAGE', 'CRIMINAL TRESPASS', 'VANDALISM'
        }
        
        self.drug_crimes = {
            'NARCOTICS', 'OTHER NARCOTIC VIOLATION'
        }
        
        logger.info(f"Initialized CrimeMonthlyAggregator with output path: {self.output_path}")

    def _load_raw_data(self) -> pd.DataFrame:
        """
        Load and combine all raw crime data files.
        
        Returns:
            Combined DataFrame with all crime records
            
        Raises:
            DataProcessingError: If no data files found or loading fails
        """
        try:
            # Find all crime data files
            file_pattern = "chicago_crimes_*.csv"
            data_files = list(self.raw_data_path.glob(file_pattern))
            
            if not data_files:
                raise DataProcessingError(f"No crime data files found in {self.raw_data_path}")
            
            logger.info(f"Found {len(data_files)} crime data files")
            
            # Load and combine data
            dataframes = []
            total_records = 0
            
            for file_path in sorted(data_files):
                logger.debug(f"Loading {file_path.name}...")
                
                # Extract year from filename for filtering
                year_str = file_path.stem.split('_')[-1]
                try:
                    file_year = int(year_str)
                except ValueError:
                    logger.warning(f"Could not extract year from {file_path.name}, skipping")
                    continue
                
                # Apply year filtering
                if self.start_year and file_year < self.start_year:
                    continue
                if file_year > self.end_year:
                    continue
                    
                df = pd.read_csv(file_path, low_memory=False)
                dataframes.append(df)
                total_records += len(df)
                logger.debug(f"Loaded {len(df)} records from {file_path.name}")
            
            if not dataframes:
                raise DataProcessingError("No data loaded after applying year filters")
                
            # Combine all dataframes
            combined_df = pd.concat(dataframes, ignore_index=True)
            logger.info(f"Successfully loaded {total_records} total crime records")
            
            return combined_df
            
        except Exception as e:
            logger.error(f"Failed to load raw data: {str(e)}")
            raise DataProcessingError(f"Failed to load raw data: {str(e)}")

    def _clean_and_prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and prepare the raw data for aggregation.
        
        Args:
            df: Raw crime data DataFrame
            
        Returns:
            Cleaned DataFrame ready for aggregation
        """
        try:
            logger.info("Starting data cleaning and preparation...")
            
            # Convert date column to datetime
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            
            # Remove records with invalid dates
            initial_count = len(df)
            df = df.dropna(subset=['date'])
            logger.info(f"Removed {initial_count - len(df)} records with invalid dates")
            
            # Create year_month column for aggregation
            df['year_month'] = df['date'].dt.to_period('M').dt.to_timestamp()
            
            # Clean crime types
            df['primary_type'] = df['primary_type'].str.upper().str.strip()
            
            # Create crime category columns
            df['is_violent'] = df['primary_type'].isin(self.violent_crimes)
            df['is_property'] = df['primary_type'].isin(self.property_crimes)  
            df['is_drug'] = df['primary_type'].isin(self.drug_crimes)
            
            # Ensure boolean columns are properly formatted
            df['arrest'] = df['arrest'].fillna(False).astype(bool)
            df['domestic'] = df['domestic'].fillna(False).astype(bool)
            
            # Clean location data
            df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
            df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
            
            # Filter out records with invalid coordinates (outside Chicago area roughly)
            valid_coords = (
                (df['latitude'].between(41.0, 42.5)) & 
                (df['longitude'].between(-88.5, -87.0))
            )
            coord_filtered = df[valid_coords].copy()
            logger.info(f"Filtered {len(df) - len(coord_filtered)} records with invalid coordinates")
            
            return coord_filtered
            
        except Exception as e:
            logger.error(f"Failed to clean and prepare data: {str(e)}")
            raise DataProcessingError(f"Failed to clean and prepare data: {str(e)}")

    def _create_monthly_aggregations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create monthly aggregations of crime data.
        
        Args:
            df: Cleaned crime data
            
        Returns:
            Monthly aggregated DataFrame
        """
        try:
            logger.info("Creating monthly aggregations...")
            
            # Group by year_month and aggregate
            monthly_agg = df.groupby('year_month').agg({
                'id': 'count',  # Total crimes
                'is_violent': 'sum',  # Violent crimes
                'is_property': 'sum',  # Property crimes
                'is_drug': 'sum',  # Drug crimes
                'arrest': 'sum',  # Total arrests
                'domestic': 'sum',  # Domestic incidents
                'latitude': 'mean',  # Average latitude
                'longitude': 'mean',  # Average longitude
                'block': 'nunique',  # Unique locations
                'district': 'nunique',  # Number of districts
                'primary_type': lambda x: x.mode().iloc[0] if not x.empty else 'UNKNOWN'  # Most common crime
            }).reset_index()
            
            # Rename columns to match schema
            monthly_agg.columns = [
                'year_month', 'total_crimes', 'violent_crimes', 'property_crimes',
                'drug_crimes', 'total_arrests', 'domestic_incidents', 'avg_latitude',
                'avg_longitude', 'unique_locations', 'total_districts', 'top_crime_type'
            ]
            
            # Calculate rates
            monthly_agg['arrest_rate'] = (
                monthly_agg['total_arrests'] / monthly_agg['total_crimes']
            ).fillna(0).round(4)
            
            monthly_agg['domestic_rate'] = (
                monthly_agg['domestic_incidents'] / monthly_agg['total_crimes']  
            ).fillna(0).round(4)
            
            # Remove intermediate columns
            monthly_agg = monthly_agg.drop(['total_arrests', 'domestic_incidents'], axis=1)
            
            logger.info(f"Created {len(monthly_agg)} monthly aggregation records")
            return monthly_agg
            
        except Exception as e:
            logger.error(f"Failed to create monthly aggregations: {str(e)}")
            raise DataProcessingError(f"Failed to create monthly aggregations: {str(e)}")

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add time-based features for analysis and forecasting.
        
        Args:
            df: Monthly aggregated data
            
        Returns:
            DataFrame with additional time features
        """
        try:
            logger.info("Adding time-based features...")
            
            # Sort by date to ensure proper ordering
            df = df.sort_values('year_month').reset_index(drop=True)
            
            # Add rolling window features (3-month and 12-month)
            for window in [3, 12]:
                df[f'total_crimes_ma_{window}'] = (
                    df['total_crimes'].rolling(window=window, min_periods=1).mean().round(2)
                )
                df[f'violent_crimes_ma_{window}'] = (
                    df['violent_crimes'].rolling(window=window, min_periods=1).mean().round(2)
                )
            
            # Add lag features (previous month values)
            df['total_crimes_lag1'] = df['total_crimes'].shift(1)
            df['violent_crimes_lag1'] = df['violent_crimes'].shift(1) 
            
            # Add year-over-year comparison
            df['total_crimes_yoy'] = df['total_crimes'].shift(12)
            df['total_crimes_yoy_change'] = (
                (df['total_crimes'] - df['total_crimes_yoy']) / df['total_crimes_yoy'] * 100
            ).round(2)
            
            # Add seasonal indicators
            df['month'] = df['year_month'].dt.month
            df['quarter'] = df['year_month'].dt.quarter
            df['is_summer'] = df['month'].isin([6, 7, 8])
            df['is_winter'] = df['month'].isin([12, 1, 2])
            
            logger.info("Successfully added time-based features")
            return df
            
        except Exception as e:
            logger.error(f"Failed to add time features: {str(e)}")
            raise DataProcessingError(f"Failed to add time features: {str(e)}")

    def _enforce_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enforce the gold schema and data types.
        
        Args:
            df: DataFrame to enforce schema on
            
        Returns:
            DataFrame with enforced schema
        """
        try:
            logger.info("Enforcing gold dataset schema...")
            
            # Ensure all required columns exist
            for col, dtype in self.gold_schema.items():
                if col not in df.columns:
                    if col in ['total_crimes', 'violent_crimes', 'property_crimes', 'drug_crimes']:
                        df[col] = 0
                    elif col in ['arrest_rate', 'domestic_rate']:
                        df[col] = 0.0
                    elif col == 'top_crime_type':
                        df[col] = 'UNKNOWN'
                    else:
                        logger.warning(f"Missing column {col} in schema enforcement")
            
            # Convert data types
            for col, dtype in self.gold_schema.items():
                if col in df.columns:
                    try:
                        df[col] = df[col].astype(dtype)
                    except Exception as e:
                        logger.warning(f"Could not convert {col} to {dtype}: {str(e)}")
            
            # Select only schema columns in correct order
            schema_cols = list(self.gold_schema.keys())
            available_cols = [col for col in schema_cols if col in df.columns]
            df_schema = df[available_cols].copy()
            
            logger.info(f"Schema enforced successfully with {len(available_cols)} columns")
            return df_schema
            
        except Exception as e:
            logger.error(f"Failed to enforce schema: {str(e)}")
            raise DataProcessingError(f"Failed to enforce schema: {str(e)}")

    def build(self, output_format: str = 'parquet') -> str:
        """
        Execute the full pipeline to build the gold monthly crime dataset.
        
        Args:
            output_format: Output format ('parquet' or 'csv')
            
        Returns:
            Path to the created output file
            
        Raises:
            DataProcessingError: If pipeline execution fails
        """
        try:
            logger.info("Starting CrimeMonthlyAggregator pipeline...")
            
            # Step 1: Load raw data
            raw_df = self._load_raw_data()
            
            # Step 2: Clean and prepare data
            clean_df = self._clean_and_prepare_data(raw_df)
            
            # Step 3: Create monthly aggregations
            monthly_df = self._create_monthly_aggregations(clean_df)
            
            # Step 4: Add time features
            featured_df = self._add_time_features(monthly_df)
            
            # Step 5: Enforce schema for core columns
            final_df = self._enforce_schema(featured_df)
            
            # Step 6: Save output
            if output_format.lower() == 'parquet':
                output_file = self.output_path / 'crime_monthly.parquet'
                final_df.to_parquet(output_file, index=False)
            else:
                output_file = self.output_path / 'crime_monthly.csv'
                final_df.to_csv(output_file, index=False)
            
            logger.info(f"Successfully created gold dataset: {output_file}")
            logger.info(f"Final dataset shape: {final_df.shape}")
            
            # Log summary statistics
            logger.info("Dataset Summary:")
            logger.info(f"  Date range: {final_df['year_month'].min()} to {final_df['year_month'].max()}")
            logger.info(f"  Total months: {len(final_df)}")
            logger.info(f"  Avg crimes per month: {final_df['total_crimes'].mean():.1f}")
            logger.info(f"  Avg arrest rate: {final_df['arrest_rate'].mean():.3f}")
            
            return str(output_file)
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            raise DataProcessingError(f"Pipeline execution failed: {str(e)}")


def main():
    """CLI entry point for building the monthly crime dataset."""
    try:
        aggregator = CrimeMonthlyAggregator()
        output_file = aggregator.build()
        print(f"✅ Gold dataset created successfully: {output_file}")
        
    except Exception as e:
        logger.error(f"Failed to build monthly crime dataset: {str(e)}")
        print(f"❌ Failed to build dataset: {str(e)}")
        raise


if __name__ == '__main__':
    main()
        
