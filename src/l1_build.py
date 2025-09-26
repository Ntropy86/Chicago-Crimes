"""
L1 Builder - Standardize and partition raw Chicago crime CSVs into Parquet.

Style: Mirrors src/data/download_data.py (standalone script, logging, exceptions).

Inputs:
  - data/raw/chicago_crimes_YYYY.csv (2018→today)
Outputs:
  - data/l1/year=YYYY/month=MM/part-*.parquet

Operations:
  - Enforce schema & dtypes
  - Parse date, extract year/month/day/hour
  - Drop rows with invalid/missing LATITUDE/LONGITUDE
  - Keep core analytical columns
"""

import os
import sys
from pathlib import Path
from typing import Dict, List
from datetime import datetime

import pandas as pd

# === Standalone Logging ===
import logging

def setup_standalone_logger():
    os.makedirs('logs', exist_ok=True)
    logger = logging.getLogger('l1_build')
    logger.setLevel(logging.DEBUG)
    
    # File handler
    fh = logging.FileHandler(f'logs/l1_build_{datetime.now().strftime("%m%d%Y")}.log')
    fh.setLevel(logging.DEBUG)
    fh_fmt = logging.Formatter('%(levelname)s : %(name)s.%(funcName)s : %(message)s')
    fh.setFormatter(fh_fmt)
    
    # Console handler  
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch_fmt = logging.Formatter('%(asctime)s - %(levelname)s : %(message)s')
    ch.setFormatter(ch_fmt)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

logger = setup_standalone_logger()

class DataProcessingError(Exception):
    pass

class ConfigError(Exception):
    pass
# === END Configs ===

RAW_DIR = Path('data/raw')
L1_DIR = Path('data/l1')

# Column name standardization mapping (raw CSV has lowercase names)
COLUMN_MAPPING: Dict[str, str] = {
    'id': 'id',  # already standardized
    'case_number': 'case_number',  # already standardized 
    'date': 'created_date',
    'block': 'block_address',
    'iucr': 'iucr_code',
    'primary_type': 'crime_type',
    'description': 'crime_description',
    'location_description': 'location_type',
    'arrest': 'arrest_made',
    'domestic': 'is_domestic',
    'beat': 'beat_id',
    'district': 'district_id', 
    'ward': 'ward_id',
    'community_area': 'community_area_id',
    'fbi_code': 'fbi_code',  # already standardized
    'x_coordinate': 'x_coordinate',  # already standardized
    'y_coordinate': 'y_coordinate',  # already standardized
    'year': 'incident_year',
    'updated_on': 'updated_on',  # already standardized
    'latitude': 'latitude',  # already standardized
    'longitude': 'longitude',  # already standardized
    'location': 'location_point',
}

# Standardized column types
CORE_COLUMNS: Dict[str, str] = {
    'id': 'string',
    'case_number': 'string',
    'created_date': 'string',
    'block_address': 'string',
    'iucr_code': 'string',
    'crime_type': 'string',
    'crime_description': 'string',
    'location_type': 'string',
    'arrest_made': 'boolean',
    'is_domestic': 'boolean',
    'beat_id': 'Int64',
    'district_id': 'Int64',
    'ward_id': 'Int64',
    'community_area_id': 'Int64',
    'fbi_code': 'string',
    'x_coordinate': 'float64',
    'y_coordinate': 'float64',
    'incident_year': 'Int64',
    'updated_on': 'string',
    'latitude': 'float64',
    'longitude': 'float64',
    'location_point': 'string',
}

KEEP_COLUMNS: List[str] = [
    'id','case_number','created_date','block_address','iucr_code','crime_type',
    'crime_description','location_type','arrest_made','is_domestic','beat_id',
    'district_id','ward_id','community_area_id','fbi_code','x_coordinate',
    'y_coordinate','incident_year','updated_on','latitude','longitude','location_point'
]


def _ensure_dirs() -> None:
    try:
        os.makedirs(L1_DIR, exist_ok=True)
        logger.debug('Created L1 directory')
    except Exception as e:
        raise ConfigError(f'Failed to create L1 directory: {e}')


def _read_raw_year(year_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(year_path)
        logger.info(f'Read raw file {year_path.name} with {len(df)} rows')
        return df
    except Exception as e:
        raise DataProcessingError(f'Failed reading {year_path}: {e}')


def _standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names from messy raw CSV to clean semantic names"""
    # Map raw column names to standardized names
    df = df.rename(columns=COLUMN_MAPPING)
    
    # Log column mapping for transparency
    mapped_cols = [col for col in df.columns if col in COLUMN_MAPPING.values()]
    logger.debug(f"Standardized {len(mapped_cols)} column names")
    
    return df


def _normalize_and_cast(df: pd.DataFrame) -> pd.DataFrame:
    # First standardize column names
    df = _standardize_column_names(df)
    
    # Keep only known columns present
    cols_present = [c for c in KEEP_COLUMNS if c in df.columns]
    df = df[cols_present].copy()
    
    logger.debug(f"Kept {len(cols_present)} out of {len(KEEP_COLUMNS)} expected columns")

    # Cast types where possible
    for col, dtype in CORE_COLUMNS.items():
        if col in df.columns:
            try:
                if dtype == 'boolean':
                    df[col] = df[col].astype('boolean')
                else:
                    df[col] = df[col].astype(dtype)
            except Exception:
                logger.warning(f'Type cast failed for {col}; leaving as-is')

    # Parse date column to datetime (now using standardized name)
    if 'created_date' in df.columns:
        try:
            df['datetime'] = pd.to_datetime(df['created_date'], errors='coerce')
        except Exception:
            df['datetime'] = pd.NaT
    else:
        df['datetime'] = pd.NaT

    # Data Quality Reporting
    total_records = len(df)
    null_datetime = df['datetime'].isna().sum()
    null_coords = df[['latitude', 'longitude']].isna().any(axis=1).sum()
    
    logger.info(f"Data Quality Check - Total records: {total_records:,}")
    logger.info(f"Data Quality Check - Null datetime: {null_datetime:,} ({null_datetime/total_records*100:.1f}%)")
    logger.info(f"Data Quality Check - Null coordinates: {null_coords:,} ({null_coords/total_records*100:.1f}%)")

    # Drop rows without valid datetime
    before = len(df)
    df = df[~df['datetime'].isna()].copy()
    dropped_datetime = before - len(df)
    logger.info(f'Dropped {dropped_datetime:,} rows without valid datetime')

    # Derive partitions
    df['year_part'] = df['datetime'].dt.year.astype('int32')
    df['month_part'] = df['datetime'].dt.month.astype('int16')

    # Filter invalid coordinates (Chicago approx bounds)
    if 'latitude' in df.columns and 'longitude' in df.columns:
        before_coords = len(df)
        lat = df['latitude']
        lon = df['longitude']
        
        # Check for null coordinates first
        has_coords = ~(lat.isna() | lon.isna())
        
        # Then check bounds for non-null coordinates
        valid_bounds = lat.between(41.0, 42.5) & lon.between(-88.5, -87.0)
        
        # Keep records with valid coordinates within Chicago bounds
        valid = has_coords & valid_bounds
        df = df[valid].copy()
        
        dropped_coords = before_coords - len(df)
        logger.info(f'Dropped {dropped_coords:,} rows with invalid/missing coordinates')
        
        # Final data quality summary
        final_records = len(df)
        total_dropped = total_records - final_records
        logger.info(f"Final L1 output: {final_records:,} records ({total_dropped:,} dropped, {total_dropped/total_records*100:.1f}%)")

    return df


def _write_partitions(df: pd.DataFrame) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    for (y, m), grp in df.groupby(['year_part', 'month_part']):
        out_dir = L1_DIR / f'year={int(y)}' / f'month={int(m):02d}'
        out_dir.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pandas(grp.drop(columns=['year_part','month_part']), preserve_index=False)
        out_file = out_dir / f'part-{int(y)}-{int(m):02d}.parquet'
        pq.write_table(table, out_file)
        logger.info(f'Wrote {len(grp)} rows → {out_file}')


def _process_year(year: int) -> None:
    year_file = RAW_DIR / f'chicago_crimes_{year}.csv'
    if not year_file.exists():
        logger.warning(f'Missing raw file for {year}, skipping')
        return
    df = _read_raw_year(year_file)
    df = _normalize_and_cast(df)
    if df.empty:
        logger.warning(f'No valid rows after cleaning for {year}')
        return
    _write_partitions(df)


def main():
    try:
        if len(sys.argv) > 1:
            start_year = int(sys.argv[1])
        else:
            start_year = 2018
        end_year = datetime.now().year
        _ensure_dirs()
        for year in range(start_year, end_year + 1):
            _process_year(year)
        logger.info('L1 build completed successfully')
    except Exception as e:
        logger.error(f'L1 build failed: {e}')
        raise


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.critical(f'Application Terminated: {str(e)}')
        sys.exit(1)
