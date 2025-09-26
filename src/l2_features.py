"""
L2 Features – Enrich L1 with spatial and temporal ML features.

- H3 hex assignment (res configurable, default 9)
- Street name normalization from BLOCK
- Temporal features: hour/day/week/month, weekend, holiday
- Cyclical encoding for hour/day_of_week/month

Style: Standalone script with logging and exceptions, mirrors downloader style.
"""

import os
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

import pandas as pd
import numpy as np

# === Standalone Logging ===
import logging
import sys
import os
try:
    import h3
    h3_version = getattr(h3, '__version__', 'unknown')
    # Direct H3 test
    try:
        test_h3 = h3.latlng_to_cell(41.977441, -87.839712, 9) if hasattr(h3, 'latlng_to_cell') else h3.geo_to_h3(41.977441, -87.839712, 9)
        print(f'Direct H3 test (41.977441, -87.839712, 9): {test_h3}')
    except Exception as e:
        print(f'Direct H3 test failed: {e}')
except Exception as e:
    h3 = None
    h3_version = f'not importable: {e}'

print('=== ENV DIAGNOSTICS (l2_features.py) ===')
print('sys.executable:', sys.executable)
print('PYTHONPATH:', os.environ.get('PYTHONPATH', ''))
print('h3 version:', h3_version)
print('========================================')
import sys
import os
try:
    import h3
    h3_version = getattr(h3, '__version__', 'unknown')
except Exception as e:
    h3 = None
    h3_version = f'not importable: {e}'

print('=== ENV DIAGNOSTICS (l2_features.py) ===')
print('sys.executable:', sys.executable)
print('PYTHONPATH:', os.environ.get('PYTHONPATH', ''))
print('h3 version:', h3_version)
print('========================================')

def setup_standalone_logger():
    os.makedirs('logs', exist_ok=True)
    logger = logging.getLogger('l2_features')
    logger.setLevel(logging.DEBUG)
    
    # File handler
    fh = logging.FileHandler(f'logs/l2_features_{datetime.now().strftime("%m%d%Y")}.log')
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

L1_DIR = Path('data/l1')
L2_DIR = Path('data/l2')


def _ensure_dirs() -> None:
    try:
        os.makedirs(L2_DIR, exist_ok=True)
    except Exception as e:
        raise ConfigError(f'Failed to create L2 directory: {e}')


def _normalize_street(block: pd.Series) -> pd.Series:
    # BLOCK examples: "012XX W SOME ST" → normalize to "W SOME ST"
    s = block.fillna("").astype(str).str.upper()
    # remove leading house number patterns like 012XX, 0000-0999, etc.
    s = s.str.replace(r"^\d{3,4}XX\s+", "", regex=True)
    s = s.str.replace(r"^\d+\s+", "", regex=True)
    return s.str.strip()


def _cyclical_encode(series: pd.Series, period: int, prefix: str) -> pd.DataFrame:
    radians = 2 * np.pi * series.astype(float) / period
    return pd.DataFrame({
        f"{prefix}_sin": np.sin(radians),
        f"{prefix}_cos": np.cos(radians),
    })


def _assign_h3(df: pd.DataFrame, res: int) -> pd.Series:
    """Assign H3 hexagon IDs with robust error handling"""
    # Process all coordinates with explicit type conversion
    h3_ids = []
    success_count = 0
    
    for idx, row in df.iterrows():
        lat = row['latitude']
        lon = row['longitude']
        
        if pd.notna(lat) and pd.notna(lon):
            try:
                # Convert to Python float to avoid numpy precision issues
                lat_float = float(lat)
                lon_float = float(lon)
                
                # Validate coordinates are in reasonable range for Chicago
                if not (40.0 <= lat_float <= 43.0 and -89.0 <= lon_float <= -86.0):
                    h3_ids.append('UNKNOWN')
                    continue
                    
                h3_id = h3.latlng_to_cell(lat_float, lon_float, res)
                h3_ids.append(h3_id)
                success_count += 1
                
            except Exception as e:
                logger.debug(f"H3 failed for row {idx}: lat={lat}, lon={lon}, error={e}")
                h3_ids.append('UNKNOWN')
        else:
            h3_ids.append('UNKNOWN')
    
    total_records = len(df)
    success_rate = (success_count / total_records) * 100 if total_records > 0 else 0
    logger.info(f"H3 assignment: {success_count:,}/{total_records:,} successful ({success_rate:.1f}%)")
    
    return pd.Series(h3_ids, index=df.index)


def _handle_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values with appropriate strategies"""
    initial_count = len(df)
    
    # Strategy 1: Drop records missing critical fields for feature engineering
    critical_fields = ['datetime', 'latitude', 'longitude']
    before_critical = len(df)
    df = df.dropna(subset=critical_fields)
    dropped_critical = before_critical - len(df)
    if dropped_critical > 0:
        logger.info(f"Dropped {dropped_critical:,} records missing critical fields (datetime/coordinates)")
    
    # Strategy 2: Fill missing categorical fields with 'UNKNOWN'
    categorical_fields = ['location_type', 'block_address', 'crime_description']
    for field in categorical_fields:
        if field in df.columns:
            null_count = df[field].isna().sum()
            if null_count > 0:
                df[field] = df[field].fillna('UNKNOWN')
                logger.debug(f"Filled {null_count:,} missing values in {field} with 'UNKNOWN'")
    
    # Strategy 3: Fill missing numeric IDs with mode (most common value)
    numeric_fields = ['beat_id', 'district_id', 'ward_id', 'community_area_id']
    for field in numeric_fields:
        if field in df.columns:
            null_count = df[field].isna().sum()
            if null_count > 0:
                mode_value = df[field].mode().iloc[0] if not df[field].mode().empty else 0
                df[field] = df[field].fillna(mode_value)
                logger.debug(f"Filled {null_count:,} missing values in {field} with mode: {mode_value}")
    
    # Strategy 4: Keep boolean nulls as nulls (arrest_made, is_domestic can be unknown)
    
    final_count = len(df)
    logger.info(f"Missing data handling: {initial_count:,} → {final_count:,} records")
    
    return df


def _process_partition(part_dir: Path, h3_res: int) -> Optional[pd.DataFrame]:
    files = list(part_dir.glob('*.parquet'))
    if not files:
        return None
    try:
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    except Exception as e:
        raise DataProcessingError(f'Failed reading parquet in {part_dir}: {e}')

    # Handle missing data before feature engineering
    df = _handle_missing_data(df)
    
    if df.empty:
        logger.warning(f"No valid records remaining after missing data handling in {part_dir}")
        return None

    # Ensure datetime column exists (L1 should have created this)
    if 'datetime' not in df.columns:
        if 'created_date' in df.columns:
            df['datetime'] = pd.to_datetime(df['created_date'], errors='coerce')
        else:
            raise DataProcessingError('No datetime or created_date column present in L1')

    # Temporal features
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6])

    # Cyclical
    df = pd.concat([df, _cyclical_encode(df['hour'], 24, 'hour')], axis=1)
    df = pd.concat([df, _cyclical_encode(df['day_of_week'], 7, 'dow')], axis=1)
    df = pd.concat([df, _cyclical_encode(df['month'], 12, 'month')], axis=1)

    # Street normalization (using standardized column name). Some L1 outputs
    # mistakenly keep the original raw column name `block` instead of
    # `block_address` (case / mapping differences). Accept `block` as a
    # safe fallback so normalization still runs.
    if 'block_address' in df.columns:
        df['street_norm'] = _normalize_street(df['block_address'])
    elif 'block' in df.columns:
        logger.info("Found 'block' column - using as 'block_address' for street normalization")
        # create standardized name from fallback and normalize
        df['block_address'] = df['block']
        df['street_norm'] = _normalize_street(df['block_address'])
    else:
        logger.warning("No block_address or block column found - skipping street normalization")
        df['street_norm'] = 'UNKNOWN'

    # H3 assignment (coordinates are guaranteed to be non-null after missing data handling)
    if 'latitude' in df.columns and 'longitude' in df.columns:
        try:
            df['h3_r9'] = _assign_h3(df, h3_res)
        except Exception as e:
            logger.error(f"H3 assignment failed: {e}")
            df['h3_r9'] = None
    else:
        logger.error("Missing latitude/longitude columns - cannot assign H3")
        df['h3_r9'] = None

    return df


def _write_l2(df: pd.DataFrame, year: int, month: int) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    out_dir = L2_DIR / f'year={year}' / f'month={month:02d}'
    out_dir.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    out_file = out_dir / f'features-{year}-{month:02d}.parquet'
    pq.write_table(table, out_file)
    logger.info(f'Wrote L2 features → {out_file} ({len(df)} rows)')


def main():
    try:
        # Treat the first CLI argument (if present) as the start year. The
        # script will process all available L1 year directories where year >=
        # start_year. If no start year is provided, process all available years.
        start_year = int(sys.argv[1]) if len(sys.argv) > 1 else None
        h3_res = 9  # Fixed H3 resolution
        _ensure_dirs()

        # Gather all available year directories under data/l1
        all_year_dirs = sorted(L1_DIR.glob('year=*'))
        if not all_year_dirs:
            logger.error('No L1 data directories found under data/l1')
            return

        # Filter by start_year if provided
        if start_year is not None:
            year_dirs = [d for d in all_year_dirs if int(d.name.split('=')[1]) >= start_year]
            if not year_dirs:
                logger.error(f'No L1 data found for start_year >= {start_year}')
                return
        else:
            year_dirs = all_year_dirs

        first_year = int(year_dirs[0].name.split('=')[1])
        last_year = int(year_dirs[-1].name.split('=')[1])
        logger.info(f'Processing years {first_year} → {last_year} (start_year={start_year})')

        for y_dir in year_dirs:
            year = int(str(y_dir.name).split('=')[1])
            logger.info(f'Processing year {year}')
            for m_dir in sorted(y_dir.glob('month=*')):
                month = int(str(m_dir.name).split('=')[1])
                logger.info(f'Processing {year}-{month:02d}')
                df = _process_partition(m_dir, h3_res)
                if df is None or df.empty:
                    logger.warning(f'Empty partition at {m_dir}, skipping')
                    continue
                _write_l2(df, year, month)

        logger.info('L2 feature build completed successfully')
    except Exception as e:
        logger.error(f'L2 build failed: {e}')
        raise


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.critical(f'Application Terminated: {str(e)}')
        sys.exit(1)
