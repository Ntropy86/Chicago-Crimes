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
from pandas.api.types import CategoricalDtype

from utils import categorize_primary_type

# === Standalone Logging ===
import logging
import sys
import os
try:
    import h3
    h3_version = getattr(h3, '__version__', 'unknown')
except Exception as e:
    h3 = None
    h3_version = f'not importable: {e}'

try:
    from h3.api.numpy_int import vectorized_latlng_to_cell
except Exception:
    vectorized_latlng_to_cell = None

# Minimal diagnostics go to logger after setup; avoid noisy prints on import.

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

# H3 resolutions to materialize at L2 (coarse -> fine). Keep small list to limit storage.
H3_RESOLUTIONS = [7, 8, 9]


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


if h3 is not None:
    if hasattr(h3, 'cell_to_parent'):
        _cell_to_parent = h3.cell_to_parent
    elif hasattr(h3, 'h3_to_parent'):
        _cell_to_parent = h3.h3_to_parent
    else:
        _cell_to_parent = None
    _int_to_str = getattr(h3, 'int_to_str', None) or getattr(h3, 'h3_to_string', None)
else:
    _cell_to_parent = None
    _int_to_str = None


def _cells_to_str(cells: np.ndarray) -> np.ndarray:
    """Normalize H3 ids to their canonical string representation."""
    if cells.dtype.kind in {'U', 'S', 'O'}:
        return cells.astype(object)
    if _int_to_str is not None:
        return np.asarray([_int_to_str(int(c)) for c in cells], dtype=object)
    return np.asarray([format(int(c), 'x') for c in cells], dtype=object)


def _assign_h3(df: pd.DataFrame, res: int, *, chunk_size: int = 50_000) -> pd.Series:
    """Assign H3 hexagon IDs in batches to minimise Python-loop overhead."""
    total_records = len(df)
    if total_records == 0 or h3 is None:
        return pd.Series([pd.NA] * total_records, dtype='string', index=df.index)

    lat = pd.to_numeric(df.get('latitude'), errors='coerce').to_numpy()
    lon = pd.to_numeric(df.get('longitude'), errors='coerce').to_numpy()

    valid_mask = (
        np.isfinite(lat) & np.isfinite(lon)
        & (40.0 <= lat) & (lat <= 43.0)
        & (-89.0 <= lon) & (lon <= -86.0)
    )

    result: np.ndarray = np.full(total_records, pd.NA, dtype=object)
    valid_lat = lat[valid_mask]
    valid_lon = lon[valid_mask]
    n_valid = valid_lat.size

    if n_valid == 0:
        logger.info('H3 assignment: 0/%s successful (0.0%%)', f"{total_records:,}")
        return pd.Series(result, index=df.index, dtype='string')

    assigned_cells: np.ndarray
    try:
        if vectorized_latlng_to_cell is not None:
            assigned_cells = _cells_to_str(vectorized_latlng_to_cell(valid_lat, valid_lon, res))
        else:
            # Fall back to chunked scalar calls to avoid per-row logging overhead.
            cells = []
            for start in range(0, n_valid, chunk_size):
                stop = start + chunk_size
                chunk_lat = valid_lat[start:stop]
                chunk_lon = valid_lon[start:stop]
                cells.extend(
                    (h3.latlng_to_cell(lat_val, lon_val, res)
                     if hasattr(h3, 'latlng_to_cell')
                     else h3.geo_to_h3(lat_val, lon_val, res))
                    for lat_val, lon_val in zip(chunk_lat, chunk_lon)
                )
            assigned_cells = np.asarray(cells, dtype=object)
    except Exception as exc:
        logger.error('Vectorised H3 assignment failed at r%s: %s', res, exc)
        return pd.Series(result, index=df.index, dtype='string')

    result[valid_mask] = assigned_cells
    success_rate = (n_valid / total_records) * 100
    logger.info('H3 assignment: %s/%s successful (%.1f%%) for r%s',
                f"{n_valid:,}", f"{total_records:,}", success_rate, res)

    return pd.Series(result, index=df.index, dtype='string')


def _derive_parent_cells(base: pd.Series, base_res: int, target_res: int) -> Optional[pd.Series]:
    """Derive coarser-resolution H3 cells from a higher-resolution series."""
    if target_res == base_res:
        return base.copy()
    if _cell_to_parent is None or h3 is None:
        return None

    parents = []
    for cell in base.astype('object'):
        if pd.isna(cell):
            parents.append(pd.NA)
            continue
        try:
            parents.append(_cell_to_parent(cell, target_res))
        except Exception as exc:
            logger.debug('Failed to derive parent for %s → r%s: %s', cell, target_res, exc)
            parents.append(pd.NA)
    return pd.Series(parents, index=base.index, dtype='string')


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


def _sanitize_and_cast_for_parquet(df: pd.DataFrame, h3_res: int) -> pd.DataFrame:
    """Ensure deterministic dtypes for parquet writing to avoid schema merge errors.

    - Avoid pandas 'category' dtype
    - Cast temporal ints to fixed-width ints
    - Ensure cyclical floats are float64
    - Ensure h3 column is string dtype with pd.NA for missing
    """
    # Convert any categorical columns to string to avoid dictionary encodings
    for col in df.columns:
        if isinstance(df[col].dtype, CategoricalDtype):
            df[col] = df[col].astype('string')

    # Temporal columns
    if 'datetime' in df.columns:
        # keep `date` as a pandas datetime64[ns] (midnight) to ensure pyarrow
        # can convert it deterministically instead of python date objects
        df['date'] = pd.to_datetime(df['datetime']).dt.normalize()
        df['year'] = pd.to_datetime(df['datetime']).dt.year.astype('int32')
        df['month'] = pd.to_datetime(df['datetime']).dt.month.astype('int8')
        df['day_of_week'] = pd.to_datetime(df['datetime']).dt.dayofweek.astype('int8')
        df['hour'] = pd.to_datetime(df['datetime']).dt.hour.astype('int8')

    # Cyclical features to float64
    cyc_cols = [c for c in df.columns if c.endswith('_sin') or c.endswith('_cos')]
    for c in cyc_cols:
        try:
            df[c] = df[c].astype('float64')
        except Exception:
            # coerce non-numeric to NaN then to float64
            df[c] = pd.to_numeric(df[c], errors='coerce').astype('float64')

    # H3 column name may vary; normalize to 'h3_r{res}'
    h3_col = f'h3_r{h3_res}'
    if h3_col not in df.columns:
        # attempt fallback names
        for candidate in ['h3', 'h3_index', 'h3id', 'h3_r9']:
            if candidate in df.columns:
                df[h3_col] = df[candidate]
                break

    if h3_col in df.columns:
        # ensure string dtype with pd.NA for missing
        df[h3_col] = df[h3_col].where(pd.notna(df[h3_col]), pd.NA).astype('string')
    else:
        df[h3_col] = pd.Series([pd.NA] * len(df), dtype='string')

    # Ensure street_norm exists and is string
    if 'street_norm' in df.columns:
        df['street_norm'] = df['street_norm'].astype('string')
    else:
        df['street_norm'] = pd.Series([pd.NA] * len(df), dtype='string')

    if 'crime_category' in df.columns:
        df['crime_category'] = df['crime_category'].astype('string')

    # Ensure boolean columns are proper booleans or pandas boolean dtype
    bool_cols = ['is_weekend', 'arrest_made', 'is_domestic']
    for b in bool_cols:
        if b in df.columns:
            try:
                df[b] = df[b].astype('boolean')
            except Exception:
                df[b] = df[b].astype('boolean')

    # Numeric IDs: use pandas nullable Int64 where possible
    int_id_cols = ['beat_id', 'district_id', 'ward_id', 'community_area_id', 'incident_year']
    for col in int_id_cols:
        if col in df.columns:
            try:
                df[col] = df[col].astype('Int64')
            except Exception:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')

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

    if 'primary_type' not in df.columns and 'crime_type' in df.columns:
        df['primary_type'] = df['crime_type']
    if 'primary_type' in df.columns:
        df['primary_type'] = df['primary_type'].astype('string')
        df['crime_category'] = df['primary_type'].map(categorize_primary_type).astype('string')
    else:
        df['crime_category'] = pd.Series(['Unclassified'] * len(df), dtype='string')

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

    base_res = max(H3_RESOLUTIONS)
    base_col = f'h3_r{base_res}'

    if 'latitude' in df.columns and 'longitude' in df.columns:
        try:
            df[base_col] = _assign_h3(df, base_res)
        except Exception as e:
            logger.error(f"H3 assignment failed at r{base_res}: {e}")
            df[base_col] = pd.Series([pd.NA] * len(df), dtype='string')
    else:
        logger.error("Missing latitude/longitude columns - cannot assign H3")
        df[base_col] = pd.Series([pd.NA] * len(df), dtype='string')

    for res in H3_RESOLUTIONS:
        if res == base_res:
            continue
        target_col = f'h3_r{res}'
        derived = _derive_parent_cells(df[base_col], base_res, res)
        if derived is not None:
            df[target_col] = derived
        else:
            try:
                df[target_col] = _assign_h3(df, res)
            except Exception as e:
                logger.warning(f"Failed assigning h3 at resolution {res}: {e}")
                df[target_col] = pd.Series([pd.NA] * len(df), dtype='string')

    # Sanitize and enforce deterministic dtypes before returning/writing
    df = _sanitize_and_cast_for_parquet(df, h3_res)

    return df


def _write_l2(df: pd.DataFrame, year: int, month: int) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    out_dir = L2_DIR / f'year={year}' / f'month={month:02d}'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build a deterministic pyarrow schema from pandas dtypes
    pa_fields = []
    for col in df.columns:
        pd_type = df[col].dtype
        if pd.api.types.is_integer_dtype(pd_type):
            pa_type = pa.int64()
        elif pd.api.types.is_float_dtype(pd_type):
            pa_type = pa.float64()
        elif pd.api.types.is_bool_dtype(pd_type):
            pa_type = pa.bool_()
        elif pd.api.types.is_datetime64_any_dtype(pd_type):
            pa_type = pa.timestamp('ns')
        else:
            # strings and object fall back to string
            pa_type = pa.string()
        pa_fields.append(pa.field(col, pa_type))

    schema = pa.schema(pa_fields)

    # Add metadata for provenance
    meta = {
        'h3_resolutions': ','.join(map(str, H3_RESOLUTIONS)),
        'producer': 'src/l2_features.py',
    }
    # attach metadata as bytes
    schema = schema.with_metadata({k: v.encode('utf8') for k, v in meta.items()})

    table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
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
