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

# === Configs ===
try:
    from src.utils.logger_config import setup_logger
    from src.utils.exceptions import DataProcessingError, ConfigError
    logger = setup_logger(__name__)
    logger.info('Logger successfully Initialized (L2)')
except Exception as e:
    print(f'CRITICAL ERROR: l2_features : Logger Config issue : {str(e)}')
    sys.exit(1)
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
    try:
        import h3
        # h3>=4 uses h3.latlng_to_cell(lat, lon, res)
        if hasattr(h3, 'latlng_to_cell'):
            return df.apply(lambda r: h3.latlng_to_cell(float(r['latitude']), float(r['longitude']), res)
                            if pd.notna(r['latitude']) and pd.notna(r['longitude']) else None, axis=1)
        else:
            # older API
            return df.apply(lambda r: h3.geo_to_h3(float(r['latitude']), float(r['longitude']), res)
                            if pd.notna(r['latitude']) and pd.notna(r['longitude']) else None, axis=1)
    except Exception as e:
        raise DataProcessingError(f'H3 assignment failed: {e}')


def _process_partition(part_dir: Path, h3_res: int) -> Optional[pd.DataFrame]:
    files = list(part_dir.glob('*.parquet'))
    if not files:
        return None
    try:
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    except Exception as e:
        raise DataProcessingError(f'Failed reading parquet in {part_dir}: {e}')

    if 'datetime' not in df.columns:
        if 'date' in df.columns:
            df['datetime'] = pd.to_datetime(df['date'], errors='coerce')
        else:
            raise DataProcessingError('No datetime or date column present in L1')

    # Temporal features
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6])

    # Cyclical
    df = pd.concat([df, _cyclical_encode(df['hour'], 24, 'hour')], axis=1)
    df = pd.concat([df, _cyclical_encode(df['day_of_week'], 7, 'dow')], axis=1)
    df = pd.concat([df, _cyclical_encode(df['month'], 12, 'month')], axis=1)

    # Street normalization
    if 'block' in df.columns:
        df['street_norm'] = _normalize_street(df['block'])
    else:
        df['street_norm'] = None

    # H3
    if 'latitude' in df.columns and 'longitude' in df.columns:
        df['h3_r9'] = _assign_h3(df, h3_res)
    else:
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
        h3_res = int(sys.argv[1]) if len(sys.argv) > 1 else 9
        _ensure_dirs()

        for y_dir in sorted(L1_DIR.glob('year=*')):
            year = int(str(y_dir.name).split('=')[1])
            for m_dir in sorted(y_dir.glob('month=*')):
                month = int(str(m_dir.name).split('=')[1])
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
