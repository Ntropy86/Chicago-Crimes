"""L3 Aggregations - daily H3-level aggregates and rolling features

Produces per-h3 per-day aggregates from L2 features and writes partitioned
parquet files to data/l3/res={res}/year=YYYY/month=MM/

Usage:
    python src/l3_aggregations.py [start_year]

This script is defensive: it reads L2 with pandas.read_parquet, uses the
sanitizer from src/l2_features.py to ensure canonical dtypes, and computes
rolling 7/14/30-day sums per h3 cell.
"""

from pathlib import Path
import sys
from datetime import timedelta
import pandas as pd
import numpy as np
import logging

# Import sanitizer from l2_features
import importlib.util
spec = importlib.util.spec_from_file_location('l2mod','src/l2_features.py')
l2mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(l2mod)

L2_DIR = Path('data/l2')
L3_DIR = Path('data/l3')

def setup_logger():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger('l3_aggregations')

logger = setup_logger()


def _find_partitions(start_year=None):
    years = sorted([p for p in L2_DIR.glob('year=*') if p.is_dir()])
    if start_year is not None:
        years = [y for y in years if int(y.name.split('=')[1]) >= start_year]
    return years


def _read_month(year:int, month:int):
    path = L2_DIR / f'year={year}' / f'month={month:02d}' / f'features-{year}-{month:02d}.parquet'
    if not path.exists():
        logger.warning(f'Missing L2 partition: {path}')
        return None
    df = pd.read_parquet(path, engine='pyarrow')
    # sanitize using l2 helper (ensures canonical dtypes)
    df = l2mod._sanitize_and_cast_for_parquet(df, 9)
    return df


def _aggregate_daily(df: pd.DataFrame, h3_res:int):
    # ensure date and h3 column
    h3_col = f'h3_r{h3_res}'
    if h3_col not in df.columns:
        logger.warning(f'{h3_col} missing from df — skipping')
        return pd.DataFrame()

    df = df.copy()
    df['date'] = pd.to_datetime(df['datetime']).dt.floor('D')
    # Resolve candidate column names with sensible fallbacks
    def first_existing(candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    id_col = first_existing(['id', 'case_number'])
    arrest_col = first_existing(['arrest_made', 'arrest', 'arrested'])
    crime_col = first_existing(['crime_type', 'primary_type', 'primarytype', 'primary type'])
    street_col = first_existing(['street_norm', 'block_address', 'block'])

    if id_col is None:
        # create synthetic id
        df['_row_id'] = range(len(df))
        id_col = '_row_id'

    # Prepare arrest series (coerce to boolean) or default to False
    if arrest_col is None:
        df['_arrest_flag'] = pd.Series([False] * len(df))
        arrest_col_use = '_arrest_flag'
    else:
        arrest_col_use = arrest_col

    # Prepare crime type or fallback
    crime_col_use = crime_col

    # Prepare street column
    if street_col is None:
        df['_street_norm'] = pd.Series([pd.NA] * len(df), dtype='string')
        street_col_use = '_street_norm'
    else:
        street_col_use = street_col

    # Basic aggregations per h3/day
    def top_mode(series):
        s = series.dropna()
        return s.mode().iloc[0] if not s.mode().empty else pd.NA

    agg = df.groupby([h3_col, 'date']).agg(
        n_crimes = (id_col, 'count'),
        n_arrests = (arrest_col_use, lambda s: int(pd.Series(s).eq(True).sum())),
        unique_streets = (street_col_use, lambda s: int(pd.Series(s).dropna().nunique())),
        top_crime_type = (crime_col_use, top_mode) if crime_col_use is not None else (id_col, lambda s: pd.NA)
    ).reset_index()

    # arrest rate
    agg['arrest_rate'] = agg['n_arrests'] / agg['n_crimes']

    # Ensure date is sorted for rolling
    agg = agg.sort_values([h3_col, 'date'])

    # Rolling sums (7/14/30 days)
    def compute_roll(df_group, window_days):
        # set date index
        g = df_group.set_index('date').asfreq('D', fill_value=0)
        rs = g['n_crimes'].rolling(window=window_days, min_periods=1).sum()
        return rs.reset_index(name=f'rolling_n_{window_days}')

    rolls = []
    for w in (7,14,30):
        parts = []
        for h3, g in agg.groupby(h3_col):
            r = compute_roll(g[['date','n_crimes']].copy(), w)
            r[h3_col] = h3
            parts.append(r)
        rolls.append(pd.concat(parts, ignore_index=True))

    # merge rolling results
    out = agg.merge(rolls[0], on=[h3_col,'date'])
    out = out.merge(rolls[1], on=[h3_col,'date'])
    out = out.merge(rolls[2], on=[h3_col,'date'])

    return out


def _write_l3(df: pd.DataFrame, h3_res:int):
    # write partitioned by year/month from date column
    if df.empty:
        return
    df = df.copy()
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month

    for (y,m), grp in df.groupby(['year','month']):
        out_dir = L3_DIR / f'res={h3_res}' / f'year={int(y)}' / f'month={int(m):02d}'
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f'h3_daily-{h3_res}-{y}-{m:02d}.parquet'
        grp.to_parquet(out_file, index=False, engine='pyarrow')
        logger.info(f'Wrote L3 {out_file} rows={len(grp)}')


def main():
    start_year = int(sys.argv[1]) if len(sys.argv) > 1 else None
    h3_res = 9

    year_dirs = _find_partitions(start_year)
    if not year_dirs:
        logger.error('No L2 year partitions found')
        return

    # For each year/month, read and aggregate (this is month-by-month safe)
    for y_dir in year_dirs:
        year = int(y_dir.name.split('=')[1])
        for m_dir in sorted(y_dir.glob('month=*')):
            month = int(m_dir.name.split('=')[1])
            logger.info(f'Processing L3 for {year}-{month:02d}')
            df = _read_month(year, month)
            if df is None or df.empty:
                logger.info(f'No data for {year}-{month:02d}, skipping')
                continue

            out = _aggregate_daily(df, h3_res)
            if out.empty:
                logger.info(f'Empty aggregation for {year}-{month:02d}')
                continue
            _write_l3(out, h3_res)

    logger.info('L3 build completed')


if __name__ == '__main__':
    main()
