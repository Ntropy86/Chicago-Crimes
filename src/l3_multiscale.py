"""L3 Multiscale Aggregator

Computes per-H3 (multi-resolution) daily and monthly aggregates with
counts, raw rates, smoothed rates, Wilson CIs, low-confidence flags,
and neighbor-pooled counts. Writes partitioned parquet to data/l3/res={res}/year=YYYY/month=MM/.

Usage:
    python src/l3_multiscale.py [year] [month]

If year/month are omitted, processes all available L2 partitions.
"""
from pathlib import Path
import sys
import logging
from datetime import datetime
from typing import Optional

import pandas as pd
import numpy as np

import importlib.util
spec = importlib.util.spec_from_file_location('l2mod','src/l2_features.py')
l2mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(l2mod)

try:
    import h3
except Exception:
    h3 = None

# detect k_ring function across h3 package variants
k_ring_func = None
if h3 is not None:
    if hasattr(h3, 'k_ring'):
        k_ring_func = h3.k_ring
    else:
        # some h3 installs expose api.basic_int.k_ring
        try:
            from h3.api import basic_int
            if hasattr(basic_int, 'k_ring'):
                k_ring_func = basic_int.k_ring
        except Exception:
            k_ring_func = None

L2_DIR = Path('data/l2')
L3_DIR = Path('data/l3')

logger = logging.getLogger('l3_multiscale')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def _prepare_arrest_flags(df: pd.DataFrame, arrest_col: Optional[str]) -> pd.Series:
    if arrest_col is None:
        return pd.Series(np.zeros(len(df), dtype=int), index=df.index)
    raw = pd.Series(df[arrest_col])
    try:
        normalized = raw.astype('boolean')
    except Exception:
        normalized = raw.fillna(False).isin([True, 'true', 'True', '1', 1])
        normalized = pd.Series(normalized, index=raw.index, dtype='boolean')
    filled = normalized.fillna(False)
    return filled.astype(int)


def aggregate_daily_counts(df: pd.DataFrame, hcol: str, arrest_col: Optional[str] = None) -> pd.DataFrame:
    subset = df[[hcol, 'date']].copy()
    subset = subset.dropna(subset=[hcol])
    if subset.empty:
        return pd.DataFrame(columns=[hcol, 'date', 'n_crimes', 'n_arrests'])
    subset['n_crime_unit'] = 1
    subset['arrest_flag'] = _prepare_arrest_flags(df, arrest_col).reindex(subset.index).fillna(0).astype(int)
    grouped = (
        subset.groupby([hcol, 'date'], observed=True)[['n_crime_unit', 'arrest_flag']]
        .sum()
        .rename(columns={'n_crime_unit': 'n_crimes', 'arrest_flag': 'n_arrests'})
        .reset_index()
    )
    return grouped


def aggregate_monthly_counts(daily_df: pd.DataFrame, hcol: str) -> pd.DataFrame:
    if daily_df.empty:
        return pd.DataFrame(columns=[hcol, 'month', 'n_crimes', 'n_arrests'])
    temp = daily_df.copy()
    temp['month'] = temp['date'].dt.to_period('M')
    monthly = (
        temp.groupby([hcol, 'month'], observed=True)[['n_crimes', 'n_arrests']]
        .sum()
        .reset_index()
    )
    monthly['month'] = monthly['month'].dt.to_timestamp()
    return monthly


def wilson_ci(k, n, z=1.96):
    # returns (low, high)
    if n == 0:
        return 0.0, 0.0
    p = k / n
    z2 = z * z
    denom = 1 + z2 / n
    centre = p + z2 / (2 * n)
    adj = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    low = (centre - adj) / denom
    high = (centre + adj) / denom
    low = max(0.0, low)
    high = min(1.0, high)
    return float(low), float(high)


def smoothed_rate(n, k, prior, alpha=5.0):
    if n <= 0:
        return float(prior)
    return float((alpha * prior + k) / (alpha + n))


def read_l2_month(year: int, month: int):
    path = L2_DIR / f'year={year}' / f'month={month:02d}' / f'features-{year}-{month:02d}.parquet'
    if not path.exists():
        logger.warning('Missing L2 partition: %s', path)
        return None
    df = pd.read_parquet(path, engine='pyarrow')
    df = l2mod._sanitize_and_cast_for_parquet(df, 9)
    return df


def aggregate_for_month(year: int, month: int, h3_resolutions=None, alpha=5.0, pool_k=1):
    if h3_resolutions is None:
        h3_resolutions = getattr(l2mod, 'H3_RESOLUTIONS', [7,8,9])

    df = read_l2_month(year, month)
    if df is None or df.empty:
        logger.info('No data for %s-%02d', year, month)
        return {}

    # ensure datetime and date fields
    df['date'] = pd.to_datetime(df['datetime']).dt.floor('D')

    results = {}

    for res in h3_resolutions:
        hcol = f'h3_r{res}'
        if hcol not in df.columns:
            logger.warning('H3 column %s not present; skipping', hcol)
            continue

        # detect id and arrest columns with common fallbacks
        id_candidates = ['id', 'case_number', 'incident_id', 'incident_number']
        arrest_candidates = ['arrest_made', 'arrest', 'arrested', 'was_arrested']
        id_col = next((c for c in id_candidates if c in df.columns), None)
        arrest_col = next((c for c in arrest_candidates if c in df.columns), None)

        grp = aggregate_daily_counts(df, hcol, arrest_col)

        if grp.empty:
            continue

        n_crimes = grp['n_crimes'].to_numpy(dtype=float)
        n_arrests = grp['n_arrests'].to_numpy(dtype=float)

        with np.errstate(divide='ignore', invalid='ignore'):
            raw_rate = np.divide(n_arrests, n_crimes, out=np.zeros_like(n_arrests, dtype=float), where=n_crimes > 0)
        grp['raw_rate'] = raw_rate
        grp['low_conf'] = grp['n_crimes'] < 5

        total_n = float(n_crimes.sum())
        total_k = float(n_arrests.sum())
        prior = float(total_k / total_n) if total_n > 0 else 0.0

        grp['smoothed_rate'] = np.where(
            grp['n_crimes'] > 0,
            (alpha * prior + grp['n_arrests']) / (alpha + grp['n_crimes']),
            float(prior)
        )

        def _wilson_interval(k_vals: np.ndarray, n_vals: np.ndarray, z: float = 1.96):
            low = np.zeros_like(k_vals, dtype=float)
            high = np.zeros_like(k_vals, dtype=float)
            mask = n_vals > 0
            if not np.any(mask):
                return low, high
            p = np.zeros_like(k_vals, dtype=float)
            p[mask] = k_vals[mask] / n_vals[mask]
            z2 = z ** 2
            denom = 1 + z2 / n_vals[mask]
            centre = p[mask] + z2 / (2 * n_vals[mask])
            adj = z * np.sqrt(p[mask] * (1 - p[mask]) / n_vals[mask] + z2 / (4 * n_vals[mask] ** 2))
            low_vals = (centre - adj) / denom
            high_vals = (centre + adj) / denom
            low[mask] = np.clip(low_vals, 0.0, 1.0)
            high[mask] = np.clip(high_vals, 0.0, 1.0)
            return low, high

        ci_low, ci_high = _wilson_interval(n_arrests, n_crimes)
        grp['ci_low'] = ci_low
        grp['ci_high'] = ci_high

        if h3 is not None and pool_k >= 1 and k_ring_func is not None:
            unique_cells = grp[hcol].dropna().unique()
            neighbor_edges = []
            for cell in unique_cells:
                try:
                    ring = k_ring_func(cell, pool_k)
                except Exception as exc:
                    logger.debug('k_ring failed for %s: %s', cell, exc)
                    ring = {cell}
                for neighbor in ring:
                    neighbor_edges.append((cell, neighbor))

            if neighbor_edges:
                neighbor_df = pd.DataFrame(neighbor_edges, columns=[hcol, 'neighbor'])
                expanded = grp.merge(neighbor_df, on=hcol, how='left')
                neighbor_counts = grp[[hcol, 'date', 'n_crimes', 'n_arrests']].rename(columns={hcol: 'neighbor'})
                expanded = expanded.merge(
                    neighbor_counts,
                    on=['date', 'neighbor'],
                    how='left',
                    suffixes=('', '_nbr')
                )
                expanded[['n_crimes_nbr', 'n_arrests_nbr']] = expanded[['n_crimes_nbr', 'n_arrests_nbr']].fillna(0)
                pooled = (
                    expanded.groupby(['date', hcol], observed=True)[['n_crimes_nbr', 'n_arrests_nbr']]
                    .sum()
                    .rename(columns={'n_crimes_nbr': 'pooled_n', 'n_arrests_nbr': 'pooled_k'})
                    .reset_index()
                )
                grp = grp.merge(pooled, on=['date', hcol], how='left')
                grp[['pooled_n', 'pooled_k']] = grp[['pooled_n', 'pooled_k']].fillna({'pooled_n': 0.0, 'pooled_k': 0.0})
                with np.errstate(divide='ignore', invalid='ignore'):
                    grp['pooled_rate'] = np.divide(
                        grp['pooled_k'],
                        grp['pooled_n'],
                        out=np.full(len(grp), prior, dtype=float),
                        where=grp['pooled_n'] > 0
                    )
                grp['pooled_smoothed'] = np.where(
                    grp['pooled_n'] > 0,
                    (alpha * prior + grp['pooled_k']) / (alpha + grp['pooled_n']),
                    grp['smoothed_rate']
                )
            else:
                grp['pooled_n'] = 0.0
                grp['pooled_k'] = 0.0
                grp['pooled_rate'] = prior
                grp['pooled_smoothed'] = grp['smoothed_rate']
        else:
            grp['pooled_n'] = 0.0
            grp['pooled_k'] = 0.0
            grp['pooled_rate'] = prior
            grp['pooled_smoothed'] = grp['smoothed_rate']

        # write daily aggregates partitioned by year/month
        out_dir = L3_DIR / f'res={res}' / f'year={year}' / f'month={month:02d}'
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f'l3-aggregates-{res}-{year}-{month:02d}.parquet'
        grp.to_parquet(out_file, index=False, engine='pyarrow')
        logger.info('Wrote L3 daily aggregates %s rows=%d', out_file, len(grp))

        # also produce monthly aggregates for the same res
        grp_month = aggregate_monthly_counts(grp, hcol)
        if 'month' in grp_month.columns:
            grp_month = grp_month.drop(columns=['month'])
        grp_month['raw_rate'] = grp_month.apply(lambda r: float(r['n_arrests']/r['n_crimes']) if r['n_crimes']>0 else 0.0, axis=1)
        grp_month['smoothed_rate'] = grp_month.apply(lambda r: smoothed_rate(r['n_crimes'], r['n_arrests'], prior, alpha=alpha), axis=1)
        month_out_dir = L3_DIR / f'res={res}' / f'year={year}' / f'month={month:02d}'
        month_out_file = month_out_dir / f'l3-aggregates-{res}-{year}-{month:02d}-monthly.parquet'
        grp_month.to_parquet(month_out_file, index=False, engine='pyarrow')
        logger.info('Wrote L3 monthly aggregates %s rows=%d', month_out_file, len(grp_month))

        results[res] = {'daily': out_file, 'monthly': month_out_file, 'rows_daily': len(grp), 'rows_monthly': len(grp_month)}

    return results


def find_year_months(start_year=None):
    years = sorted([p for p in L2_DIR.glob('year=*') if p.is_dir()])
    if start_year is not None:
        years = [y for y in years if int(y.name.split('=')[1]) >= start_year]
    ym = []
    for y in years:
        year = int(y.name.split('=')[1])
        for m in sorted(y.glob('month=*')):
            month = int(m.name.split('=')[1])
            ym.append((year, month))
    return ym


def main():
    if len(sys.argv) >= 3:
        year = int(sys.argv[1]); month = int(sys.argv[2])
        aggregate_for_month(year, month, h3_resolutions=getattr(l2mod, 'H3_RESOLUTIONS', [7,8,9]))
    elif len(sys.argv) == 2:
        year = int(sys.argv[1])
        ym = [ym for ym in find_year_months(year) if ym[0]==year]
        for y,m in ym:
            aggregate_for_month(y,m, h3_resolutions=getattr(l2mod, 'H3_RESOLUTIONS', [7,8,9]))
    else:
        # process all known partitions
        for y,m in find_year_months(None):
            aggregate_for_month(y,m, h3_resolutions=getattr(l2mod, 'H3_RESOLUTIONS', [7,8,9]))


if __name__ == '__main__':
    main()
