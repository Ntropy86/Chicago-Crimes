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

        if id_col is not None:
            n_crime_agg = (id_col, 'count')
        else:
            # fallback to size if no id column
            n_crime_agg = (df.columns[0], 'size')

        def arrest_counter(s):
            if arrest_col is None:
                return 0
            # coerce truthy values to boolean
            return int(pd.Series(s).eq(True).sum())

        grp = df.groupby([hcol, 'date']).agg(
            n_crimes = n_crime_agg,
            n_arrests = (arrest_col if arrest_col is not None else df.columns[0], lambda s: arrest_counter(s))
        ).reset_index()

        # compute raw rate, low confidence flag
        grp['raw_rate'] = grp.apply(lambda r: float(r['n_arrests'] / r['n_crimes']) if r['n_crimes']>0 else 0.0, axis=1)
        grp['low_conf'] = grp['n_crimes'] < 5

        # global prior for this month/resolution (avoid division by zero)
        total_n = grp['n_crimes'].sum()
        total_k = grp['n_arrests'].sum()
        prior = float(total_k / total_n) if total_n > 0 else 0.0

        # smoothed rate and CI
        grp['smoothed_rate'] = grp.apply(lambda r: smoothed_rate(r['n_crimes'], r['n_arrests'], prior, alpha=alpha), axis=1)
        cis = grp.apply(lambda r: wilson_ci(r['n_arrests'], r['n_crimes']), axis=1)
        grp[['ci_low','ci_high']] = pd.DataFrame(cis.tolist(), index=grp.index)

        # neighbor pooling (k-ring) for low confidence cells
        if h3 is not None and pool_k >= 1 and k_ring_func is not None:
            # build counts map for quick lookup
            counts_map = {}
            arrests_map = {}
            for _, row in grp.iterrows():
                hid = row[hcol]
                if pd.isna(hid):
                    continue
                try:
                    k = int(row['n_crimes'])
                except Exception:
                    k = 0
                try:
                    a = int(row['n_arrests'])
                except Exception:
                    a = 0
                counts_map[str(hid)] = k
                arrests_map[str(hid)] = a

            def pooled_counts(hid):
                if pd.isna(hid):
                    return 0,0
                try:
                    ring = k_ring_func(hid, pool_k)
                except Exception:
                    # if k_ring fails for this hid, return self counts
                    return counts_map.get(str(hid), 0), arrests_map.get(str(hid), 0)
                n = sum(counts_map.get(r, 0) for r in ring)
                k_ = sum(arrests_map.get(r, 0) for r in ring)
                return int(n), int(k_)

            pooled = grp[hcol].apply(pooled_counts)
            grp[['pooled_n','pooled_k']] = pd.DataFrame(pooled.tolist(), index=grp.index)
            grp['pooled_rate'] = grp.apply(lambda r: float(r['pooled_k']/r['pooled_n']) if r['pooled_n']>0 else prior, axis=1)
            grp['pooled_smoothed'] = grp.apply(lambda r: smoothed_rate(r['pooled_n'], r['pooled_k'], prior, alpha=alpha), axis=1)
        else:
            grp['pooled_n'] = 0
            grp['pooled_k'] = 0
            grp['pooled_rate'] = prior
            grp['pooled_smoothed'] = grp['smoothed_rate']

        # write daily aggregates partitioned by year/month
        out_dir = L3_DIR / f'res={res}' / f'year={year}' / f'month={month:02d}'
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f'l3-aggregates-{res}-{year}-{month:02d}.parquet'
        grp.to_parquet(out_file, index=False, engine='pyarrow')
        logger.info('Wrote L3 daily aggregates %s rows=%d', out_file, len(grp))

        # also produce monthly aggregates for the same res
        grp_month = grp.groupby(hcol).agg(
            n_crimes = ('n_crimes','sum'),
            n_arrests = ('n_arrests','sum')
        ).reset_index()
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
