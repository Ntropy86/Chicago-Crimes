"""Clustering prototype: UMAP + HDBSCAN

Reads a month of L2, computes a low-dimensional embedding with UMAP, runs
HDBSCAN, and writes cluster assignments back to parquet for analysis.

Usage:
    python src/l3_clustering_prototype.py 2024 9

This file is a prototype and expects packages: umap-learn, hdbscan, scikit-learn.
"""
from pathlib import Path
import sys
import logging

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

L2_DIR = Path('data/l2')
OUT_DIR = Path('data/l3/clusters')

logger = logging.getLogger('l3_clustering')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def read_l2_month(year: int, month: int):
    path = L2_DIR / f'year={year}' / f'month={month:02d}' / f'features-{year}-{month:02d}.parquet'
    if not path.exists():
        logger.error('Missing L2 partition: %s', path)
        return None
    return pd.read_parquet(path, engine='pyarrow')


def run_clustering(year:int, month:int, res:int=9):
    df = read_l2_month(year, month)
    if df is None or df.empty:
        return

    hcol = f'h3_r{res}'
    if hcol not in df.columns:
        logger.error('Missing H3 column %s', hcol)
        return

    # pick features for embedding: cyclical + maybe counts
    feat_cols = [c for c in df.columns if c.endswith('_sin') or c.endswith('_cos')]
    if not feat_cols:
        logger.error('No cyclical features found for embedding')
        return

    X = df[feat_cols].fillna(0).to_numpy()
    X = StandardScaler().fit_transform(X)

    try:
        import umap
        import hdbscan
    except Exception as e:
        logger.error('UMAP/HDBSCAN not installed: %s', e)
        return

    reducer = umap.UMAP(n_components=2, random_state=42)
    emb = reducer.fit_transform(X)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=50)
    labels = clusterer.fit_predict(emb)

    out = df[[hcol]].copy()
    out['umap_x'] = emb[:,0]
    out['umap_y'] = emb[:,1]
    out['cluster'] = labels

    out_dir = OUT_DIR / f'res={res}' / f'year={year}' / f'month={month:02d}'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f'clusters-{res}-{year}-{month:02d}.parquet'
    out.to_parquet(out_file, engine='pyarrow', index=False)
    logger.info('Wrote cluster assignments %s rows=%d', out_file, len(out))


if __name__ == '__main__':
    if len(sys.argv) >= 3:
        run_clustering(int(sys.argv[1]), int(sys.argv[2]))
    else:
        print('Usage: python src/l3_clustering_prototype.py YEAR MONTH')
