"""Precompute parent→child H3 mappings for drill-down interaction.

Usage:
    python scripts/precompute_h3_drilldown.py

Outputs:
    - data/h3_mappings/parents_res_9_to_10.parquet
    - data/h3_mappings/parents_res_8_to_9.parquet
    - ... etc depending on configured combinations
"""
import json
from pathlib import Path
from typing import Dict

import pandas as pd
import h3

# configure parent/child resolution pairs we need for drill-down
RESOLUTION_PAIRS = [
    (9, 10),  # city view at r9 -> drill to r10
    (8, 9),
    (7, 8),
    (6, 7),
]

OUTPUT_DIR = Path('data/h3_mappings')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

all_rows = []
for parent_res, child_res in RESOLUTION_PAIRS:
    rows = []
    seen_parents = set()
    # derive parents from existing child cells across all L3 partitions
    l3_root = Path('data/l3') / f'res={child_res}'
    if not l3_root.exists():
        continue
    for path in l3_root.rglob(f'l3-aggregates-{child_res}-*.parquet'):
        df = pd.read_parquet(path, columns=[f'h3_r{child_res}'])
        df = df.dropna().drop_duplicates()
        for child in df[f'h3_r{child_res}']:
            parent = h3.cell_to_parent(child, parent_res)
            rows.append({'child_res': child_res, 'parent_res': parent_res, 'parent': parent, 'child': child})
            seen_parents.add(parent)
    if rows:
        out_df = pd.DataFrame(rows).drop_duplicates()
        out_path = OUTPUT_DIR / f'parents_res_{parent_res}_to_{child_res}.parquet'
        out_df.to_parquet(out_path, index=False)
        print(f'Wrote {len(out_df)} rows to {out_path}')
        all_rows.append(out_df)

if all_rows:
    union_df = pd.concat(all_rows, ignore_index=True)
    union_df.to_parquet(OUTPUT_DIR / 'parent_child_mapping.parquet', index=False)
    print('Combined mapping rows:', len(union_df))
