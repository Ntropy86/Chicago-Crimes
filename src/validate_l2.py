"""Small validation script for L2 outputs.

Reads a single L2 partition, sanitizes via l2_features helper, writes a temp parquet,
and attempts to read it with pyarrow to validate schema determinism.
"""
from pathlib import Path
import sys
import pandas as pd
import pyarrow.parquet as pq

import importlib.util
spec = importlib.util.spec_from_file_location('l2mod','src/l2_features.py')
l2mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(l2mod)

def validate_sample(year=2024, month=9):
    path = Path(f'data/l2/year={year}/month={month:02d}/features-{year}-{month:02d}.parquet')
    if not path.exists():
        print('No L2 sample found at', path)
        return 2
    df = pd.read_parquet(path, engine='pyarrow')
    print('Loaded sample shape', df.shape)
    san = l2mod._sanitize_and_cast_for_parquet(df, 9)
    tmp = Path('temp/validate-l2-sanitized.parquet')
    tmp.parent.mkdir(parents=True, exist_ok=True)
    san.to_parquet(tmp, engine='pyarrow', index=False)
    try:
        t = pq.read_table(str(tmp))
        print('pyarrow read_table OK; schema:')
        print(t.schema)
    except Exception as e:
        print('pyarrow read_table failed:', e)
        return 1
    print('Validation succeeded')
    return 0

if __name__ == '__main__':
    sys.exit(validate_sample())
