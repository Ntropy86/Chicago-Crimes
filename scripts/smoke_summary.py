#!/usr/bin/env python3
"""Generate a smoke summary CSV for L2 and L3 partitions.

Writes: reports/smoke_summary.csv
"""
from pathlib import Path
import pandas as pd
import sys

ROOT = Path(__file__).resolve().parents[1]
L2_DIR = ROOT / 'data' / 'l2'
L3_DIR = ROOT / 'data' / 'l3'
OUT = ROOT / 'reports' / 'smoke_summary.csv'

def scan_l2():
    rows = []
    if not L2_DIR.exists():
        print('No data/l2 directory found', file=sys.stderr)
        return rows
    for year_dir in sorted(L2_DIR.glob('year=*')):
        for month_dir in sorted(year_dir.glob('month=*')):
            # find features-YYYY-MM.parquet
            for p in month_dir.glob('features-*.parquet'):
                try:
                    df = pd.read_parquet(p)
                    rows.append({
                        'type':'l2',
                        'path': str(p.relative_to(ROOT)),
                        'year': year_dir.name.split('=')[1],
                        'month': month_dir.name.split('=')[1],
                        'exists': True,
                        'rows': len(df),
                        'h3_r7': 'h3_r7' in df.columns,
                        'h3_r8': 'h3_r8' in df.columns,
                        'h3_r9': 'h3_r9' in df.columns,
                    })
                except Exception as e:
                    rows.append({
                        'type':'l2',
                        'path': str(p.relative_to(ROOT)),
                        'year': year_dir.name.split('=')[1],
                        'month': month_dir.name.split('=')[1],
                        'exists': False,
                        'rows': None,
                        'h3_r7': False,
                        'h3_r8': False,
                        'h3_r9': False,
                        'error': str(e),
                    })
    return rows

def scan_l3():
    rows = []
    if not L3_DIR.exists():
        print('No data/l3 directory found', file=sys.stderr)
        return rows
    for res_dir in sorted(L3_DIR.glob('res=*')):
        res = res_dir.name.split('=')[1]
        for year_dir in sorted(res_dir.glob('year=*')):
            for month_dir in sorted(year_dir.glob('month=*')):
                for p in month_dir.glob('l3-aggregates-*.parquet'):
                    try:
                        df = pd.read_parquet(p)
                        rows.append({
                            'type':'l3',
                            'res': int(res),
                            'path': str(p.relative_to(ROOT)),
                            'year': year_dir.name.split('=')[1],
                            'month': month_dir.name.split('=')[1],
                            'exists': True,
                            'rows': len(df),
                        })
                    except Exception as e:
                        rows.append({
                            'type':'l3',
                            'res': int(res),
                            'path': str(p.relative_to(ROOT)),
                            'year': year_dir.name.split('=')[1],
                            'month': month_dir.name.split('=')[1],
                            'exists': False,
                            'rows': None,
                            'error': str(e),
                        })
    return rows

def main():
    L2_rows = scan_l2()
    L3_rows = scan_l3()
    all_rows = L2_rows + L3_rows
    if not all_rows:
        print('No L2/L3 files found; nothing to report.', file=sys.stderr)
        return 2
    df = pd.DataFrame(all_rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print('Wrote', OUT)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
