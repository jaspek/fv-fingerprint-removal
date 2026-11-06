"""Concatenate result tables that match a pattern into one CSV; duplicate rows (repeated baselines) are dropped.

Example:
  python merge_csv.py --pattern "<work>/results_pad_train_v210_PLUS_development_*.csv" --out <work>/results_pad_train_v210_PLUS_development.csv
"""
import argparse
import glob

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pattern', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    files = sorted(glob.glob(a.pattern))
    if not files:
        raise SystemExit(f'no files match {a.pattern}')
    table = pd.concat([pd.read_csv(f) for f in files]).drop_duplicates()
    table.to_csv(a.out, index=False)
    print(table.to_string(index=False))
    print('->', a.out, f'({len(files)} files)')


if __name__ == '__main__':
    main()
