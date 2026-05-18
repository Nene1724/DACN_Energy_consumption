#!/usr/bin/env python3
"""Create filtered copies of datasets excluding models listed in the 169 list.

Usage:
  python exclude_169_models.py --source data/benchmark_reports.csv --out data/benchmark_reports.filtered.csv
"""
import argparse
import csv
from pathlib import Path

def load_169_list(path: Path):
    names = set()
    with path.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            nm = r.get('model') or r.get('name')
            if nm:
                names.add(nm.strip())
    return names

def filter_file(source: Path, out: Path, exclude_names: set):
    with source.open('r', encoding='utf-8', newline='') as inf:
        reader = csv.DictReader(inf)
        rows = [r for r in reader if (r.get('model') or r.get('name') or '').strip() not in exclude_names]
        if not rows:
            print('Warning: no rows remaining after filtering')
        fieldnames = reader.fieldnames or []

    with out.open('w', encoding='utf-8', newline='') as outf:
        writer = csv.DictWriter(outf, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--list', default='ml-controller/test/169_models_filtered.csv', help='CSV list of models to exclude')
    p.add_argument('--source', required=True, help='Source CSV to filter')
    p.add_argument('--out', required=True, help='Output filtered CSV path')
    args = p.parse_args()

    exclude = load_169_list(Path(args.list))
    print(f'Excluding {len(exclude)} models')
    filter_file(Path(args.source), Path(args.out), exclude)
    print(f'Wrote filtered file to {args.out}')

if __name__ == '__main__':
    main()
