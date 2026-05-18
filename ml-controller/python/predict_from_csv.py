#!/usr/bin/env python3
"""Batch energy prediction script.

Reads a CSV of models (expects columns: model, params_m, gflops, gmacs, size_mb, input_size,
Latency (s),Throughput (iter/s)) and runs the production `EnergyPredictorService` predict
flow (device-aware routing). Outputs results to JSON and CSV.

Usage:
  python predict_from_csv.py ml-controller/test/169_models_filtered.csv --device jetson_nano
"""
import argparse
import csv
import json
import os
from pathlib import Path

from energy_predictor_service import EnergyPredictorService


def read_input(csv_path: Path):
    rows = []
    with csv_path.open('r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            # Normalize column names used by predictor
            try:
                params_m = float(r.get('params_m') or r.get('params') or 0)
                gflops = float(r.get('gflops') or 0)
                gmacs = float(r.get('gmacs') or 0)
                size_mb = float(r.get('size_mb') or 0)
                latency = float(r.get('Latency (s)') or r.get('latency_avg_s') or 0)
                throughput = float(r.get('Throughput (iter/s)') or r.get('throughput_iter_per_s') or 0)
            except Exception:
                params_m = float(r.get('params_m') or 0)
                gflops = float(r.get('gflops') or 0)
                gmacs = float(r.get('gmacs') or 0)
                size_mb = float(r.get('size_mb') or 0)
                latency = float(r.get('Latency (s)') or 0)
                throughput = float(r.get('Throughput (iter/s)') or 0)

            rows.append({
                'model': r.get('model') or r.get('name'),
                'params_m': params_m,
                'gflops': gflops,
                'gmacs': gmacs,
                'size_mb': size_mb,
                'latency_avg_s': latency,
                'throughput_iter_per_s': throughput,
            })
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('input_csv', help='CSV file with model rows')
    p.add_argument('--device', default='jetson_nano', help='Target device type for routing')
    p.add_argument('--artifacts', default=str(Path(__file__).parent.parent / 'artifacts'), help='Artifacts directory')
    p.add_argument('--out', default='predictions_output', help='Output folder prefix')
    args = p.parse_args()

    input_csv = Path(args.input_csv)
    if not input_csv.exists():
        raise SystemExit(f'Input CSV not found: {input_csv}')

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read CSV
    payloads = read_input(input_csv)

    # Attach device type
    for pld in payloads:
        pld['device_type'] = args.device

    # Initialize service
    artifacts_dir = os.path.abspath(args.artifacts)
    svc = EnergyPredictorService(artifacts_dir)

    # Run predictions
    results = svc.predict(payloads)

    # Write JSON and CSV outputs
    json_out = out_dir / f'{input_csv.stem}_{args.device}_predictions.json'
    csv_out = out_dir / f'{input_csv.stem}_{args.device}_predictions.csv'

    with json_out.open('w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # CSV: flatten features_used into columns
    fieldnames = ['model_name', 'device_type', 'prediction_mwh', 'ci_lower_mwh', 'ci_upper_mwh', 'model_used', 'mape_pct', 'error']
    # collect feature names if available
    extra_fields = []
    for r in results:
        if isinstance(r.get('features_used'), dict):
            extra_fields = list(r['features_used'].keys())
            break
    if extra_fields:
        fieldnames += extra_fields

    with csv_out.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {k: r.get(k) for k in fieldnames}
            # if features present, copy them
            if 'features_used' in r and isinstance(r['features_used'], dict):
                for fk, fv in r['features_used'].items():
                    row[fk] = fv
            writer.writerow(row)

    print(f'Wrote: {json_out}\n      {csv_out}')


if __name__ == '__main__':
    main()
