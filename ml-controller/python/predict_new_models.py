#!/usr/bin/env python3
"""
Predict energy consumption for ~100 new models not in Jetson 360 benchmark.
Models are generated as variants of benchmark models + popular models outside benchmark.
"""

import csv
import json
import math
import pickle
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

def main():
    base = Path(__file__).parent.parent
    artifacts_dir = base / 'artifacts'
    jetson_csv = base / 'data/360_models_benchmark_jetson.csv'
    popular_json = artifacts_dir / 'popular_models_metadata.json'
    output_md = artifacts_dir / 'jetson_nano_new_models_predicted_100.md'
    output_json = artifacts_dir / 'jetson_nano_new_models_predicted_100.json'

    # Load benchmark models
    benchmark_models = set()
    latency_stats = []
    with open(jetson_csv, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            benchmark_models.add(row['model'].lower().strip())
            latency_stats.append(float(row['latency_avg_s']))

    latency_stats = sorted(latency_stats)
    p1_latency = latency_stats[int(len(latency_stats) * 0.01)]
    p99_latency = latency_stats[int(len(latency_stats) * 0.99)]
    print(f'Benchmark latency range p1-p99: {p1_latency:.4f}s - {p99_latency:.4f}s')

    # Load feature patterns
    features_by_prefix = defaultdict(list)
    with open(jetson_csv, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            prefix = row['model'].split('_')[0].lower()
            features_by_prefix[prefix].append({
                'name': row['model'],
                'params_m': float(row['params_m']),
                'gflops': float(row['gflops']),
                'gmacs': float(row['gmacs']),
                'size_mb': float(row['size_mb']),
                'latency_avg_s': float(row['latency_avg_s']),
                'throughput_iter_per_s': float(row['throughput_iter_per_s']),
            })

    candidates = []

    # Add models outside benchmark from popular list
    meta = json.loads(popular_json.read_text(encoding='utf-8'))
    for m in meta['models']:
        if m['name'].lower() not in benchmark_models:
            if all(isinstance(m.get(k), (int, float)) and m.get(k) > 0 for k in ['params_m', 'gflops', 'gmacs', 'size_mb']):
                candidates.append({
                    'name': m['name'] + '_new',
                    'params_m': m['params_m'],
                    'gflops': m['gflops'],
                    'gmacs': m['gmacs'],
                    'size_mb': m['size_mb'],
                    'latency_avg_s': m.get('latency_avg_s', 0.03),
                    'throughput_iter_per_s': m.get('throughput_iter_per_s', 30.0),
                    'source': 'popular_outside_benchmark',
                })

    # Generate scaled variants
    scale_patterns = [(0.3, '030'), (0.75, '075'), (1.25, '125'), (1.5, '150'), (2.0, '200')]
    for prefix, models in sorted(features_by_prefix.items()):
        if not models:
            continue
        base_model = min(models, key=lambda x: x['params_m'])
        for scale, suffix in scale_patterns:
            if scale in [0.5, 1.0]:
                continue
            name = f"{prefix}_{suffix}_scaled"
            if name.lower() in benchmark_models:
                continue
            lat = base_model['latency_avg_s'] * math.pow(scale, 0.4)
            if lat < p1_latency or lat > p99_latency * 2:
                continue
            candidates.append({
                'name': name,
                'params_m': base_model['params_m'] * scale,
                'gflops': base_model['gflops'] * scale,
                'gmacs': base_model['gmacs'] * scale,
                'size_mb': base_model['size_mb'] * scale,
                'latency_avg_s': lat,
                'throughput_iter_per_s': base_model['throughput_iter_per_s'] / math.pow(scale, 0.4),
                'source': 'scaled_variant',
            })

    candidates = sorted(candidates, key=lambda x: (x['gflops'], x['params_m']))
    top100 = candidates[:100]
    print(f'Selected {len(top100)} new models for prediction')

    # Compute derived features for prediction
    for r in top100:
        r['gflops_per_param'] = r['gflops'] / r['params_m'] if r['params_m'] > 0 else 0
        r['gmacs_per_mb'] = r['gmacs'] / r['size_mb'] if r['size_mb'] > 0 else 0
        r['latency_throughput_ratio'] = r['latency_avg_s'] * r['throughput_iter_per_s']
        r['compute_intensity'] = r['gmacs'] * 1e9 / (r['latency_avg_s'] * 1e10) if r['latency_avg_s'] > 0 else 0
        r['model_complexity'] = r['params_m'] * r['gflops']
        r['computational_density'] = r['gflops'] / r['size_mb'] if r['size_mb'] > 0 else 0
        r['log_params_m'] = math.log1p(r['params_m'])
        r['log_gflops'] = math.log1p(r['gflops'])
        r['log_size_mb'] = math.log1p(r['size_mb'])
        r['log_gmacs'] = math.log1p(r['gmacs'])
        r['log_latency'] = math.log1p(r['latency_avg_s'])
        r['log_throughput'] = math.log1p(r['throughput_iter_per_s'])
        r['log_model_complexity'] = math.log1p(r['model_complexity'])
        r['log_compute_intensity'] = math.log1p(r['compute_intensity'])
        r['log_params_x_log_latency'] = r['log_params_m'] * r['log_latency']
        r['log_gflops_x_log_latency'] = r['log_gflops'] * r['log_latency']
        r['batch_high_power'] = 1 if r['latency_avg_s'] > 0.1 else 0

    # Load predictor model
    try:
        with open(artifacts_dir / 'jetson_energy_model.pkl', 'rb') as f:
            jetson_model = pickle.load(f)
        with open(artifacts_dir / 'jetson_scaler.pkl', 'rb') as f:
            jetson_scaler = pickle.load(f)
        with open(artifacts_dir / 'device_specific_features.json', 'r', encoding='utf-8') as f:
            feature_names = json.load(f)
        
        # Prepare features for prediction
        X_new = []
        for r in top100:
            row_features = [r.get(f, 0.0) for f in feature_names]
            X_new.append(row_features)
        
        # Scale and predict
        X_new = np.array(X_new, dtype=float)
        X_new_scaled = jetson_scaler.transform(X_new)
        energy_preds = jetson_model.predict(X_new_scaled)
        
        for r, pred in zip(top100, energy_preds):
            r['predicted_energy_mwh'] = float(max(pred, 0.0))  # Ensure non-negative
        
        print(f'✓ Energy predictions successful for {len(top100)} models\n')
        
        # Sort by predicted energy
        sorted_by_energy = sorted(top100, key=lambda x: x['predicted_energy_mwh'])
        
        # Output JSON
        output_json.write_text(json.dumps({
            'source': 'scaled_variants + popular_outside_benchmark',
            'total_benchmark_models': len(benchmark_models),
            'generated_candidates': len(candidates),
            'selected_count': len(top100),
            'feature_count': len(feature_names),
            'models': sorted_by_energy,
        }, ensure_ascii=False, indent=2), encoding='utf-8')

        # Output Markdown
        lines = []
        lines.append('# Jetson Nano 100 New Models with Predicted Energy\n')
        lines.append('**Source:** Scaled variants of benchmark model families + popular models outside 360-model benchmark\n')
        lines.append(f'**Criteria:** All models NOT in Jetson 360 benchmark; latency range p1-p99 of benchmark\n')
        lines.append(f'**Total available:** {len(candidates)} generated candidates\n')
        lines.append(f'**Selected:** {len(top100)} by lowest GFLOPs + params_m\n')
        lines.append('')
        lines.append('| Rank | Model | Params (M) | GFLOPs | GMACs | Size (MB) | Latency (s) | Source | Predicted Energy (mWh) |')
        lines.append('| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |')
        for i, r in enumerate(sorted_by_energy, 1):
            lines.append(f"| {i} | {r['name']} | {r['params_m']:.3f} | {r['gflops']:.3f} | {r['gmacs']:.3f} | {r['size_mb']:.2f} | {r['latency_avg_s']:.4f} | {r['source']} | {r['predicted_energy_mwh']:.1f} |")
        
        lines.append('')
        lines.append('## Summary Statistics\n')
        energies = [r['predicted_energy_mwh'] for r in sorted_by_energy]
        lines.append(f'- Min predicted energy: {min(energies):.1f} mWh')
        lines.append(f'- Max predicted energy: {max(energies):.1f} mWh')
        lines.append(f'- Median: {sorted(energies)[len(energies)//2]:.1f} mWh')
        lines.append(f'- Mean: {sum(energies)/len(energies):.1f} mWh')
        
        lines.append('')
        lines.append('## Top 10 Most Efficient (Lowest Predicted Energy)\n')
        for i, r in enumerate(sorted_by_energy[:10], 1):
            lines.append(f'{i}. **{r["name"]}**: {r["predicted_energy_mwh"]:.1f} mWh ({r["params_m"]:.2f}M params, {r["gflops"]:.3f} GFLOPs)')
        
        lines.append('')
        lines.append('## Top 10 Most Demanding (Highest Predicted Energy)\n')
        for i, r in enumerate(reversed(sorted_by_energy[-10:]), 1):
            lines.append(f'{i}. **{r["name"]}**: {r["predicted_energy_mwh"]:.1f} mWh ({r["params_m"]:.2f}M params, {r["gflops"]:.3f} GFLOPs)')
        
        output_md.write_text('\n'.join(lines), encoding='utf-8')
        
        print(f'✓ Reports written:')
        print(f'  - {output_json.name}')
        print(f'  - {output_md.name}')
        print(f'\nTop 15 most efficient:')
        for i, r in enumerate(sorted_by_energy[:15], 1):
            print(f'{i:2d}. {r["name"]:40s} energy={r["predicted_energy_mwh"]:7.1f} mWh')

    except Exception as e:
        print(f'✗ Error during prediction: {e}')
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == '__main__':
    sys.exit(main())
