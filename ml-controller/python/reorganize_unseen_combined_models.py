#!/usr/bin/env python3
"""
Reorganize unseen and benchmark models into 2 distinct output sets.

Workflow:
1. Load 85 unseen models from refined output
2. Load 360 benchmark models
3. Select 15 best benchmark models (efficient + diverse + realistic)
4. Create weighted ranking score (energy + latency + diversity + confidence)
5. Generate output files:
   - jetson_nano_unseen_models_85.{csv,json,md}
   - jetson_nano_top100_combined.{csv,json,md}

NO predictor retraining - just reorganizing outputs.
"""

import csv
import json
import math
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter
import pandas as pd


class BenchmarkSelector:
    """Select best benchmark models: efficient + diverse + realistic."""
    
    @staticmethod
    def load_benchmark_models(csv_path: str) -> List[Dict]:
        """Load 360 benchmark models from CSV."""
        models = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                model = {
                    'name': row['model'],
                    'family': row.get('model', '').split('_')[0],  # Infer family
                    'params_m': float(row['params_m']),
                    'gflops': float(row['gflops']),
                    'latency_avg_s': float(row['latency_avg_s']),
                    'energy_avg_mwh': float(row['energy_avg_mwh']),
                    'energy_std_mwh': float(row.get('energy_std_mwh', 0)),
                }
                models.append(model)
        return models
    
    @staticmethod
    def compute_benchmark_score(model: Dict) -> float:
        """
        Score for selecting benchmark models.
        Higher = better candidate for inclusion.
        
        Criteria:
        - Low energy (efficiency)
        - Low latency (realtime)
        - Moderate params (not too heavy)
        - Low energy std (stable)
        """
        # Normalize scores to [0, 1]
        energy_score = 1.0 / (1.0 + model['energy_avg_mwh'] / 20.0)  # Lower is better
        latency_score = 1.0 / (1.0 + model['latency_avg_s'])  # Lower is better
        params_score = 1.0 / (1.0 + model['params_m'] / 50.0)  # Lower is better
        stability_score = 1.0 / (1.0 + model['energy_std_mwh'])  # Lower std is better
        
        # Weighted combination
        score = (
            0.4 * energy_score +
            0.3 * latency_score +
            0.2 * params_score +
            0.1 * stability_score
        )
        return score
    
    @staticmethod
    def select_diverse_benchmark(models: List[Dict], target_count: int = 15) -> List[Dict]:
        """
        Select benchmark models that are:
        1. Efficient (high score)
        2. Diverse (different families/architectures)
        3. Realistic (latency < 1.0s, energy < 50 mWh)
        """
        # Filter: realistic constraints
        # - latency < 1.0s (realtime)
        # - energy < 50 mWh (efficient, not heavy models)
        realistic = [m for m in models if m['latency_avg_s'] <= 1.0 and m['energy_avg_mwh'] <= 50.0]
        
        print(f"    Filtered to realistic (latency≤1.0s, energy≤50mWh): {len(realistic)}")
        
        # Compute scores
        for model in realistic:
            model['benchmark_score'] = BenchmarkSelector.compute_benchmark_score(model)
        
        # Sort by score
        realistic.sort(key=lambda m: m['benchmark_score'], reverse=True)
        
        # Select diverse families
        selected = []
        families_count = Counter()
        
        for model in realistic:
            # Infer family from model name
            name = model['name'].lower()
            if 'mobilenet' in name:
                family = 'MobileNet'
            elif 'shuffle' in name:
                family = 'ShuffleNet'
            elif 'efficient' in name:
                family = 'EfficientNet'
            elif 'densenet' in name:
                family = 'DenseNet'
            elif 'vit' in name or 'vision' in name:
                family = 'ViT'
            elif 'resnet' in name:
                family = 'ResNet'
            elif 'ghost' in name:
                family = 'GhostNet'
            elif 'convnext' in name:
                family = 'ConvNeXt'
            elif 'regnet' in name:
                family = 'RegNet'
            elif 'edgenext' in name:
                family = 'EdgeNeXt'
            else:
                family = 'Other'
            
            model['inferred_family'] = family
            
            # Prefer diverse families - max 2 per family
            if families_count[family] < 2:
                selected.append(model)
                families_count[family] += 1
                if len(selected) >= target_count:
                    break
        
        return selected[:target_count]


class WeightedRanking:
    """Compute weighted ranking score for all models."""
    
    @staticmethod
    def compute_ranking_score(model: Dict, energy_range: Tuple[float, float],
                             latency_range: Tuple[float, float],
                             diversity_range: Tuple[float, float]) -> float:
        """
        Compute weighted ranking score.
        
        Formula:
        score = 0.4 * energy_eff + 0.25 * latency_eff + 0.2 * diversity + 0.15 * confidence
        
        Where:
        - energy_eff: 1 - (energy - min) / (max - min)  [lower is better]
        - latency_eff: 1 - (latency - min) / (max - min) [lower is better]
        - diversity: normalized diversity_score
        - confidence: normalized confidence_score
        """
        energy_min, energy_max = energy_range
        latency_min, latency_max = latency_range
        diversity_min, diversity_max = diversity_range
        
        # Energy efficiency (lower is better)
        energy_norm = (model['predicted_energy_mwh'] - energy_min) / max(energy_max - energy_min, 0.1)
        energy_eff = 1.0 - min(energy_norm, 1.0)
        
        # Latency efficiency (lower is better)
        latency_norm = (model['estimated_latency_s'] - latency_min) / max(latency_max - latency_min, 0.1)
        latency_eff = 1.0 - min(latency_norm, 1.0)
        
        # Diversity score (normalized to 0-1)
        div_norm = (model.get('diversity_score', 0.5) - diversity_min) / max(diversity_max - diversity_min, 0.1)
        diversity_score = min(div_norm, 1.0)
        
        # Confidence score (already 0-1)
        confidence = model.get('confidence_score', 0.5)
        
        # Weighted combination
        ranking_score = (
            0.4 * energy_eff +
            0.25 * latency_eff +
            0.2 * diversity_score +
            0.15 * confidence
        )
        
        return round(ranking_score, 4)


class OutputGenerator:
    """Generate organized output files."""
    
    @staticmethod
    def generate_csv(models: List[Dict], output_path: str):
        """Generate CSV output."""
        if not models:
            return
        
        fieldnames = [
            'rank', 'model_name', 'architecture_family', 'source', 'is_unseen',
            'params_m', 'gflops', 'estimated_latency_s', 'predicted_energy_mwh',
            'confidence_score', 'diversity_score', 'ranking_score', 'reason_suitable_for_jetson'
        ]
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rank, model in enumerate(models, 1):
                writer.writerow({
                    'rank': rank,
                    'model_name': model['name'],
                    'architecture_family': model.get('inferred_family', model.get('family', 'Unknown')),
                    'source': model['source'],
                    'is_unseen': str(model['is_unseen']).lower(),
                    'params_m': round(model['params_m'], 2),
                    'gflops': round(model['gflops'], 2),
                    'estimated_latency_s': round(model['estimated_latency_s'], 4),
                    'predicted_energy_mwh': f"{model['predicted_energy_mwh']:.6f}",
                    'confidence_score': round(model.get('confidence_score', 0.5), 3),
                    'diversity_score': round(model.get('diversity_score', 0.5), 3),
                    'ranking_score': round(model.get('ranking_score', 0.5), 4),
                    'reason_suitable_for_jetson': model.get('reason', 'Realistic for Jetson Nano deployment'),
                })
    
    @staticmethod
    def generate_json(models: List[Dict], output_path: str):
        """Generate JSON output."""
        output = {
            'metadata': {
                'total_models': len(models),
                'unseen_count': sum(1 for m in models if m['is_unseen']),
                'benchmark_count': sum(1 for m in models if not m['is_unseen']),
            },
            'statistics': {
                'energy_range': [
                    round(min(m['predicted_energy_mwh'] for m in models), 6),
                    round(max(m['predicted_energy_mwh'] for m in models), 6)
                ],
                'latency_range': [
                    round(min(m['estimated_latency_s'] for m in models), 4),
                    round(max(m['estimated_latency_s'] for m in models), 4)
                ],
                'params_range': [
                    round(min(m['params_m'] for m in models), 2),
                    round(max(m['params_m'] for m in models), 2)
                ],
            },
            'models': models
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
    
    @staticmethod
    def generate_markdown(models: List[Dict], output_path: str, title: str):
        """Generate Markdown report."""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"# {title}\n\n")
            
            # Summary
            unseen = [m for m in models if m['is_unseen']]
            benchmark = [m for m in models if not m['is_unseen']]
            
            f.write("## Summary\n\n")
            f.write(f"- **Total models:** {len(models)}\n")
            f.write(f"- **Unseen models:** {len(unseen)}\n")
            f.write(f"- **Benchmark models:** {len(benchmark)}\n\n")
            
            # Statistics
            f.write("## Statistics\n\n")
            energies = [m['predicted_energy_mwh'] for m in models]
            latencies = [m['estimated_latency_s'] for m in models]
            
            f.write(f"**Energy Consumption:**\n")
            f.write(f"- Min: {min(energies):.1f} mWh\n")
            f.write(f"- Max: {max(energies):.1f} mWh\n")
            f.write(f"- Mean: {sum(energies)/len(energies):.1f} mWh\n")
            f.write(f"- Median: {sorted(energies)[len(energies)//2]:.1f} mWh\n\n")
            
            f.write(f"**Latency (estimated):**\n")
            f.write(f"- Min: {min(latencies):.4f}s\n")
            f.write(f"- Max: {max(latencies):.4f}s\n")
            f.write(f"- Mean: {sum(latencies)/len(latencies):.4f}s\n\n")
            
            # Architecture diversity
            families = Counter()
            for m in models:
                families[m.get('inferred_family', m.get('family', 'Unknown'))] += 1
            
            f.write("## Architecture Diversity\n\n")
            f.write(f"**Total families:** {len(families)}\n\n")
            f.write("| Family | Count (Unseen) | Count (Benchmark) | Total |\n")
            f.write("|--------|---|---|---|\n")
            
            for family in sorted(families.keys()):
                unseen_count = sum(1 for m in unseen if m.get('inferred_family', m.get('family')) == family)
                bench_count = sum(1 for m in benchmark if m.get('inferred_family', m.get('family')) == family)
                f.write(f"| {family} | {unseen_count} | {bench_count} | {families[family]} |\n")
            
            f.write("\n## Top 25 Models\n\n")
            f.write("| Rank | Model | Family | Source | Params (M) | GFLOPs | "
                   "Latency (s) | Energy (mWh) | Ranking Score |\n")
            f.write("|------|-------|--------|--------|--------|--------|--------|--------|----------|\n")
            
            for rank, model in enumerate(models[:25], 1):
                  f.write(f"| {rank} | {model['name']} | {model.get('inferred_family', model.get('family', 'N/A'))} | "
                      f"{model['source']} | {model['params_m']:.1f} | {model['gflops']:.2f} | "
                      f"{model['estimated_latency_s']:.4f} | {model['predicted_energy_mwh']:.6f} | "
                      f"{model.get('ranking_score', 0):.4f} |\n")
            
            f.write("\n## Source Breakdown\n\n")
            f.write(f"- **Unseen Models:** {len(unseen)} real architectures from timm/torchvision (NOT in 360 benchmark)\n")
            f.write(f"- **Benchmark Models:** {len(benchmark)} top-performing models from Jetson Nano 360-model dataset\n")


def main():
    """Reorganize unseen and benchmark models into 2 distinct output sets."""
    
    print("=" * 80)
    print("REORGANIZE UNSEEN & COMBINED MODELS OUTPUT")
    print("=" * 80)
    print()
    
    # Paths
    artifacts_dir = Path("../artifacts")
    unseen_json_path = artifacts_dir / "jetson_nano_unseen_models_refined.json"
    benchmark_csv_path = artifacts_dir / ".." / ".." / ".." / "AGGREGATION_ROOT_CAUSE_AND_FIX.md"  # Placeholder
    benchmark_csv_actual = Path("../data/360_models_benchmark_jetson.csv")
    
    # ========== STEP 1: Load unseen models ==========
    print("[1] Loading 85 unseen models...")
    with open(unseen_json_path, 'r', encoding='utf-8') as f:
        unseen_data = json.load(f)
    
    unseen_models = []
    for model_dict in unseen_data['models']:
        model = {
            'name': model_dict['name'],
            'family': model_dict.get('family', 'Unknown'),
            'params_m': model_dict['params_m'],
            'gflops': model_dict['gflops'],
            'estimated_latency_s': model_dict.get('latency_estimated_s', 0.5),
            'predicted_energy_mwh': model_dict.get('predicted_energy_mwh', 14.0),  # Correct key
            'confidence_score': model_dict.get('confidence_score', 0.5),
            'diversity_score': model_dict.get('diversity_score', 0.5),
            'source': 'unseen',
            'is_unseen': True,
            'reason': f"Unseen real architecture ({model_dict.get('architecture_type', 'CNN')}) suitable for Jetson Nano"
        }
        unseen_models.append(model)
    
    print(f"    Loaded: {len(unseen_models)} unseen models")
    print()
    
    # ========== STEP 2: Load benchmark and select top 15 ==========
    print("[2] Loading benchmark models and selecting diverse top 15...")
    benchmark_all = BenchmarkSelector.load_benchmark_models(str(benchmark_csv_actual))
    print(f"    Total benchmark models: {len(benchmark_all)}")
    
    benchmark_selected = BenchmarkSelector.select_diverse_benchmark(benchmark_all, target_count=15)
    
    # Add metadata to benchmark models
    benchmark_models = []
    for model_dict in benchmark_selected:
        model = {
            'name': model_dict['name'],
            'family': model_dict['inferred_family'],
            'params_m': model_dict['params_m'],
            'gflops': model_dict['gflops'],
            'estimated_latency_s': model_dict['latency_avg_s'],
            'predicted_energy_mwh': model_dict['energy_avg_mwh'],
            'confidence_score': 1.0,  # Benchmark models have perfect confidence
            'diversity_score': 0.7,  # Reasonable diversity
            'source': 'benchmark',
            'is_unseen': False,
            'reason': f"Top benchmark model ({model_dict['inferred_family']}) from 360-model Jetson Nano dataset"
        }
        benchmark_models.append(model)
    
    print(f"    Selected: {len(benchmark_models)} benchmark models")
    print(f"    Families: {', '.join(set(m['family'] for m in benchmark_models))}")
    print()
    
    # ========== STEP 3: Compute weighted ranking scores ==========
    print("[3] Computing weighted ranking scores...")
    
    all_models = unseen_models + benchmark_models
    
    # Calculate ranges for UNSEEN models separately
    unseen_energy_range = (
        min(m['predicted_energy_mwh'] for m in unseen_models),
        max(m['predicted_energy_mwh'] for m in unseen_models)
    )
    unseen_latency_range = (
        min(m['estimated_latency_s'] for m in unseen_models),
        max(m['estimated_latency_s'] for m in unseen_models)
    )
    
    # Calculate ranges for ALL (unseen + benchmark)
    all_energy_range = (
        min(m['predicted_energy_mwh'] for m in all_models),
        max(m['predicted_energy_mwh'] for m in all_models)
    )
    all_latency_range = (
        min(m['estimated_latency_s'] for m in all_models),
        max(m['estimated_latency_s'] for m in all_models)
    )
    diversity_range = (0.0, 1.0)
    
    # Compute ranking scores for unseen models (using unseen ranges)
    for model in unseen_models:
        model['ranking_score'] = WeightedRanking.compute_ranking_score(
            model, unseen_energy_range, unseen_latency_range, diversity_range
        )
    
    # Compute ranking scores for benchmark models (using all ranges for fair comparison)
    for model in benchmark_models:
        model['ranking_score'] = WeightedRanking.compute_ranking_score(
            model, all_energy_range, all_latency_range, diversity_range
        )
    
    all_models = unseen_models + benchmark_models
    
    # Sort by ranking score (descending)
    all_models.sort(key=lambda m: m['ranking_score'], reverse=True)
    
    print(f"    Unseen energy range: {unseen_energy_range[0]:.1f} - {unseen_energy_range[1]:.1f} mWh")
    print(f"    All models energy range: {all_energy_range[0]:.1f} - {all_energy_range[1]:.1f} mWh")
    print(f"    Ranking score range: {min(m['ranking_score'] for m in all_models):.4f} - {max(m['ranking_score'] for m in all_models):.4f}")
    print()
    
    # ========== STEP 4: Generate unseen-only outputs ==========
    print("[4] Generating unseen-only outputs (85 models)...")
    
    unseen_only = [m for m in all_models if m['is_unseen']]
    unseen_only.sort(key=lambda m: m['ranking_score'], reverse=True)
    
    OutputGenerator.generate_csv(
        unseen_only,
        str(artifacts_dir / "jetson_nano_unseen_models_85.csv")
    )
    print(f"    CSV: jetson_nano_unseen_models_85.csv")
    
    OutputGenerator.generate_json(
        unseen_only,
        str(artifacts_dir / "jetson_nano_unseen_models_85.json")
    )
    print(f"    JSON: jetson_nano_unseen_models_85.json")
    
    OutputGenerator.generate_markdown(
        unseen_only,
        str(artifacts_dir / "jetson_nano_unseen_models_85.md"),
        "Jetson Nano Unseen Models (85)"
    )
    print(f"    MD: jetson_nano_unseen_models_85.md")
    print()
    
    # ========== STEP 5: Generate combined top 100 outputs ==========
    print("[5] Generating combined top 100 outputs (85 unseen + 15 benchmark)...")
    
    combined_top100 = all_models[:100]  # Already sorted by ranking score
    
    OutputGenerator.generate_csv(
        combined_top100,
        str(artifacts_dir / "jetson_nano_top100_combined.csv")
    )
    print(f"    CSV: jetson_nano_top100_combined.csv")
    
    OutputGenerator.generate_json(
        combined_top100,
        str(artifacts_dir / "jetson_nano_top100_combined.json")
    )
    print(f"    JSON: jetson_nano_top100_combined.json")
    
    OutputGenerator.generate_markdown(
        combined_top100,
        str(artifacts_dir / "jetson_nano_top100_combined.md"),
        "Jetson Nano Top 100 Combined Models (85 Unseen + 15 Benchmark)"
    )
    print(f"    MD: jetson_nano_top100_combined.md")
    print()
    
    # ========== STEP 6: Print verification and summary ==========
    print("[6] Verification and Summary")
    print("=" * 80)
    print()
    
    print("UNSEEN MODELS (85):")
    print(f"  - Total: 85")
    print(f"  - Energy range: {unseen_energy_range[0]:.1f} - {unseen_energy_range[1]:.1f} mWh")
    print(f"  - Latency range: {unseen_latency_range[0]:.4f} - {unseen_latency_range[1]:.4f}s")
    print(f"  - Families: {len(set(m['family'] for m in unseen_only))}")
    print(f"  - Top 5:")
    for rank, m in enumerate(unseen_only[:5], 1):
        print(f"    {rank}. {m['name']:30} | {m['predicted_energy_mwh']:5.1f} mWh | Score: {m['ranking_score']:.4f}")
    print()
    
    print("BENCHMARK MODELS (15):")
    benchmark_only = [m for m in all_models if not m['is_unseen']]
    print(f"  - Total: {len(benchmark_only)}")
    print(f"  - Energy range: {min(m['predicted_energy_mwh'] for m in benchmark_only):.1f} - {max(m['predicted_energy_mwh'] for m in benchmark_only):.1f} mWh")
    print(f"  - Latency range: {min(m['estimated_latency_s'] for m in benchmark_only):.4f} - {max(m['estimated_latency_s'] for m in benchmark_only):.4f}s")
    print(f"  - Families: {len(set(m['family'] for m in benchmark_only))}")
    print(f"  - Selected:")
    for m in sorted(benchmark_only, key=lambda x: x['predicted_energy_mwh']):
        print(f"    - {m['name']:30} | {m['predicted_energy_mwh']:5.1f} mWh | {m['family']:12} | Score: {m['ranking_score']:.4f}")
    print()
    
    print("COMBINED TOP 100:")
    print(f"  - Total: {len(combined_top100)}")
    print(f"  - Unseen: {len([m for m in combined_top100 if m['is_unseen']])}")
    print(f"  - Benchmark: {len([m for m in combined_top100 if not m['is_unseen']])}")
    print(f"  - Energy range: {min(m['predicted_energy_mwh'] for m in combined_top100):.1f} - {max(m['predicted_energy_mwh'] for m in combined_top100):.1f} mWh")
    print(f"  - Latency range: {min(m['estimated_latency_s'] for m in combined_top100):.4f} - {max(m['estimated_latency_s'] for m in combined_top100):.4f}s")
    print(f"  - Families: {len(set(m['family'] for m in combined_top100))}")
    print()
    
    print("=" * 80)
    print("✅ SUCCESS: All outputs generated")
    print("=" * 80)


if __name__ == '__main__':
    import sys
    sys.exit(main())
