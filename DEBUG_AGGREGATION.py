#!/usr/bin/env python3
"""
Debug script to trace aggregation-level semantic mismatch.
This identifies exact paths where TOTAL vs AVG energy semantics diverge.
"""

import json
import sys

# Simulate the exact scenario the user reported

print("\n" + "="*80)
print("AGGREGATION-LEVEL SEMANTIC MISMATCH - DEBUG TRACE")
print("="*80 + "\n")

print("SCENARIO:")
print("- Benchmark runs 30 inferences on Jetson")
print("- FNB58 meter accumulates energy over entire benchmark window")
print("- After benchmark completes, FNB58 sends one telemetry payload")
print()

# Stage 1: Benchmark setup
print("STAGE 1: BENCHMARK START")
print("-" * 80)
benchmark_runs = 30
predicted_energy_per_inference_mwh = 9.46  # Typical prediction
energy_budget_per_inference_mwh = 10.0  # Per-inference budget

print(f"Configuration:")
print(f"  benchmark_runs: {benchmark_runs}")
print(f"  predicted_energy_per_inference: {predicted_energy_per_inference_mwh} mWh")
print(f"  energy_budget_per_inference: {energy_budget_per_inference_mwh} mWh")
print()

# Stage 2: Meter baseline setup
print("STAGE 2: METER BASELINE")
print("-" * 80)
baseline_energy_wh = 0.1234  # FNB58 starts at this value
print(f"  baseline_energy_wh: {baseline_energy_wh} Wh")
print()

# Stage 3: Benchmark execution (NO per-inference measurement wrapping)
print("STAGE 3: BENCHMARK EXECUTION")
print("-" * 80)
print("Note: benchmark_loaded_model() holds INFERENCE_EXECUTION_LOCK for ALL 30 runs")
print("      but does NOT call start_inference_measurement()/end_inference_measurement()")
print("      Result: NO per-inference energy records created")
print()

# Stage 4: Energy accumulation
print("STAGE 4: ENERGY ACCUMULATION (during benchmark)")
print("-" * 80)
actual_energy_per_inference = 0.79  # Actual measured
total_energy_consumed_mwh = actual_energy_per_inference * benchmark_runs
print(f"  Actual energy per inference: {actual_energy_per_inference} mWh")
print(f"  Total energy for {benchmark_runs} runs: {total_energy_consumed_mwh:.4f} mWh")
print()

# Stage 5: FNB58 telemetry arrival
print("STAGE 5: FNB58 TELEMETRY ARRIVAL (AFTER BENCHMARK)")
print("-" * 80)
after_benchmark_energy_wh = baseline_energy_wh + (total_energy_consumed_mwh / 1000.0)
print(f"  FNB58 meter before benchmark: {baseline_energy_wh:.4f} Wh")
print(f"  FNB58 meter after benchmark: {after_benchmark_energy_wh:.6f} Wh")
print()

# Stage 6: Session energy computation
print("STAGE 6: SESSION ENERGY COMPUTATION")
print("-" * 80)
session_energy_delta_wh = after_benchmark_energy_wh - baseline_energy_wh
session_energy_delta_mwh = session_energy_delta_wh * 1000.0
print(f"  session_energy_delta = ({after_benchmark_energy_wh:.6f} - {baseline_energy_wh:.4f}) * 1000")
print(f"  session_energy_delta = {session_energy_delta_mwh:.4f} mWh")
print(f"  This is STORED in: meter['measured_energy_mwh']")
print(f"  This is STORED in: meter['session_energy_mwh']")
print(f"  SEMANTIC: SESSION CUMULATIVE (sum of all 30 inferences)")
print()

# Stage 7: Telemetry processing
print("STAGE 7: TELEMETRY INGESTION")
print("-" * 80)
print("Question: What does FNB58 exporter send as telemetry payload?")
print()
print("Option A: Cumulative energy_wh payload")
telemetry_cumulative_wh = after_benchmark_energy_wh
payload_cumulative = {
    "energy_kind": "cumulative",
    "energy_wh": telemetry_cumulative_wh,
    "power_w": 0.79  # Typical
}
print(f"  Payload: {json.dumps(payload_cumulative, indent=2)}")
print(f"  Processing: Energy is treated as CUMULATIVE (meter state only)")
print(f"  NO record_energy_sample() called for cumulative")
print(f"  Result: budget_check NOT TRIGGERED")
print()

print("Option B: Delta energy_mwh payload (WRONG SEMANTICS)")
telemetry_delta_mwh = total_energy_consumed_mwh
payload_delta_wrong = {
    "energy_kind": "delta",
    "delta_mwh": telemetry_delta_mwh,
}
print(f"  Payload: {json.dumps(payload_delta_wrong, indent=2)}")
print(f"  Processing: Energy is treated as SINGLE INFERENCE DELTA")
print(f"  record_energy_sample(energy_mwh={telemetry_delta_mwh:.4f}) CALLED")
print(f"  Budget check: {telemetry_delta_mwh:.4f} > {energy_budget_per_inference_mwh}?")
print()

# Stage 8: Budget comparison
print("STAGE 8: BUDGET COMPARISON")
print("-" * 80)
is_over_budget = telemetry_delta_mwh > energy_budget_per_inference_mwh
print(f"  if ({telemetry_delta_mwh:.4f} > {energy_budget_per_inference_mwh}): over_budget=True")
print(f"  Result: {is_over_budget}")
print()

if is_over_budget:
    print(f"  🔴 FALSE POSITIVE: Budget exceeded")
    print(f"     Reason: Compared TOTAL benchmark energy ({telemetry_delta_mwh:.4f} mWh)")
    print(f"             against PER-INFERENCE budget ({energy_budget_per_inference_mwh} mWh)")
    print()

# Stage 9: Correct semantics
print("STAGE 9: CORRECT SEMANTICS (WHAT SHOULD HAPPEN)")
print("-" * 80)
benchmark_avg_energy_mwh = total_energy_consumed_mwh / benchmark_runs
print(f"  Benchmark total energy: {total_energy_consumed_mwh:.4f} mWh")
print(f"  Benchmark average per inference: {benchmark_avg_energy_mwh:.4f} mWh")
print(f"  Per-inference budget: {energy_budget_per_inference_mwh} mWh")
print(f"  Comparison: {benchmark_avg_energy_mwh:.4f} > {energy_budget_per_inference_mwh}?")
print(f"  Result: {benchmark_avg_energy_mwh > energy_budget_per_inference_mwh}")
print()

# Stage 10: Root cause identification
print("STAGE 10: ROOT CAUSE IDENTIFICATION")
print("-" * 80)
print("Problem 1: Benchmark has NO per-inference measurement wrapping")
print("  - benchmark_loaded_model() does not call start_inference_measurement()")
print("  - benchmark_loaded_model() does not call end_inference_measurement()")
print("  - Result: latest_inference_delta_mwh is NOT updated")
print()
print("Problem 2: Session energy is confused with average energy")
print("  - meter['measured_energy_mwh'] = session cumulative (TOTAL)")
print("  - But UI/telemetry sometimes treats it as per-inference")
print("  - Result: TOTAL is compared against PER-INFERENCE budget")
print()
print("Problem 3: FNB58 telemetry window spans entire benchmark")
print("  - If only ONE telemetry arrives during 30-inference window")
print("  - And it's sent as 'delta', it will be TOTAL, not per-inference")
print("  - Result: record_energy_sample() records TOTAL instead of AVG")
print()

# Recommendations
print("REQUIRED FIXES")
print("-" * 80)
print("1. Wrap EACH benchmark iteration in measurement:")
print("     for i in range(benchmark_runs):")
print("         measurement_id = start_inference_measurement(trigger='benchmark')")
print("         _run_loaded_model_once_locked(...)")
print("         end_inference_measurement(measurement_id)")
print()
print("2. Compute and store benchmark metrics explicitly:")
print("     benchmark_total_energy_mwh = sum of deltas")
print("     benchmark_avg_energy_mwh = total / runs")
print()
print("3. Budget checks MUST use AVERAGE, not TOTAL:")
print("     over_budget = benchmark_avg_energy_mwh > budget")
print()
print("4. UI displays MUST separate semantics:")
print("     - Benchmark Total Energy: XX mWh")
print("     - Average Per-Inference: XX mWh")
print("     - Latest Inference Delta: XX mWh")
print("     - Session Cumulative: XX mWh")
print()

print("="*80)
print("END TRACE")
print("="*80 + "\n")
