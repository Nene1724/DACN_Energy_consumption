# AGGREGATION-LEVEL SEMANTIC MISMATCH: ROOT CAUSE ANALYSIS

## Issue Summary
Benchmark energy (23.7796 mWh for 30 runs) is being compared against per-inference prediction (9.46-10.29 mWh), causing false budget exceeded events.

## Critical Paths to Trace

### Path 1: Benchmark Energy Accumulation
1. **Benchmark Start** → `/benchmark` endpoint called
2. **Meter Baseline Setup** → baseline_energy_wh set when FNB58 first connects
3. **30 Inferences Run** → Each inference runs, FNB58 accumulates power consumption
4. **Session Energy Computed** → `session_energy_mwh = (total_energy_wh - baseline_energy_wh) * 1000`
5. **Stored in** → `meter["measured_energy_mwh"]` and `meter["session_energy_mwh"]`

**Result:** measured_energy_mwh = 23.7796 mWh (TOTAL for all 30 runs)

### Path 2: Per-Inference Prediction
1. **Model Deploy** → prediction_energy_mwh = 9.46-10.29 mWh
2. **Budget Set** → energy_budget_mwh = per-inference budget
3. **Telemetry Arrives** → Delta or cumulative readings from FNB58

**Expected:** Compare 9.46-10.29 mWh against budget
**Actual:** ???

### Path 3: Budget Comparison Logic
Location: `record_energy_sample()` in server.py line 679
```python
over_budget = budget is not None and sample["energy_mwh"] > budget
```

**Question:** What is passed as `energy_mwh` when?
- During `/predict` → per-inference delta (correct)
- During `/benchmark` → ???
- From `/telemetry` delta payload → depends on payload source

### Path 4: measured_energy_mwh Usage
Stored at: server.py line 943
```python
meter["measured_energy_mwh"] = round(session_energy_mwh, 4)
```

**Semantic:** This is SESSION cumulative, NOT per-inference

**Problem:** Is this value ever passed to budget checks?

## REQUIRED DEBUG OUTPUTS

### Query 1: Where is latest_inference_delta_mwh set?
- Answer: Line 1098 in server.py during `end_inference_measurement()`
- Source: Computed from FNB58 meter deltas (should be per-inference)

### Query 2: When does measured_energy_mwh get populated?
- Answer: Line 943 in server.py during meter snapshot updates
- Source: `(total_energy_wh - baseline_energy_wh) * 1000` (SESSION cumulative)

### Query 3: Where is measured_energy_mwh used in budget checks?
- Search needed: Look for direct usage of measured_energy_mwh in budget logic

### Query 4: What telemetry payload arrives during benchmark?
- Does FNB58 send ONE big delta for all 30 runs?
- Does it send 30 small deltas?
- Is there a telemetry window mismatch?

### Query 5: What gets stored in energy_metrics.history?
- Is per-inference delta recorded?
- Or is session total recorded?
- When is record_energy_sample() called during benchmark?

## HYPOTHESIS

The benchmark path likely does NOT call `start_inference_measurement()` / `end_inference_measurement()` for each run.

If benchmark just runs 30 inferences without per-inference measurement wrapping:
1. No `latest_inference_delta_mwh` is set
2. Only `measured_energy_mwh` (session cumulative) gets updated
3. If telemetry arrives AFTER benchmark completes, a single "delta" payload might contain the TOTAL benchmark energy
4. This total is compared against per-inference budget → FALSE "over_budget"

## REQUIRED INVESTIGATION STEPS

1. Check if benchmark_loaded_model calls measurement functions
2. Check if FNB58 telemetry is being batched into single large payloads
3. Check if record_energy_sample is being called with aggregated energy values
4. Verify budget comparison uses LATEST INFERENCE DELTA, not measured_energy_mwh
5. Check UI rendering logic - is it comparing correct semantics?

## EXPECTED FIX FLOW

1. **Separate semantics explicitly:**
   - `benchmark_total_energy_mwh` = sum of all runs
   - `benchmark_avg_energy_mwh` = total / runs
   - `latest_inference_delta_mwh` = last single inference

2. **Budget comparison uses:**
   - benchmark_avg_energy_mwh (NOT benchmark_total_energy_mwh)

3. **UI displays separately:**
   - "Benchmark Total Energy: XX mWh"
   - "Average Per-Inference: XX mWh"
   - "Latest Inference: XX mWh"
   - "Session Cumulative: XX mWh"

