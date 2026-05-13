# Energy Measurement Isolation Architecture

## Problem Statement

**Symptom:** Measured inference energy exceeded the model's per-inference power budget even during single prediction requests.

**Root Cause:** The inference energy measurement window was too broad. A single inference request would open a measurement window that remained active for:
- The model inference execution itself
- Concurrent HTTP requests (`GET /status`, `GET /metrics`, `GET /camera/snapshot`, etc.)
- Background polling activity from the controller
- Frame encoding and streaming operations

This meant that the measured energy delta included not just the model computation, but also the overhead of controller polling, device health checks, and frame capture operations. The longer the window, the more unrelated activity could be captured.

**Example Contamination Sequence:**
```
1. POST /predict starts → start_inference_measurement(inference_id=123)
2. Model invokes (20ms, ~5mJ)
3. Still holding window → GET /status arrives (10ms, ~2mJ for telemetry sync)
4. GET /metrics arrives (5ms, ~1mJ)
5. POST /benchmark arrives (100ms, ~25mJ for dummy load)
6. end_inference_measurement() → delta = 33mJ (should be ~5mJ)
```

---

## Solution Architecture

### Design Principles

1. **Window Ownership:** Each inference execution has exclusive ownership of a measurement window. No other inference request can start until the current window is finalized.

2. **Inference-Scoped Boundaries:** The window opens immediately before `model.invoke()` and closes immediately after, not around HTTP request handling or frame capture.

3. **Request Contamination Tracking:** All HTTP requests that arrive while a window is active are logged as "non-inference activity" with their timing and endpoint.

4. **Serialization Behind Lock:** Multiple inference paths (`/predict`, `/benchmark`, `/camera/fall-detect` frames) share a single execution lock to prevent overlapping windows.

---

## Key Components

### 1. Measurement Window State (`ACTIVE_MEASUREMENT_WINDOW`)

Tracks the currently-active inference measurement with metadata:

```python
ACTIVE_MEASUREMENT_WINDOW = {
    'measurement_id': str,        # Unique ID (UUID)
    'inference_id': str,          # Link to the inference request
    'trigger_source': str,        # 'predict' | 'benchmark' | 'camera_frame'
    'window_start_ts': float,     # Unix timestamp (seconds)
    'window_end_ts': float,       # Unix timestamp (seconds, None if open)
    'before_total_wh': float,     # Device total energy at start
    'after_total_wh': float,      # Device total energy at end (None if open)
    'requests_during_window': [], # List of {'endpoint': str, 'method': str, 'ts': float}
    'inference_count': int,       # Number of model.invoke() calls in window
    'non_inference_activity_detected': bool
}
```

### 2. Exclusive Execution Lock (`INFERENCE_EXECUTION_LOCK`)

A threading.RLock that serializes all model inference paths:
- `/predict` requests
- `/benchmark` requests
- `/camera/fall-detect` frame inferences

Only one thread can hold the lock and have an active measurement window at a time.

### 3. Measurement Lifecycle Functions

#### `start_inference_measurement(inference_id, trigger_source)`
- Acquires `INFERENCE_EXECUTION_LOCK`
- Records `window_start_ts` and `before_total_wh` from FNB58
- Initializes `ACTIVE_MEASUREMENT_WINDOW`
- Returns `measurement_id` for logging and tracing

#### `end_inference_measurement(inference_id, delta_mwh=None, error=None)`
- Records `window_end_ts` and `after_total_wh`
- Computes delta as `after_total_wh - before_total_wh` (if not provided)
- Logs structured JSON with:
  - measurement ID and timing
  - energy delta (mWh)
  - window duration (ms)
  - requests that arrived during window
  - contamination flag
- Clears `ACTIVE_MEASUREMENT_WINDOW`
- Releases `INFERENCE_EXECUTION_LOCK`

#### `_begin_measurement_window(inference_id, trigger_source)` (internal)
- Called inside lock before model invocation
- Records: measurement_id, trigger_source, before_total_wh, window_start_ts

#### `_end_measurement_window(delta_mwh, inference_count)` (internal)
- Called inside lock after model invocation
- Records: after_total_wh, window_end_ts, inference_count
- Emits structured telemetry log

### 4. Request Contamination Tracing (`_trace_active_measurement_request()`)

Registered as a Flask `before_request` hook that executes on every HTTP request:
- Checks if `ACTIVE_MEASUREMENT_WINDOW` is open
- If open: logs the request endpoint, method, and timestamp to `requests_during_window`
- Sets `non_inference_activity_detected = True` if any non-trivial request arrives

This function is **non-blocking**—it does not delay or interfere with the request.

---

## Strict Telemetry Contract

The `/telemetry` endpoint enforces a strict contract for energy payloads:

### Valid Cumulative Payload
```json
{
  "energy_kind": "cumulative",
  "energy_wh": 0.0182,
  "power_w": 2.9,
  "duration_s": 22.6,
  "timestamp": "2026-03-30T15:10:00Z"
}
```
- **Semantics:** Total device energy since boot or last reset.
- **Expected Source:** FNB58 meter cumulative reading.

### Valid Delta Payload
```json
{
  "energy_kind": "delta",
  "delta_mwh": 12.5,
  "power_mw": 2900,
  "timestamp": "2026-03-30T15:10:00Z"
}
```
- **Semantics:** Energy consumed during a specific measurement window.
- **Expected Source:** Inference measurement window delta.

### Rejected (Ambiguous) Payload
```json
{
  "energy_kind": "delta",
  "energy_wh": 0.0182  # ← FORBIDDEN: contradicts energy_kind
}
```
- **Error:** HTTP 400 + structured rejection log
- **Reason:** Cannot determine if `0.0182 Wh` is absolute or relative.

---

## Expected Log Outputs

### Measurement Window JSON (on `end_inference_measurement()`)

```json
{
  "timestamp": "2026-05-11T10:30:45.123Z",
  "event": "ENERGY-WINDOW-END",
  "measurement_id": "meas_550e8400-e29b-41d4-a716-446655440000",
  "inference_id": "req_550e8400-e29b-41d4-a716-446655440001",
  "trigger_source": "predict",
  "window_duration_ms": 45,
  "window_start_ts": 1715421045.000,
  "window_end_ts": 1715421045.045,
  "before_total_wh": 0.1234,
  "after_total_wh": 0.1246,
  "delta_mwh": 12.0,
  "inference_count": 1,
  "requests_during_window": [
    {"endpoint": "/status", "method": "GET", "ts": 1715421045.020},
    {"endpoint": "/metrics", "method": "GET", "ts": 1715421045.030}
  ],
  "non_inference_activity_detected": true,
  "status": "success"
}
```

### Measurement Window Start Log (on `start_inference_measurement()`)

```json
{
  "timestamp": "2026-05-11T10:30:45.000Z",
  "event": "ENERGY-WINDOW-START",
  "measurement_id": "meas_550e8400-e29b-41d4-a716-446655440000",
  "inference_id": "req_550e8400-e29b-41d4-a716-446655440001",
  "trigger_source": "predict",
  "device_state": {
    "total_energy_wh": 0.1234,
    "power_w": 2.9,
    "temperature_c": 58.5
  }
}
```

### Contamination Detection Log (on `_trace_active_measurement_request()`)

```json
{
  "timestamp": "2026-05-11T10:30:45.020Z",
  "event": "MEASUREMENT-CONTAMINATION",
  "measurement_id": "meas_550e8400-e29b-41d4-a716-446655440000",
  "contaminant_endpoint": "/status",
  "contaminant_method": "GET",
  "contaminant_latency_ms": 15,
  "inference_activity_ongoing": true
}
```

---

## Measurement Window Lifecycle Diagram

```
Time →

Thread A (inference request)      Thread B (polling)
─────────────────────────        ──────────────────
POST /predict
  │
  └─ Acquire INFERENCE_EXECUTION_LOCK
       │
       └─ start_inference_measurement()
            ├─ ENERGY-WINDOW-START log
            │
            └─ _begin_measurement_window()
                 ├─ Record before_total_wh
                 │
                 └─ model.invoke()
                    (45 ms)
                      │
                      ├─ GET /status arrives ◄─ _trace_active_measurement_request()
                      │  (logs contamination)
                      │
                      └─ model returns
                 
                 └─ _end_measurement_window()
                    ├─ Record after_total_wh
                    └─ Compute delta
            │
            └─ end_inference_measurement()
                 ├─ ENERGY-WINDOW-END log
                 │  (includes requests_during_window)
                 │
                 └─ Release INFERENCE_EXECUTION_LOCK
  │
  └─ Return response
     (includes latest_inference_delta_mwh)
```

---

## Implementation Details

### Per-Frame Fall Detection Isolation

The `/camera/fall-detect` endpoint processes up to 16 frames in a loop. **Each frame gets its own measurement window:**

```python
for frame in frames:
    measurement_id = start_inference_measurement(
        inference_id=f"{request_id}:frame_{frame_num}",
        trigger_source="camera_frame"
    )
    try:
        pose = _run_movenet_on_frame(frame)
        # Isolation: only pose inference is measured
    finally:
        end_inference_measurement(measurement_id, ...)
```

This means:
- Frame 1: window_duration ≈ 20–30 ms
- Frame 2: window_duration ≈ 20–30 ms
- ... (16 windows total)
- **Not:** one 500 ms window around all 16 frames

### Single-Inference Powercap Path

The `/predict` endpoint without camera mode measures a single model inference:

```python
measurement_id = start_inference_measurement(
    inference_id=request_id,
    trigger_source="predict"
)
try:
    results = model.invoke(dummy_input)
finally:
    end_inference_measurement(measurement_id, ...)
```

Window duration: typically 30–60 ms on Jetson Nano.

### Benchmark Path Serialization

The `/benchmark` endpoint runs many inferences, but each is isolated:

```python
for i in range(num_runs):
    measurement_id = start_inference_measurement(
        inference_id=f"{benchmark_id}:run_{i}",
        trigger_source="benchmark"
    )
    try:
        model.invoke(dummy_input)
    finally:
        end_inference_measurement(measurement_id, ...)
    
    # Between runs, the lock is released, allowing
    # other requests to proceed
```

---

## Verification Checklist

### 1. Measurement Window Duration
- [ ] `/predict` on dummy input: 30–80 ms window
- [ ] `/predict` with camera input: 20–50 ms per frame
- [ ] `/benchmark` single run: 30–80 ms per iteration
- [ ] No window exceeds 200 ms

### 2. Contamination Detection
- [ ] Send `GET /status` while inference is running
- [ ] Check logs for `MEASUREMENT-CONTAMINATION` entry
- [ ] Verify `requests_during_window` array is populated in `ENERGY-WINDOW-END`
- [ ] Confirm `non_inference_activity_detected: true` flag is set

### 3. Energy Delta Accuracy
- [ ] Measure 10 inferences with polling traffic in background
- [ ] Verify mean delta ≈ single-inference baseline (no accumulated drift)
- [ ] Verify delta variance is low (±10% of mean)

### 4. Lock Serialization
- [ ] Send concurrent `/predict` and `/benchmark` requests
- [ ] Verify second request waits for first to complete
- [ ] Check logs show no overlapping measurement windows

### 5. Telemetry Strict Contract
- [ ] Submit valid cumulative payload → accepted (HTTP 200)
- [ ] Submit valid delta payload → accepted (HTTP 200)
- [ ] Submit ambiguous payload (delta with `energy_wh` field) → rejected (HTTP 400)
- [ ] Check rejection logs include structured error details

---

## Operational Best Practices

### 1. Minimize Background Polling During Measurement
- Controller should use longer polling intervals when inference is active
- Or implement a `GET /inference-status` endpoint that does not trigger contamination logs

### 2. Decode Contamination Logs in Controller
- Parse `requests_during_window` array to assess measurement quality
- Flag results where `non_inference_activity_detected: true` for auditing
- Consider excluding highly-contaminated measurements from energy budget averages

### 3. Per-Frame Energy Reporting for Camera Fall Detection
- Aggregate per-frame deltas to compute session energy
- Report both "inference energy" (sum of window deltas) and "total session energy" (meter snapshot)
- Difference highlights overhead from frame capture, pose analysis, fall scoring

### 4. Benchmark Energy Normalization
- Divide total benchmark energy by number of runs to get per-run estimate
- Compare with `/predict` single-inference energy to detect measurement variance
- Use variance to set energy budget tolerance (±15–20%)

---

## Troubleshooting

### Symptom: Window Duration > 100 ms
**Possible Cause:** Lock held longer than expected  
**Investigation:**
- Check logs for nested lock acquisitions
- Verify no blocking I/O inside the lock
- Inspect camera frame capture latency

### Symptom: `non_inference_activity_detected: false` despite visible polling requests
**Possible Cause:** Polling requests are finishing before window starts or after it ends  
**Investigation:**
- Check polling interval vs window duration
- Increase polling rate to ensure overlap
- Verify request timing in logs

### Symptom: Energy delta > model's TDP budget
**Possible Cause:** Measurement window still too broad (e.g., includes multiple frames)  
**Investigation:**
- Confirm `inference_count: 1` in log
- Check frame count for camera fall-detect
- Verify lock is released between calls

---

## References

- [jetson-ml-agent/app/server.py](../app/server.py) - Measurement functions and isolation code
- [jetson-ml-agent/app/fnb58_telemetry_collector.py](../app/fnb58_telemetry_collector.py) - Telemetry payload normalization
- [jetson-ml-agent/tests/test_energy_pipeline.py](../tests/test_energy_pipeline.py) - Isolation regression tests
