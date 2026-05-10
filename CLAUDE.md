# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

**DACN_Energy_consumption** is a two-part ML deployment and monitoring system for IoT edge devices:

1. **ML Controller** (`ml-controller/`): Flask web UI running on Windows/Linux for:
   - Model analysis and energy consumption prediction
   - Device fleet management (Balena Cloud integration)
   - Model deployment orchestration to Jetson Nano / Raspberry Pi 5
   - Fall detection evaluation using UR Fall dataset
   - Energy benchmarking and prediction

2. **Jetson ML Agent** (`jetson-ml-agent/`): Docker containerized inference runtime for edge devices:
   - TensorFlow Lite and ONNX model inference
   - Fall detection using MoveNet pose estimation
   - Energy telemetry collection (FNB58 power meter integration)
   - Camera streaming and live inference
   - Auto-restart on reboot with Balena Cloud

## Architecture

### Inter-Service Communication

- **Controller → Agent**: REST API calls (model deployment, inference requests)
  - Devices accessed via Balena public URL (if available) or direct LAN IP at `:8000`
  - Model downloads: `http://<device>:8000/deploy` → agent downloads from controller URL
  - Inference: `http://<device>:8000/predict` returns predictions and energy metrics

- **Agent → Controller**: Telemetry push (energy reports, fall events)
  - POST `/api/energy/report` — energy measurement + model performance data
  - POST `/api/medical/fall-events` — fall detection events with confidence scores

### Key Design Patterns

**Energy Prediction Pipeline:**
- `ModelAnalyzer` loads benchmark CSVs (`360_models_benchmark_jetson.csv`, `253_models_benchmark_rpi5.csv`)
- `EnergyPredictorService` uses device-specific models:
  - Jetson Nano: `jetson_energy_model.pkl` + `jetson_scaler.pkl` (22% MAPE)
  - Raspberry Pi 5: `rpi5_energy_model.pkl` + `rpi5_scaler.pkl` (15% MAPE)
  - Fallback: `energy_predictor.pkl` (50% MAPE) for unknown devices
- Features extracted from ONNX/TFLite headers: `params_m`, `gflops`, `size_mb`, `latency_avg_s`, `throughput_iter_per_s`
- Energy thresholds (p25/p50/p75) guide recommendations: "excellent" → "good" → "acceptable" → "high"

**Model Artifact Resolution:**
- Models stored in `ml-controller/model_store/` and `ml-controller/new_models/`
- Resolution by normalized name (alphanumeric-only comparison, case-insensitive)
- Preferred extensions: `.tflite`, `.onnx`, `.pth`, `.pt`, `.bin`
- Device downloads resolved via `CONTROLLER_PUBLIC_URL` env var (for cross-network) or inferred LAN IP

**Fall Detection:**
- MoveNet TFLite model (`movenet_singlepose_lightning_f16.tflite`) detects poses
- Agent `/camera/fall-detect` runs inference on video frames
- Returns keypoints and pose statistics; controller `/api/medical/fall-events` logs detections
- Batch testing via `ur_fall_batch_test.py` uses UR Fall Dataset MP4s

**Energy Telemetry (FNB58 Power Meter):**
- Agent reads via USB/serial/hidraw interfaces
- `FNB58Reader`, `FNB58USBReader`, `FNB58HIDRawReader` abstractions
- `fnb58_telemetry_collector.py` pushes to controller energy reports
- Used to compare predicted energy (`model_info.predicted_mwh`) vs. measured (`measured_energy_mwh`)

### Controller (ml-controller/)

```
ml-controller/
├── python/
│   ├── app.py                           # Main Flask app (40+ endpoints)
│   ├── model_analyzer.py               # Benchmark data loader & model listing
│   ├── energy_predictor_service.py     # Device-specific energy prediction
│   ├── onnx_feature_extractor.py       # Extract features from ONNX/TFLite headers
│   ├── experiment_logger.py             # Experiment tracking & data logging
│   ├── log_manager.py                   # Deployment log CRUD
│   ├── energy_benchmark_workbench.py   # Local benchmarking utilities
│   └── ur_fall_batch_test.py           # Fall detection evaluation script
├── templates/
│   ├── index.html                       # Main dashboard
│   ├── deployment.html                  # Model upload/deploy UI
│   ├── monitoring.html                  # Device metrics view
│   ├── analytics.html                   # Energy analysis
│   └── medical.html                     # Fall detection UI
├── data/
│   ├── 360_models_benchmark_jetson.csv # Jetson benchmark data
│   ├── 253_models_benchmark_rpi5.csv   # RPi5 benchmark data
│   ├── deployment_logs.json             # Deployment history
│   ├── energy_measurements.json         # Telemetry from devices
│   ├── benchmark_reports.json           # Inference benchmarks
│   └── fall_detection_events.json       # Fall events from agents
├── model_store/                         # Pre-loaded model artifacts
├── new_models/                          # User-uploaded model artifacts
└── artifacts/                           # Energy models & metadata
    ├── jetson_energy_model.pkl
    ├── rpi5_energy_model.pkl
    ├── device_specific_features.json
    └── energy_thresholds.json
```

**Key Controller Endpoints (40+):**
- `/api/models/all` — list benchmarked models with energy predictions
- `/api/models/popular` — recommended models for budgets
- `/api/predict-energy` — predict energy for custom model features
- `/api/deploy` — deploy model to Balena device
- `/api/balena/devices` — list connected Jetson/RPi5 devices
- `/api/energy/report` — receive telemetry from agents
- `/api/medical/fall-events` — view fall detection logs
- `/api/device/benchmark` — trigger inference benchmark on remote device

### Agent (jetson-ml-agent/)

```
jetson-ml-agent/
├── app/
│   ├── server.py                        # Main Flask server (30+ endpoints)
│   ├── movenet_fall_detection.py       # Fall detection pose analysis
│   ├── fnb58_reader.py                 # Serial port reader for FNB58
│   ├── fnb58_usb_reader.py             # USB/libusb reader
│   ├── fnb58_hidraw_reader.py          # Linux hidraw interface
│   ├── fnb58_exporter_reader.py        # External command/file reader
│   ├── fnb58_telemetry_collector.py    # Push telemetry to controller
│   └── benchmark_mape.py               # MAPE evaluation against telemetry
├── Dockerfile                           # Balena-based build for Jetson Nano
├── docker-compose.yml                   # Local dev environment
├── nginx.conf                           # Reverse proxy config
└── requirements.txt                     # Python dependencies
```

**Key Agent Endpoints (30+):**
- `/status` — agent readiness
- `/metrics` — device system metrics
- `/deploy` — receive model download URL & load
- `/start`, `/stop` — inference control
- `/predict` — run inference (dummy input or MoveNet camera)
- `/camera/fall-detect` — analyze video/frames for falls
- `/camera/fall-watch/start|stop|status` — continuous monitoring
- `/camera/stream` — MJPEG stream
- `/benchmark` — local inference benchmarking
- `/fnb58/status` — power meter status
- `/measure_energy` — benchmark with FNB58 logging
- `/telemetry` — push energy history to controller

## Development Workflow

### Run Controller Locally

**Windows (PowerShell):**
```powershell
./start_web.ps1
# Opens http://localhost:5000
```

**Manual (any OS):**
```bash
py -3.11 -m venv .venv          # or python3 -m venv .venv
.venv/Scripts/python -m pip install -r ml-controller/requirements.txt
cd ml-controller/python
../../.venv/Scripts/python app.py      # or ../.venv/bin/python on Unix
# Listens on 0.0.0.0:5000
```

**Optional Environment Variables** (set in `ml-controller/.env`):
- `BALENA_API_TOKEN` / `BALENA_API_KEY` — authenticate to Balena Cloud API
- `CONTROLLER_PUBLIC_URL` — URL for devices to download models (e.g., `https://my.server.com:5000`)
- `MODEL_ANALYZE_MAX_MB` — max file size for feature extraction (default: 128)

### Run Agent Locally

**Via Docker Compose:**
```bash
cd jetson-ml-agent
docker-compose up --build
# Listens on 0.0.0.0:8000 and 0.0.0.0:80
```

**Manual (on Jetson or macOS with TensorFlow installed):**
```bash
pip install -r requirements.txt
MODEL_DIR_OVERRIDE=/tmp/models python3 app/server.py
```

**Test Agent API:**
```bash
curl http://127.0.0.1:8000/status
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
  -d '{"duration_s": 2.5}'
```

### Test Fall Detection with UR Dataset

1. Deploy `movenet_singlepose_lightning_f16.tflite` to Jetson via controller UI
2. Run batch evaluation from repo root:
```bash
.venv/Scripts/python ml-controller/python/ur_fall_batch_test.py \
  --agent http://<JETSON_IP>:8000 \
  --kind all --start 1 --end 40
```
- Downloads MP4s to `.cache/ur_fall/`
- Uploads to agent, runs fall detection, writes JSON report with TP/TN/FP/FN
- Report saved to `ml-controller/artifacts/report_figures/ur_fall_eval_<timestamp>.json`

### Energy Prediction Notebooks

Jupyter notebooks in `ml-controller/notebooks/`:
- `energy_prediction_model_multidevice.ipynb` — train Jetson + RPi5 models
- `energy_prediction_model_complete.ipynb` — unified + device-specific comparison

Run cells to export models to `ml-controller/artifacts/`:
- `jetson_energy_model.pkl` + `jetson_scaler.pkl`
- `rpi5_energy_model.pkl` + `rpi5_scaler.pkl`
- `device_specific_metadata.json`

## Common Tasks

### Add a New Model to Benchmarks

1. Place ONNX/TFLite file in `ml-controller/new_models/`
2. Manually benchmark on Jetson:
   - Call agent `/benchmark` endpoint or use `energy_benchmark_workbench.py`
   - Record latency, throughput, energy
3. Update CSV: add row to `ml-controller/data/360_models_benchmark_jetson.csv`
4. Run `generate_popular_models_metadata.py` to refresh popular models list

### Deploy Model to Device

1. Controller `/api/models/upload` → store file in `ml-controller/artifacts/`
2. Controller `/api/deploy` → agent downloads from controller URL, saves to `/data/models/current_model.*`
3. Agent `/start` → loads model into memory (TFLite or ONNX runtime)
4. Call `/predict` for inference

### Integrate FNB58 Power Meter

1. On device SSH: `lsusb | grep -i fnirsi` to verify USB connection
2. In agent container, run:
```bash
python3 app/fnb58_telemetry_collector.py \
  --agent-url http://127.0.0.1:8000 \
  --json-command "python3 /data/fnb58_exporter.py" \
  --interval 3 \
  --source fnb58
```
3. Agent `/telemetry` returns energy history
4. Controller `/api/energy/report` receives and logs measurements

### Debug Controller Issues

- Check logs: `ml-controller/data/deployment_logs.json`
- Enable debug: set `FLASK_DEBUG=1` before running `app.py`
- Model resolution: `resolve_model_artifact()` uses normalized alphanumeric names
- Energy prediction fallback: if model is `None`, uses unified model with 50% MAPE warning

### Debug Agent Issues

- Logs printed to stdout (captured by Docker/Balena logs)
- State saved in `/data/models/agent_state.json`
- Camera issues: check `CAMERA_DEVICE` env var (default `/dev/video0`)
- TensorFlow/ONNX runtime detection: logs at startup which backend loaded

## Model File Organization

**Stored locations:**
- `ml-controller/model_store/` — production-ready models (shared with agents)
- `ml-controller/new_models/` — user uploads & experimental models
- `/data/models/` on agent — deployed model at `current_model.tflite` or `current_model.onnx`

**File naming:**
- Case-insensitive, spaces/dashes/underscores treated as separators
- Extensions: `.tflite`, `.onnx`, `.pt`, `.pth`, `.bin`
- Examples: `mobilenetv3_small_075.onnx` matches "MobileNetV3 Small 075"

## Data Flow Examples

### Energy Prediction for Custom Model

```
User uploads model specs (params_m, gflops, size_mb, latency_avg_s)
  → POST /api/predict-energy
  → EnergyPredictorService.predict() selects device-specific model
  → Returns prediction_mwh + CI + recommendation
  → Controller suggests: "excellent" (deploy) or "not_recommend" (optimize)
```

### Fall Detection Workflow

```
User deploys MoveNet model to Jetson
  → Agent loads TFLite model, tagged as fall_detection_pose
  → User calls /camera/fall-detect with duration_s=2.5
  → Agent captures frames from /dev/video0
  → extract_keypoints() → analyze_pose() → return fall_score
  → User calls /api/medical/fall-events to view history
  → Controller aggregates events with timestamps
```

### Telemetry Collection

```
Agent runs inference with FNB58 meter active
  → FNB58Reader collects voltage/current/energy
  → Agent sends POST /api/energy/report with measured_mwh
  → Controller logs to energy_measurements.json
  → benchmark_mape.py compares predicted_mwh vs measured_mwh
  → Reports stored in artifacts/report_figures/
```

## Testing & Validation

- **Unit tests:** None currently; validation via manual testing or notebooks
- **Integration tests:** `ur_fall_batch_test.py` for fall detection
- **Benchmarking:** `energy_benchmark_workbench.py` for local latency/throughput
- **MAPE validation:** `benchmark_mape.py` on agent with FNB58 telemetry

## External Dependencies

**Core:**
- Flask 3.0.0 — web framework
- NumPy, Pandas, scikit-learn — analytics
- TensorFlow 2.13.1 / ONNX Runtime 1.16+ — inference
- OpenCV (headless) — image processing

**Optional:**
- Torch, Timm — model download/conversion (controller only)
- PySerial, PyUSB — FNB58 reader
- Matplotlib, Seaborn — plotting
