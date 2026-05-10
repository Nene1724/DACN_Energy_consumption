# Energy-Aware Adaptive Edge AI Deployment

**DACN Thesis — University of Information Technology (UIT)**  
Authors: Tran Thu Ngan (22520937), Ho Thi Huynh My (22520837)  
Advisor: Le Minh Khanh Hoi

---

## What Was Built

A complete controller–agent platform for energy-aware ML model deployment on IoT edge devices (Jetson Nano, Raspberry Pi 5). The system integrates energy prediction, OTA deployment via Balena Cloud, runtime telemetry, rollback, and medical fall detection.

---

## Energy Prediction Model Results

Trained on 613 benchmark samples (360 Jetson Nano + 253 RPi5), 26 features per model, using **XGBoost (device-specific)**:

| Device | RMSE | MAE | Median MAPE | R² | Pearson r |
|--------|------|-----|-------------|-----|-----------|
| Jetson Nano | 177.335 mWh | 64.058 mWh | 14.28% | 0.9359 | 0.9748 |
| Raspberry Pi 5 | 5.628 mWh | 3.832 mWh | 8.87% | 0.9826 | 0.9937 |
| **Combined (123 held-out)** | **135.726 mWh** | **39.086 mWh** | **12.04%** | **0.9553** | **0.9826** |

XGBoost with 26 engineered features significantly outperforms the prior Extra Trees baseline (R²: 0.796 → 0.955).

---

## Adaptive vs. Static Deployment Results

| Mode | N | Mean Latency (ms) | Mean Energy (mWh) | vs. Static |
|------|----|-------------------|-------------------|------------|
| Static | 592 | 38,943.2 | 204.837 | baseline |
| **Adaptive** | **47** | **5,247.3** | **4.315** | **−86.5% latency, −97.9% energy** |

Total: 639 logged deployment runs (370 Jetson Nano + 269 CPU-replay).

---

## System Architecture

```
┌─────────────────────────────────────────────┐
│          ML Controller (Windows/Linux)       │
│               Flask  port 5000               │
│  ┌──────────┐  ┌──────────┐  ┌───────────┐  │
│  │Deployment│  │Monitoring│  │ Analytics │  │
│  └──────────┘  └──────────┘  └───────────┘  │
│  ┌──────────┐  ┌──────────────────────────┐  │
│  │ Medical  │  │  Energy Predictor (XGB)  │  │
│  └──────────┘  └──────────────────────────┘  │
└─────────────────┬───────────────────────────┘
                  │ REST / Balena OTA
     ┌────────────┴───────────┐
     ▼                        ▼
┌──────────────┐       ┌──────────────┐
│ Jetson Nano  │       │  Raspberry   │
│  ML Agent    │       │   Pi 5       │
│  port 8000   │       │  ML Agent    │
│  TFLite/ONNX │       │  port 8000   │
│  FNB58 meter │       └──────────────┘
└──────────────┘
```

---

## Project Structure

```
DACN_Energy_consumption/
├── ml-controller/
│   ├── python/
│   │   ├── app.py                         # 40+ REST endpoints
│   │   ├── energy_predictor_service.py    # XGBoost device-specific prediction
│   │   ├── model_analyzer.py              # Benchmark CSV loader
│   │   └── onnx_feature_extractor.py      # Auto-extract features from ONNX/TFLite
│   ├── templates/                         # Web UI (Deployment, Monitoring, Medical, Analytics)
│   ├── data/                              # 613-sample benchmark CSVs + deployment logs
│   ├── artifacts/                         # Trained XGBoost .pkl models + thresholds
│   │   ├── jetson_energy_model.pkl
│   │   ├── rpi5_energy_model.pkl
│   │   └── device_specific_metadata.json
│   ├── model_store/                       # Pre-loaded model artifacts
│   └── notebooks/                         # Training notebooks (XGBoost, multi-device)
│
├── jetson-ml-agent/
│   ├── app/
│   │   ├── server.py                      # 30+ REST endpoints
│   │   ├── movenet_fall_detection.py      # Pose estimation + fall scoring
│   │   ├── fnb58_*.py                     # FNB58 power meter (serial/USB/HID backends)
│   │   └── fnb58_telemetry_collector.py
│   ├── Dockerfile                         # Balena build for Jetson Nano
│   └── docker-compose.yml
└── start_web.ps1                          # Windows: start controller
```

---

## Quick Start

### Run the Controller (Windows)

```powershell
# Option 1: PowerShell script
.\start_web.ps1

# Option 2: Manual
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install -r ml-controller/requirements.txt
.\.venv\Scripts\python ml-controller/python/app.py
# Opens at http://localhost:5000
```

**Required environment variables** (`ml-controller/.env`):

```env
BALENA_API_TOKEN=your_token_here
CONTROLLER_PUBLIC_URL=http://your-pc-ip:5000
```

### Run the Agent Locally

```bash
cd jetson-ml-agent
docker-compose up --build
# Agent at http://localhost:8000
```

### SSH to Jetson Nano

```bash
balena device ssh 7c5c930
```

### Deploy a Model (Full Workflow)

1. Open `http://localhost:5000`
2. Go to **Deployment** tab → Refresh Balena devices
3. Upload `.onnx` or `.tflite` (features auto-extracted from file headers)
4. Click **Predict Energy** — XGBoost returns mWh prediction + confidence interval
5. Click **Upload And Deploy** — OTA push via Balena Cloud
6. Monitor canary results in Deployment Logs
7. Switch to **Monitoring** to track live energy telemetry from FNB58


### Compile the Paper

```bash
cd Paper
pdflatex -interaction=nonstopmode ver2.tex
pdflatex -interaction=nonstopmode ver2.tex   # run twice for cross-references
```

---

## Web Interface Views

| View | URL | Purpose |
|------|-----|---------|
| **Deployment** | `/` | Upload model → predict energy (XGBoost) → OTA deploy to device |
| **Monitoring** | `/monitoring` | Live device metrics, predicted vs. measured energy, FNB58 readings |
| **Medical** | `/medical` | MoveNet fall detection, live camera, fall event timeline |
| **Analytics** | `/analytics` | Model leaderboard, energy distribution, Pareto frontier, device comparison |

---

## Key API Endpoints

### Controller (port 5000)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/models/all` | GET | All benchmarked models with energy predictions |
| `/api/predict-energy` | POST | Predict energy given model feature dict |
| `/api/deploy` | POST | OTA deploy to Balena device |
| `/api/balena/devices` | GET | List connected Balena devices |
| `/api/analytics/summary` | GET | Leaderboard, Pareto frontier, device stats |
| `/api/energy/report` | POST | Receive measured energy telemetry from agent |
| `/api/medical/fall-events` | GET | Fall detection event history |
| `/api/device/benchmark` | POST | Trigger on-device inference benchmark |

### Agent (port 8000)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/status` | GET | Agent health check |
| `/deploy` | POST | Receive model URL, download and load |
| `/predict` | POST | Run inference (dummy or camera input) |
| `/benchmark` | POST | Benchmark model (latency, throughput, energy) |
| `/camera/fall-watch/start` | POST | Start continuous fall detection |
| `/camera/fall-watch/stop` | POST | Stop fall detection |
| `/camera/stream` | GET | MJPEG live camera stream |
| `/fnb58/status` | GET | FNB58 power meter connection status |
| `/telemetry` | GET | Energy history from device |

---

## Hardware Requirements

| Component | Details |
|-----------|---------|
| Controller PC | Windows 10/11 or Linux, Python 3.11 |
| Jetson Nano | JetPack 4.6, L4T r32.7, 4 GB RAM |
| Raspberry Pi 5 | Raspberry Pi OS Bookworm, 8 GB RAM |
| Power meter | FNIRSI FNB58 (USB-C, serial/HID) |
| Camera | USB webcam at `/dev/video0` (Sonix 1b17:0211 tested) |
| OTA platform | Balena Cloud (free tier: up to 10 devices) |

---

## Datasets

| Dataset | Samples | Description |
|---------|---------|-------------|
| `360_models_benchmark_jetson.csv` | 360 | Jetson Nano: latency, energy, throughput per model |
| `253_models_benchmark_rpi5.csv` | 253 | RPi5: same schema |
| `deployment_logs.json` | 639 | OTA deployment runs with mode, energy, latency |
| UR Fall Dataset | 70 sequences | Fall/non-fall MP4s for MoveNet evaluation |

---

## Device Details (Jetson Nano — UUID: 7c5c930)

- **SSH**: `balena device ssh 7c5c930`
- **USB Webcam**: Sonix Technology Co. (ID 1b17:0211) at `/dev/video0`
- **OS**: Linux 4.9.253-l4t-r32.7 (NVIDIA L4T)
- **Agent**: Balena container, port 8000, auto-restart on reboot
