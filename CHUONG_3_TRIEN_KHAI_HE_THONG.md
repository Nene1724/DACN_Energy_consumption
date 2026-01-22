# CHƯƠNG 3: TRIỂN KHAI HỆ THỐNG

## I. Xây Dựng Bộ Dữ Liệu và Huấn Luyện Mô Hình

### 1.1 Quy Trình Thu Thập Dữ Liệu (Data Collection)

Để xây dựng mô hình dự báo năng lượng chính xác, nhóm không sử dụng các bộ dữ liệu lý thuyết có sẵn mà thực hiện đo đạc thực tế (benchmarking) trên thiết bị phần cứng. Quy trình này tự động hóa hoàn toàn thông qua các script Python.

#### 1.1.1 Thiết Bị Thực Nghiệm

| Thiết Bị | Năm | Specs | Số Mẫu |
|----------|-----|-------|---------|
| **NVIDIA Jetson Nano 2GB** | 2019 | Tegra X1 (4-core ARM), 2GB RAM, CUDA GPU | 247 |
| **Raspberry Pi 5** | 2023 | BCM2712 (4-core ARM 64-bit), 4GB RAM, No GPU | 27 |
| **BeagleBone Black** | 2013 | AM3358 (Cortex-A8), 512MB RAM | Sẵn sàng |

**Tổng cộng: 274 mô hình benchmark thực tế** 📊

#### 1.1.2 Phương Pháp Đo Đạc

**Quy Trình Benchmark:**

```
Mỗi mô hình:
    ↓
[Chuẩn bị input]
├─ Kích thước chuẩn (224x224, 512x512,...)
├─ Batch size = 1 (inference đơn)
└─ Input format: Float32 (FP32)
    ↓
[Chạy 100 lần inference]
├─ Lần 1: Warm-up (loại bỏ do overhead đầu tiên)
├─ Lần 2-100: Ghi dữ liệu
└─ Đo sau mỗi lần: CPU, RAM, Temperature, Power
    ↓
[Tính toán chỉ số]
├─ Latency trung bình (ms)
├─ Latency min/max (ms)
├─ Throughput (inferences/sec) = 1 / latency_avg
├─ Công suất trung bình (mW)
├─ Năng lượng tích lũy (mWh)
└─ Power usage variation (std dev)
    ↓
[Lưu vào CSV]
└─ model, params_m, gflops, gmacs, size_mb, latency, energy, ...
```

**Công Cụ Đo Đạc:**

| Thiết Bị | Công Cụ | Phương Pháp |
|----------|---------|-----------|
| **Tất cả thiết bị** | FNB58 (Power Meter) | Đo trực tiếp Voltage/Current/Power/Energy (inline) — sử dụng để thu thập dataset |
| **Jetson Nano** | tegrastats | Read từ `/sys/devices/virtual/thermal/` (nhiệt độ/clock) |
| **Jetson Nano** | jtop (GPU monitoring) | Python library jtop |
| **Raspberry Pi 5** | vcgencmd | Frequency + voltage monitoring |
| **Raspberry Pi 5** | psutil | CPU/RAM/Temperature sampling |

#### 1.1.3 Dataset Chuẩn Hóa

**Ví Dụ Data từ 247_models_benchmark_jetson.csv:**

```csv
model,params_m,gflops,gmacs,size_mb,latency_avg_s,throughput_iter_per_s,energy_avg_mwh,power_avg_mw,device
mobilenetv3_small_050,1.53,0.024,0.012,6.1,0.008,125.0,18.5,2310,jetson_nano_2gb
mobilenetv3_small_075,2.04,0.087,0.044,8.5,0.012,83.3,28.4,2365,jetson_nano_2gb
mobilenetv3_small_100,2.54,0.109,0.055,10.3,0.015,66.7,35.2,2350,jetson_nano_2gb
edgenext_xx_small,1.33,0.031,0.016,3.8,0.009,111.1,19.8,2200,jetson_nano_2gb
ghostnet_100,5.17,0.086,0.043,10.8,0.015,67.0,42.3,2820,jetson_nano_2gb
efficientnet_b0,5.28,0.389,0.195,29.3,0.042,23.8,125.8,2995,jetson_nano_2gb
resnet18,11.69,1.814,0.909,45.0,0.089,11.2,387.2,4350,jetson_nano_2gb
```

**Công Thức Tính Năng Lượng:**

$$E(\text{mWh}) = P(\text{mW}) \times t(\text{s}) \div 3600$$

Lưu ý: Trong bộ dữ liệu, trường `energy_avg_mwh` được đọc trực tiếp từ thiết bị đo FNB58 (tổng năng lượng ghi nhận trong suốt phiên benchmark). Công thức dưới đây được dùng để đối chiếu/kiểm chứng chéo.

Ví dụ: Mobilenetv3 Small 0.5x trên Jetson
- Average Power: 2310 mW
- Inference Time: 0.008 s
- Energy = 2310 × 0.008 / 3600 = **0.00513 mWh** (1 inference)
- Per 100 inferences ≈ **0.513 mWh**

### 1.2 Huấn Luyện và Đánh Giá Mô Hình

#### 1.2.1 Feature Engineering (Kỹ Thuật Đặc Trưng)

**Input Features (6 cơ bản):**

```python
# From model metadata
params_m          # Số tham số (triệu)
gflops            # Floating-point operations (tỷ)
gmacs             # Multiply-accumulate operations (tỷ)
size_mb           # File size (MB)
latency_avg_s     # Inference time (s)
throughput_iter_per_s  # Inferences per second
```

**Derived Features (6 phát sinh):**

```python
# Computed from basic features
gflops_per_param = gflops / (params_m + 1e-6)
gmacs_per_mb = gmacs / (size_mb + 1e-6)
latency_throughput_ratio = latency_avg_s * throughput_iter_per_s
compute_intensity = gflops * latency_avg_s
model_complexity = params_m * gflops
computational_density = gflops / (size_mb + 1e-6)
```

**Total: 12 Features** → Đầu vào cho Gradient Boosting

#### 1.2.2 Feature Engineering Code

**File: Từ Jupyter Notebook**

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv('247_models_benchmark_jetson.csv')

# Basic validation
print(f"Dataset shape: {df.shape}")  # (247, 12)
print(f"Missing values: {df.isnull().sum()}")

# Feature Engineering
df['gflops_per_param'] = df['gflops'] / (df['params_m'] + 1e-6)
df['gmacs_per_mb'] = df['gmacs'] / (df['size_mb'] + 1e-6)
df['latency_throughput_ratio'] = df['latency_avg_s'] * df['throughput_iter_per_s']
df['compute_intensity'] = df['gflops'] * df['latency_avg_s']
df['model_complexity'] = df['params_m'] * df['gflops']
df['computational_density'] = df['gflops'] / (df['size_mb'] + 1e-6)

# Select features for model
feature_names = [
    'params_m', 'gflops', 'gmacs', 'size_mb', 
    'latency_avg_s', 'throughput_iter_per_s',
    'gflops_per_param', 'gmacs_per_mb', 'latency_throughput_ratio',
    'compute_intensity', 'model_complexity', 'computational_density'
]

X = df[feature_names].values
y = df['energy_avg_mwh'].values

print(f"Features shape: {X.shape}")  # (247, 12)
print(f"Target shape: {y.shape}")    # (247,)
```

#### 1.2.3 Model Training

**Algorithm: Gradient Boosting Regressor**

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, r2_score
import pickle

# Train-Test Split (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Train samples: {len(X_train)}")  # ~198
print(f"Test samples: {len(X_test)}")    # ~50

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Training
model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    verbose=1
)

model.fit(X_train_scaled, y_train)

# Prediction
y_pred_test = model.predict(X_test_scaled)

# Evaluation Metrics
mape = mean_absolute_percentage_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)
mae = np.mean(np.abs(y_test - y_pred_test))
rmse = np.sqrt(np.mean((y_test - y_pred_test) ** 2))

print(f"\n=== JETSON NANO MODEL METRICS ===")
print(f"MAPE: {mape:.4f} ({mape*100:.2f}%)")
print(f"R²: {r2:.4f}")
print(f"MAE: {mae:.4f} mWh")
print(f"RMSE: {rmse:.4f} mWh")

# Save Model
with open('jetson_energy_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('jetson_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save Feature Names
with open('device_specific_features.json', 'w') as f:
    json.dump(feature_names, f)
```

#### 1.2.4 Kết Quả Training

| Metric | Jetson Nano (247 mẫu) | Raspberry Pi 5 (27 mẫu) |
|--------|----------------------|--------------------------|
| **MAPE** | 18.69% | 15.88% ✨ |
| **R²** | 0.8605 | 0.9463 ✨ |
| **MAE** | 24.5 mWh | 1.8 mWh |
| **RMSE** | 52.3 mWh | 2.1 mWh |
| **Test Samples** | 50 | Leave-One-Out CV (27) |

**Diễn Giải:**
- ✅ Jetson model: Dự báo trong vòng ±18.69% (khá tốt cho thiết bị nhúng)
- ✅ RPi5 model: Dự báo trong vòng ±15.88% (rất tốt, do ít biến động)
- ✅ R² > 0.85 đạt mục tiêu

---

## II. Hiện Thực Hóa ML Controller (Server)

### 2.1 Cấu Trúc Ứng Dụng Server

```
ml-controller/
├── python/
│   ├── app.py                           # Flask main app (1866 lines)
│   ├── energy_predictor_service.py      # ML prediction logic (195 lines)
│   ├── model_analyzer.py                # Recommendation engine (222 lines)
│   ├── log_manager.py                   # Deployment logging (220 lines)
│   ├── generate_popular_models_metadata.py  # Generate metadata
│   └── download_models.py               # Download from timm
│
├── templates/
│   ├── index.html                       # Single Page App Dashboard
│   ├── deployment.html                  # Deployment tab
│   ├── monitoring.html                  # Monitoring tab
│   └── analytics.html                   # Analytics tab
│
├── artifacts/
│   ├── jetson_energy_model.pkl          # Trained model
│   ├── jetson_scaler.pkl                # Feature scaler
│   ├── rpi5_energy_model.pkl
│   ├── rpi5_scaler.pkl
│   ├── device_specific_features.json
│   ├── device_specific_metadata.json
│   └── energy_thresholds.json
│
├── model_store/                         # Downloaded artifacts
│   ├── mobilenetv3_small_050.onnx
│   ├── mobilenetv3_small_075.onnx
│   ├── edgenext_xx_small.onnx
│   └── ... (14 models)
│
├── data/
│   ├── 247_models_benchmark_jetson.csv
│   ├── 27_models_benchmark_rpi5.csv
│   └── deployment_logs.json
│
├── requirements.txt
└── .env                                 # Configuration
```

### 2.2 API Endpoints Chính

**Flask Server Initialization:**

```python
# File: app.py (Dòng 1-50)

from flask import Flask, render_template, request, jsonify
from energy_predictor_service import EnergyPredictorService
from model_analyzer import ModelAnalyzer
from log_manager import LogManager
import os

app = Flask(__name__, 
    template_folder=os.path.join(os.path.dirname(__file__), '..', 'templates')
)

# Initialize services
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
MODEL_STORE_DIR = os.path.join(BASE_DIR, "model_store")
DATA_DIR = os.path.join(BASE_DIR, "data")
CSV_PATH = os.path.join(DATA_DIR, "247_models_benchmark_jetson.csv")
RPI5_CSV_PATH = os.path.join(DATA_DIR, "27_models_benchmark_rpi5.csv")
LOG_FILE_PATH = os.path.join(DATA_DIR, "deployment_logs.json")

predictor_service = EnergyPredictorService(ARTIFACTS_DIR)
analyzer = ModelAnalyzer(CSV_PATH, 
    predictor_service=predictor_service,
    model_store_dir=MODEL_STORE_DIR,
    rpi5_csv_path=RPI5_CSV_PATH
)
log_manager = LogManager(LOG_FILE_PATH, max_logs=50)
```

### 2.3 Giao Diện Dashboard

**HTML Structure (index.html):**

```html
<!DOCTYPE html>
<html>
<head>
    <title>IoT ML Energy Manager</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css">
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
        .navbar { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
        .card-header { background-color: #f8f9fa; font-weight: bold; }
        .energy-excellent { color: #28a745; }
        .energy-good { color: #ffc107; }
        .energy-high { color: #dc3545; }
        .metric-box { 
            border-left: 4px solid #667eea;
            padding: 15px;
            margin: 10px 0;
            background: #f8f9fa;
        }
    </style>
</head>
<body>
    <nav class="navbar navbar-dark">
        <span class="navbar-brand mb-0 h1">⚡ IoT ML Energy Manager</span>
    </nav>

    <div class="container-fluid mt-4">
        <!-- Tabs -->
        <ul class="nav nav-tabs" role="tablist">
            <li class="nav-item">
                <a class="nav-link active" data-bs-toggle="tab" href="#deployment">
                    🚀 Deployment
                </a>
            </li>
            <li class="nav-item">
                <a class="nav-link" data-bs-toggle="tab" href="#monitoring">
                    📊 Monitoring
                </a>
            </li>
            <li class="nav-item">
                <a class="nav-link" data-bs-toggle="tab" href="#analytics">
                    📈 Analytics
                </a>
            </li>
        </ul>

        <!-- Tab Content -->
        <div class="tab-content">
            <!-- Deployment Tab -->
            <div id="deployment" class="tab-pane fade show active">
                <div class="row mt-4">
                    <div class="col-md-6">
                        <div class="card">
                            <div class="card-header">
                                Device Configuration
                            </div>
                            <div class="card-body">
                                <div class="mb-3">
                                    <label class="form-label">Target Device:</label>
                                    <select id="device-select" class="form-select">
                                        <option value="jetson_nano_2gb">NVIDIA Jetson Nano 2GB</option>
                                        <option value="raspberry_pi5">Raspberry Pi 5</option>
                                        <option value="beaglebone_black">BeagleBone Black</option>
                                    </select>
                                </div>

                                <div class="mb-3">
                                    <label class="form-label">Device IP Address:</label>
                                    <input type="text" id="device-ip" 
                                        class="form-control" 
                                        placeholder="192.168.1.100">
                                </div>

                                <div class="mb-3">
                                    <label class="form-label">Energy Budget (mWh):</label>
                                    <input type="number" id="energy-budget" 
                                        class="form-control" 
                                        value="50" min="0">
                                </div>

                                <button class="btn btn-primary w-100" onclick="loadModels()">
                                    📦 Load Models
                                </button>
                            </div>
                        </div>
                    </div>

                    <div class="col-md-6">
                        <div class="card">
                            <div class="card-header">
                                Energy Prediction
                            </div>
                            <div class="card-body">
                                <div id="prediction-result">
                                    <p class="text-muted">Select a model to predict energy consumption</p>
                                </div>

                                <div id="prediction-details" style="display: none;">
                                    <div class="metric-box">
                                        <strong>Predicted Energy:</strong>
                                        <h3 id="pred-energy">-- mWh</h3>
                                    </div>

                                    <div class="metric-box">
                                        <strong>Confidence Interval (95%):</strong>
                                        <p id="pred-ci">--</p>
                                    </div>

                                    <div class="metric-box">
                                        <strong>Category:</strong>
                                        <p id="pred-category">--</p>
                                    </div>

                                    <button class="btn btn-success w-100" onclick="deployModel()">
                                        🚀 Deploy Model
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Model Selection -->
                <div class="row mt-4">
                    <div class="col-md-12">
                        <div class="card">
                            <div class="card-header">
                                Model Library (Filters)
                            </div>
                            <div class="card-body">
                                <div class="btn-group" role="group">
                                    <button type="button" class="btn btn-outline-primary" 
                                        onclick="filterModels('all')">
                                        All
                                    </button>
                                    <button type="button" class="btn btn-outline-success" 
                                        onclick="filterModels('recommended')">
                                        ⭐ Recommended
                                    </button>
                                    <button type="button" class="btn btn-outline-info" 
                                        onclick="filterModels('excellent')">
                                        🟢 < 50 mWh
                                    </button>
                                    <button type="button" class="btn btn-outline-warning" 
                                        onclick="filterModels('good')">
                                        🟡 < 30 MB
                                    </button>
                                </div>

                                <div id="model-list" class="mt-3">
                                    <!-- Models will be rendered here -->
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Deployment Logs -->
                <div class="row mt-4">
                    <div class="col-md-12">
                        <div class="card">
                            <div class="card-header">
                                Deployment Logs
                            </div>
                            <div class="card-body" style="max-height: 300px; overflow-y: auto;">
                                <div id="deployment-logs">
                                    <p class="text-muted">No logs yet</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Monitoring Tab -->
            <div id="monitoring" class="tab-pane fade">
                <div class="row mt-4">
                    <div class="col-md-4">
                        <div class="card">
                            <div class="card-body">
                                <h5>CPU Usage</h5>
                                <div id="cpu-chart" style="height: 200px;"></div>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="card">
                            <div class="card-body">
                                <h5>Memory Usage</h5>
                                <div id="memory-chart" style="height: 200px;"></div>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="card">
                            <div class="card-body">
                                <h5>Temperature</h5>
                                <div id="temp-chart" style="height: 200px;"></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Analytics Tab -->
            <div id="analytics" class="tab-pane fade">
                <div class="row mt-4">
                    <div class="col-md-12">
                        <div class="card">
                            <div class="card-header">
                                System Statistics
                            </div>
                            <div class="card-body">
                                <div class="row">
                                    <div class="col-md-3">
                                        <div class="metric-box">
                                            <strong>Total Deployments:</strong>
                                            <h3 id="stat-deployments">0</h3>
                                        </div>
                                    </div>
                                    <div class="col-md-3">
                                        <div class="metric-box">
                                            <strong>Success Rate:</strong>
                                            <h3 id="stat-success">0%</h3>
                                        </div>
                                    </div>
                                    <div class="col-md-3">
                                        <div class="metric-box">
                                            <strong>Avg Energy Predicted:</strong>
                                            <h3 id="stat-avg-energy">0 mWh</h3>
                                        </div>
                                    </div>
                                    <div class="col-md-3">
                                        <div class="metric-box">
                                            <strong>Online Devices:</strong>
                                            <h3 id="stat-devices">0</h3>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // Placeholder for JavaScript functions
        // loadModels(), filterModels(), deployModel(), etc.
    </script>
</body>
</html>
```

---

## III. Hiện Thực Hóa ML Agent và Tích Hợp Phần Cứng

### 3.1 Dockerization

**Dockerfile cho BBB Agent:**

```dockerfile
FROM balenalib/beaglebone-black-debian-python:3.9-bullseye-run

RUN apt-get update && apt-get install -y \
    ca-certificates \
    python3-numpy \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/app

COPY requirements.txt ./

RUN pip install --no-cache-dir --no-deps \
    tflite-runtime==2.14.0 && \
    pip install --no-cache-dir \
    flask==3.0.0 \
    requests==2.32.3 \
    psutil

COPY . .

EXPOSE 8000
ENV PYTHONUNBUFFERED=1

CMD ["python", "-u", "app/server.py"]
```

**docker-compose.yml:**

```yaml
version: '3.8'

services:
  ml-agent:
    build: .
    container_name: ml-agent
    restart: unless-stopped
    ports:
      - "8000:8000"
    volumes:
      - ./local_data/models:/data/models
    environment:
      - MODEL_DIR_OVERRIDE=/data/models
      - FLASK_ENV=production
      - PYTHONUNBUFFERED=1
    networks:
      - ml-network

networks:
  ml-network:
    driver: bridge
```

### 3.2 Edge Logic - Agent Implementation

**Quy Trình Agent:**

```
┌─────────────────────────────────┐
│    ML Agent (Flask Server)       │
├─────────────────────────────────┤
│ 1. Listen on /deploy endpoint    │
│    (Receive deployment command)  │
│                                 │
│ 2. Download model from server    │
│    POST {model_url}              │
│                                 │
│ 3. Load into TFLite/ONNX         │
│    runtime.allocate_tensors()    │
│                                 │
│ 4. Telemetry Thread             │
│    └─ Every 5s: collect metrics │
│       - CPU, RAM, Temp, Storage │
│       - Check energy budget      │
│       - Auto-stop if over budget │
│                                 │
│ 5. Inference Thread             │
│    └─ Run inference on input     │
│       - Batch size = 1           │
│       - Log latency & energy     │
└─────────────────────────────────┘
```

**Agent Code (server.py - Snippet):**

```python
# File: bbb-ml-agent/app/server.py

from flask import Flask, request, jsonify
import requests
import threading
import os
import json
import psutil
from datetime import datetime
import time

try:
    import tflite_runtime.interpreter as tflite
    TFLITE_AVAILABLE = True
except:
    TFLITE_AVAILABLE = False

app = Flask(__name__)

# Global state
STATE = {
    "model_name": None,
    "status": "idle",           # idle | downloading | ready | running | error
    "inference_active": False,
    "energy_metrics": {
        "budget_mwh": None,
        "latest_mwh": None,
        "avg_mwh": None,
        "status": "no_data",    # ok | over_budget
        "history": []           # Last 40 samples
    }
}

LOADED_INTERPRETER = None
MODEL_LOCK = threading.RLock()
MODEL_DIR = os.getenv("MODEL_DIR_OVERRIDE", "/data/models")
os.makedirs(MODEL_DIR, exist_ok=True)

@app.route("/status", methods=["GET"])
def get_status():
    """Health check endpoint"""
    return jsonify({
        "status": STATE.get("status"),
        "model_name": STATE.get("model_name"),
        "inference_active": STATE.get("inference_active"),
        "uptime_seconds": int(time.time()),
        "energy_metrics": STATE.get("energy_metrics"),
        "device_metrics": {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage("/").percent
        }
    })

@app.route("/deploy", methods=["POST"])
def deploy():
    """Deploy model endpoint"""
    global LOADED_INTERPRETER
    
    try:
        data = request.get_json()
        model_name = data.get("model_name")
        model_url = data.get("model_url")
        energy_budget = data.get("energy_budget_mwh")
        
        # Update state
        STATE["status"] = "downloading"
        STATE["model_name"] = model_name
        
        # Download model
        print(f"[AGENT] Downloading {model_name} from {model_url}")
        resp = requests.get(model_url, stream=True, timeout=60)
        resp.raise_for_status()
        
        # Save model
        model_path = os.path.join(MODEL_DIR, f"{model_name}.tflite")
        with open(model_path, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        file_size = os.path.getsize(model_path) / (1024 * 1024)
        print(f"[AGENT] Model saved: {file_size:.2f} MB")
        
        # Load interpreter
        with MODEL_LOCK:
            interpreter = tflite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            LOADED_INTERPRETER = interpreter
        
        # Update state
        STATE["status"] = "ready"
        STATE["energy_metrics"] = {
            "budget_mwh": energy_budget,
            "latest_mwh": None,
            "avg_mwh": None,
            "status": "ok",
            "history": []
        }
        
        return jsonify({
            "success": True,
            "message": f"Model {model_name} deployed successfully",
            "model_path": model_path,
            "file_size_mb": round(file_size, 2)
        })
    
    except Exception as e:
        STATE["status"] = "error"
        STATE["error"] = str(e)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route("/inference/start", methods=["POST"])
def start_inference():
    """Start inference loop"""
    data = request.get_json() or {}
    
    STATE["inference_active"] = True
    STATE["status"] = "running"
    
    # Start inference thread
    thread = threading.Thread(
        target=inference_loop,
        args=(data.get("iterations", 100),)
    )
    thread.daemon = True
    thread.start()
    
    return jsonify({
        "success": True,
        "message": "Inference started"
    })

@app.route("/inference/stop", methods=["POST"])
def stop_inference():
    """Stop inference loop"""
    STATE["inference_active"] = False
    STATE["status"] = "ready"
    
    return jsonify({
        "success": True,
        "message": "Inference stopped"
    })

def inference_loop(iterations):
    """Run inference with energy monitoring"""
    global LOADED_INTERPRETER
    
    with MODEL_LOCK:
        if LOADED_INTERPRETER is None:
            print("[AGENT] No model loaded")
            return
        
        interpreter = LOADED_INTERPRETER
    
    # Get input/output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"[AGENT] Starting inference: {iterations} iterations")
    
    energy_history = []
    
    for i in range(iterations):
        if not STATE["inference_active"]:
            break
        
        try:
            # Prepare input (random data for testing)
            import numpy as np
            input_shape = input_details[0]['shape']
            input_data = np.random.randn(*input_shape).astype(np.float32)
            
            # Run inference
            t_start = time.time()
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            t_end = time.time()
            
            latency_ms = (t_end - t_start) * 1000
            
            # Estimate energy (simple: latency × avg_power)
            avg_power_mw = 2500  # Rough estimate for BBB
            energy_mwh = (latency_ms / 1000) * avg_power_mw / 3600
            
            energy_history.append(energy_mwh)
            
            # Update metrics
            STATE["energy_metrics"]["latest_mwh"] = round(energy_mwh, 4)
            STATE["energy_metrics"]["avg_mwh"] = round(
                sum(energy_history[-40:]) / len(energy_history[-40:]), 4
            )
            
            # Check budget
            if STATE["energy_metrics"]["budget_mwh"]:
                total_energy = sum(energy_history)
                if total_energy > STATE["energy_metrics"]["budget_mwh"]:
                    print(f"[AGENT] ⚠️ Energy budget exceeded!")
                    STATE["energy_metrics"]["status"] = "over_budget"
                    STATE["inference_active"] = False
                    break
            
            print(f"[AGENT] Iter {i+1}: {latency_ms:.2f}ms, Energy: {energy_mwh:.4f}mWh")
        
        except Exception as e:
            print(f"[AGENT] Inference error: {e}")
            break
    
    STATE["status"] = "ready"
    print(f"[AGENT] Inference completed")

def telemetry_thread():
    """Collect system telemetry every 5 seconds"""
    while True:
        try:
            metrics = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "cpu_percent": psutil.cpu_percent(interval=0.5),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage("/").percent,
                "temperature_c": get_device_temp()
            }
            
            # Post to controller (optional)
            if STATE.get("status") == "running":
                try:
                    # requests.post("http://controller:5000/api/device/telemetry", json=metrics)
                    pass
                except:
                    pass
            
            time.sleep(5)
        
        except Exception as e:
            print(f"[TELEMETRY] Error: {e}")
            time.sleep(5)

def get_device_temp():
    """Get device temperature"""
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return int(f.read()) / 1000
    except:
        return 0

# Start telemetry thread
telemetry = threading.Thread(target=telemetry_thread, daemon=True)
telemetry.start()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
```

---

## IV. Kịch Bản Kiểm Thử End-to-End

### 4.1 Test Scenario 1: Dự Báo Năng Lượng

```
Bước 1: User chọn model "mobilenetv3_small_075" trên Dashboard
Bước 2: System tự động gọi /api/predict-energy
   Input:
   {
     "device_type": "jetson_nano_2gb",
     "model_name": "mobilenetv3_small_075",
     "params_m": 2.04,
     "gflops": 0.087,
     "gmacs": 0.044,
     "size_mb": 8.5,
     "latency_avg_s": 0.012,
     "throughput_iter_per_s": 83.3
   }

Bước 3: Server gọi energy_predictor_service.predict()
   └─ Load jetson_energy_model.pkl + jetson_scaler.pkl
   └─ Scale features
   └─ Run GradientBoosting.predict()
   └─ Tính confidence interval: ±18.69%
   └─ Classify: 28.4 mWh < 34.6 mWh → "EXCELLENT"

Bước 4: Return response:
   {
     "success": true,
     "data": {
       "predicted_energy_mwh": 28.4,
       "confidence_interval_lower": 24.1,
       "confidence_interval_upper": 32.7,
       "category": "EXCELLENT"
     }
   }

Bước 5: Dashboard hiển thị kết quả với icon 🟢
```

### 4.2 Test Scenario 2: End-to-End Deployment

```
Bước 1: User chọn device IP "192.168.1.100" (RPi5)
Bước 2: User chọn model + nhập energy budget 30 mWh
Bước 3: User nhấn "🚀 Deploy Model"

Bước 4: Server kiểm tra budget
   └─ Dự báo energy = 28.4 mWh < 30 mWh ✅

Bước 5: Server resolve artifact
   └─ Tìm "mobilenetv3_small_075.onnx" trong model_store/

Bước 6: Server gọi /deploy trên device
   POST http://192.168.1.100:8000/deploy
   {
     "model_name": "mobilenetv3_small_075",
     "model_url": "http://192.168.1.36:5000/models/mobilenetv3_small_075.onnx",
     "energy_budget_mwh": 30.0
   }

Bước 7: Agent nhận lệnh
   └─ Download model từ server (streaming)
   └─ Load vào ONNX runtime
   └─ Update state = "ready"
   └─ Return 200 OK

Bước 8: Server ghi log
   └─ Deployment successful
   └─ Duration: 45 seconds
   └─ File size: 8.5 MB

Bước 9: Dashboard hiển thị status "Ready for inference"
```

---

## V. Tóm Tắt Chương 3

✅ **Dataset Collection**
- 247 models benchmark thực tế trên Jetson Nano
- 27 models benchmark trên Raspberry Pi 5
- Thiết bị đo năng lượng: FNB58 (inline power meter)
- Công cụ telemetry: tegrastats, vcgencmd, psutil

✅ **Feature Engineering**
- 6 features cơ bản + 6 features phát sinh = 12 total
- StandardScaler chuẩn hóa features

✅ **Model Training**
- Algorithm: Gradient Boosting Regressor
- Jetson: MAPE 18.69%, R² 0.86
- RPi5: MAPE 15.88%, R² 0.95

✅ **ML Controller**
- Flask server với 20+ API endpoints
- 3 HTML dashboard tabs: Deployment, Monitoring, Analytics
- EnergyPredictorService cho dự báo

✅ **ML Agent**
- Docker containerization
- Dual threads: Inference + Telemetry
- Energy budget enforcement

✅ **End-to-End Testing**
- Scenario 1: Dự báo năng lượng
- Scenario 2: Triển khai tự động
- Scenario 3: Giám sát real-time

---

**→ Tiếp theo: Chương 4 sẽ trình bày kết quả thực nghiệm chi tiết**
