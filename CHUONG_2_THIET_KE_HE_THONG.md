# CHƯƠNG 2: THIẾT KẾ HỆ THỐNG

## I. Kiến Trúc Tổng Thể (System Architecture)

### 1.1 Tổng Quan Kiến Trúc

Để giải quyết bài toán quản lý tập trung và triển khai tự động trên nhiều thiết bị biên không đồng nhất, hệ thống được thiết kế theo mô hình **Client-Server** kết hợp với kiến trúc **Containerization**. Kiến trúc này đảm bảo tính tách biệt giữa logic quản lý (tại Server) và môi trường thực thi suy luận (tại Edge Device).

#### Thành Phần Chính

```
┌────────────────────────────────────────────────────────────────┐
│                    NGƯỜI DÙNG (Web Browser)                    │
│                   http://localhost:5000                        │
└──────────────────┬─────────────────────────────────────────────┘
                   │ HTTP/HTTPS
                   ▼
        ┌──────────────────────────────┐
        │   ML CONTROLLER (SERVER)     │
        │  ┌──────────────────────────┐ │
        │  │  Web Dashboard (Flask)   │ │
        │  │  - UI Deployment        │ │
        │  │  - UI Monitoring        │ │
        │  │  - UI Analytics         │ │
        │  └──────────────────────────┘ │
        │  ┌──────────────────────────┐ │
        │  │  RESTful API Endpoints   │ │  20+ endpoints
        │  │  - Models Management    │ │
        │  │  - Energy Prediction    │ │
        │  │  - Deployment Control   │ │
        │  │  - Device Monitoring    │ │
        │  └──────────────────────────┘ │
        │  ┌──────────────────────────┐ │
        │  │  Core Services           │ │
        │  │  - EnergyPredictorService│ │
        │  │  - ModelAnalyzer        │ │
        │  │  - LogManager           │ │
        │  └──────────────────────────┘ │
        │  ┌──────────────────────────┐ │
        │  │  Data & Artifacts        │ │
        │  │  - artifacts/            │ │
        │  │  - model_store/          │ │
        │  │  - data/                 │ │
        │  └──────────────────────────┘ │
        └──────────────────────────────┘
                   │ HTTP REST API
         ┌─────────┼─────────┬──────────┐
         │         │         │          │
         ▼         ▼         ▼          ▼
    ┌─────────┐ ┌──────┐ ┌──────────┐ ┌──────────┐
    │ Jetson  │ │ RPi  │ │ BBB      │ │ Other    │
    │ Nano 2G │ │ 5    │ │ Black    │ │ Devices  │
    │         │ │      │ │          │ │          │
    │┌───────┐│ │┌────┐│ │┌────────┐│ │┌────────┐│
    ││Docker ││ ││Dock││ ││ Docker ││ ││ Docker ││
    ││ ML    ││ ││ ML ││ ││  ML    ││ ││  ML    ││
    ││Agent  ││ ││Agent││ ││ Agent  ││ ││ Agent  ││
    ││:8000  ││ ││:8000││ ││ :8000  ││ ││ :8000  ││
    │└───────┘│ │└────┘│ │└────────┘│ │└────────┘│
    │CUDA GPU │ │ARM   │ │ARM CPU   │ │          │
    │TFLite   │ │TFLite│ │TFLite    │ │          │
    └─────────┘ └──────┘ └──────────┘ └──────────┘
         ▲         ▲         ▲          ▲
         │         │         │          │
         └─────────┴─────────┴──────────┘
           Quản lý từ xa qua Balena Cloud
```

### 1.2 Mô Tả Các Thành Phần

| Thành Phần | Vị Trí | Chức Năng | Công Nghệ |
|-----------|--------|----------|-----------|
| **ML Controller** | Máy tính chủ / Cloud | Quản lý tập trung, dự đoán năng lượng, giao diện web | Flask, Python |
| **ML Agent** | Thiết bị IoT (Jetson/RPi/BBB) | Nhận lệnh, thực thi inference, báo cáo telemetry | Docker, Python, TFLite/ONNX |
| **Balena Cloud** | Cloud (Balena) | Quản lý fleet device từ xa, push updates | Balena OS |
| **Model Store** | Local disk (Controller) | Lưu trữ model artifacts (.onnx, .tflite) | File system |
| **Database** | File-based JSON | Logs, deployment history, device metrics | JSON files |

---

## II. Phân Tích Các Phân Hệ Chức Năng

### 2.1 Phân Hệ Thu Thập & Chuẩn Hóa Dữ Liệu (Data Collection & Normalization)

#### 2.1.1 Mục Đích

Đây là nền tảng đầu vào cho mô hình dự báo. Phân hệ này thực hiện:

```
Thiết bị IoT → Chạy Models → Đo Metrics (Power, Energy, Latency...)
                    ↓
            Trích Xuất Features
                    ↓
            Chuẩn Hóa & Validate
                    ↓
            Lưu CSV (Dataset)
```

#### 2.1.2 Quy Trình Chi Tiết

**Bước 1: Benchmark Tự Động**

Hệ thống tự động chạy 247 mô hình trên Jetson Nano, đo đạc:
- Năng lượng tiêu thụ (mWh)
- Công suất (mW)
- Độ trễ (ms)
- Thông lượng (inferences/sec)

**Ví dụ dữ liệu benchmark:**

```csv
model,params_m,gflops,gmacs,size_mb,latency_avg_s,throughput_iter_per_s,energy_avg_mwh,power_avg_mw,device
mobilenetv3_small_050,1.53,0.024,0.012,6.1,0.008,125.0,18.5,2310,jetson_nano_2gb
ghostnet_100,5.17,0.086,0.043,10.8,0.015,67.0,42.3,2820,jetson_nano_2gb
efficientnet_b0,5.28,0.389,0.195,29.3,0.042,23.8,125.8,2995,jetson_nano_2gb
resnet18,11.69,1.814,0.909,45.0,0.089,11.2,387.2,4350,jetson_nano_2gb
```

**Bước 2: Trích Xuất Đặc Trưng (Feature Extraction)**

Từ file model được download, hệ thống tự động tính:

| Đặc Trưng | Công Thức | Ý Nghĩa |
|----------|-----------|---------|
| **params_m** | Từ model metadata | Số tham số (triệu) |
| **gflops** | MACs × 2 / input_size² | Phép tính FP32 (tỷ) |
| **gmacs** | MACs (từ FLOPS counter) | Phép tính multiply-accumulate (tỷ) |
| **size_mb** | File size | Kích thước model (MB) |
| **latency_avg_s** | Trung bình 100 runs | Thời gian inference (giây) |
| **throughput_iter_per_s** | 1 / latency_avg_s | Số inferences mỗi giây |
| **gflops_per_param** | gflops / params_m | Computational efficiency |
| **gmacs_per_mb** | gmacs / size_mb | Memory efficiency |
| **latency_throughput_ratio** | latency × throughput | Temporal consistency |
| **compute_intensity** | gflops × latency | Computational intensity |
| **model_complexity** | params_m × gflops | Độ phức tạp tổng thể |
| **computational_density** | gflops / size_mb | Mật độ tính toán |

**Bước 3: Chuẩn Hóa Dữ Liệu**

Dữ liệu từ các thiết bị khác nhau được quy về cùng một đơn vị chuẩn:

```python
# Ví dụ từ code:
df['latency_avg_s'] = df['latency_avg_ms'] / 1000  # ms → s
df['energy_avg_mwh'] = df['energy_avg_mj'] / 3.6   # mJ → mWh
df['power_avg_mw'] = df['power_avg_mw']             # Đã chuẩn hóa
```

**Bước 4: Lưu Trữ**

Dữ liệu được lưu dưới dạng CSV với metadata:

```
ml-controller/
├── data/
│   ├── 247_models_benchmark_jetson.csv (247 models × 12 features)
│   ├── 27_models_benchmark_rpi5.csv    (27 models × 12 features)
│   └── deployment_logs.json            (lịch sử deploy)
```

### 2.2 Phân Hệ Dự Báo Năng Lượng (Energy Predictor Service)

#### 2.2.1 Kiến Trúc

```
Input: Model Features + Device Type
         ↓
    Device-Aware Routing
    ├─→ Jetson Nano? → Jetson Model + Jetson Scaler
    ├─→ Raspberry Pi 5? → RPi5 Model + RPi5 Scaler
    └─→ Unknown? → Unified Fallback Model
         ↓
    Feature Engineering (từ 6 features → 12 features)
         ↓
    StandardScaler.transform(features)
         ↓
    GradientBoostingRegressor.predict(scaled_features)
         ↓
    Post-processing (confidence interval, classification)
         ↓
Output: Energy (mWh) + Confidence Interval + Category
```

#### 2.2.2 Code Thực Tế - Dự Báo Năng Lượng

**File: `energy_predictor_service.py`**

```python
class EnergyPredictorService:
    def __init__(self, artifacts_dir: str):
        # Load device-specific models (PRODUCTION)
        self.jetson_model = self._load_pickle("jetson_energy_model.pkl")
        self.jetson_scaler = self._load_pickle("jetson_scaler.pkl")
        self.rpi5_model = self._load_pickle("rpi5_energy_model.pkl")
        self.rpi5_scaler = self._load_pickle("rpi5_scaler.pkl")
        
        # Load unified fallback model
        self.unified_model = self._load_pickle("energy_predictor.pkl")
        self.unified_scaler = self._load_pickle("energy_scaler.pkl")
        
        # Load MAPE for confidence intervals
        self.jetson_mape = 0.1869  # 18.69%
        self.rpi5_mape = 0.1588    # 15.88%
        self.unified_mape = 0.50   # Conservative fallback

    def predict(self, payloads: List[Dict]) -> List[Dict]:
        """
        Batch prediction với device-aware routing.
        
        Input payload example:
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
        """
        results = []
        
        for payload in payloads:
            # Extract device type
            device_type = payload.get("device_type", "")
            
            # Select appropriate model
            model, scaler, mape = self._select_model_and_scaler(device_type)
            
            # Build feature row
            df_row = self._build_feature_row(payload)
            
            # Scale features
            df_scaled = scaler.transform(df_row)
            
            # Predict energy
            energy_pred = model.predict(df_scaled)[0]
            
            # Calculate confidence interval (95%)
            ci_margin = energy_pred * mape * 1.96  # 95% CI
            ci_lower = max(energy_pred - ci_margin, 0)
            ci_upper = energy_pred + ci_margin
            
            # Classify energy level
            category = self._classify_energy(energy_pred, device_type)
            
            results.append({
                "model_name": payload.get("model_name"),
                "device_type": device_type,
                "predicted_energy_mwh": round(energy_pred, 2),
                "confidence_interval_lower": round(ci_lower, 2),
                "confidence_interval_upper": round(ci_upper, 2),
                "confidence_percent": 95,
                "category": category,
                "model_accuracy_mape": mape * 100
            })
        
        return results

    def _classify_energy(self, energy_mwh: float, device_type: str) -> str:
        """Phân loại mức năng lượng: EXCELLENT / GOOD / ACCEPTABLE / HIGH"""
        device_type_lower = device_type.lower()
        
        if "jetson" in device_type_lower:
            thresholds = {
                "EXCELLENT": 34.6,   # P25
                "GOOD": 104.6,       # P50
                "ACCEPTABLE": 235.3  # P75
            }
        elif "rpi" in device_type_lower or "raspberry" in device_type_lower:
            thresholds = {
                "EXCELLENT": 11.1,   # P25
                "GOOD": 18.0,        # P50
                "ACCEPTABLE": 30.7   # P75
            }
        else:
            thresholds = {
                "EXCELLENT": 50,
                "GOOD": 150,
                "ACCEPTABLE": 300
            }
        
        if energy_mwh <= thresholds["EXCELLENT"]:
            return "EXCELLENT"
        elif energy_mwh <= thresholds["GOOD"]:
            return "GOOD"
        elif energy_mwh <= thresholds["ACCEPTABLE"]:
            return "ACCEPTABLE"
        else:
            return "HIGH"
```

#### 2.2.3 Model Artifacts

| Tệp | Kích Thước | Mô Tả |
|-----|-----------|------|
| `jetson_energy_model.pkl` | ~100KB | GradientBoostingRegressor cho Jetson |
| `jetson_scaler.pkl` | ~10KB | StandardScaler (Jetson features) |
| `rpi5_energy_model.pkl` | ~100KB | GradientBoostingRegressor cho RPi5 |
| `rpi5_scaler.pkl` | ~10KB | StandardScaler (RPi5 features) |
| `device_specific_features.json` | ~1KB | Danh sách 12 features |
| `device_specific_metadata.json` | ~2KB | MAPE, R², training info |
| `energy_thresholds.json` | ~1KB | Ngưỡng percentile (P10/P25/P50/P75/P90) |

### 2.3 Phân Hệ Quản Lý Triển Khai (Deployment Manager)

#### 2.3.1 Quy Trình Triển Khai

```
User chọn Model
    ↓
[Energy Budget Check]
  ├─→ Vượt ngưỡng? → Cảnh báo/Từ chối
  └─→ OK? → Tiếp tục
    ↓
[Model Artifact Resolve]
  ├─→ Có sẵn? → Dùng
  └─→ Không? → Download từ timm
    ↓
[Format Conversion]
  ├─→ Convert sang TFLite (nếu cần)
  └─→ Convert sang ONNX (nếu cần)
    ↓
[Deploy to Device]
  ├─→ POST http://{device_ip}:8000/deploy
  ├─→ Device nhận lệnh
  ├─→ Device download model từ server
  ├─→ Device load model vào memory
  ├─→ Device báo cáo success/error
    ↓
[Log & Update State]
  ├─→ Ghi vào deployment_logs.json
  └─→ Update device state
```

#### 2.3.2 API Endpoint Triển Khai

**Endpoint: `POST /api/deploy`**

```python
@app.route("/api/deploy", methods=["POST"])
def deploy():
    """
    Triển khai model lên device IoT
    
    Request body:
    {
        "bbb_ip": "192.168.1.100",
        "model_name": "mobilenetv3_small_075",
        "max_energy": 50.0  # optional: energy budget in mWh
    }
    
    Response:
    {
        "success": true,
        "message": "Deployment initiated",
        "deployment_id": "deploy_20260118_124530",
        "device": {
            "ip": "192.168.1.100",
            "status": "downloading"
        },
        "model": {
            "name": "mobilenetv3_small_075",
            "predicted_energy_mwh": 28.4,
            "category": "EXCELLENT"
        }
    }
    """
    data = request.get_json()
    bbb_ip = data.get("bbb_ip")
    model_name = data.get("model_name")
    max_energy = data.get("max_energy")
    
    # 1. Resolve artifact
    artifact = resolve_model_artifact(model_name)
    if not artifact:
        return jsonify({"success": False, "error": "Model artifact not found"}), 404
    
    # 2. Build model URL
    pc_ip = get_local_ip_for_device(bbb_ip)
    model_url = f"http://{pc_ip}:5000/models/{artifact}"
    
    # 3. Prepare payload for agent
    model_info = analyzer.get_model_info(model_name)
    payload = {
        "model_name": model_name,
        "model_url": model_url,
        "model_info": model_info
    }
    if max_energy:
        payload["energy_budget_mwh"] = max_energy
    
    # 4. Send to device
    try:
        resp = requests.post(
            f"http://{bbb_ip}:8000/deploy",
            json=payload,
            timeout=30
        )
        resp.raise_for_status()
        
        # 5. Log deployment
        log_manager.add_log(
            log_type="info",
            message=f"Deployment initiated: {model_name} → {bbb_ip}",
            metadata={
                "model_name": model_name,
                "device_ip": bbb_ip,
                "energy_budget": max_energy
            }
        )
        
        return jsonify({
            "success": True,
            "device_ip": bbb_ip,
            "model_name": model_name
        })
    
    except Exception as e:
        log_manager.add_log(
            log_type="error",
            message=f"Deployment failed: {str(e)}",
            metadata={"model_name": model_name, "device_ip": bbb_ip}
        )
        return jsonify({"success": False, "error": str(e)}), 500
```

#### 2.3.3 Flow Triển Khai Chi Tiết

```
Controller (Server)                          Agent (Device)
    │                                             │
    │  1. User chọn model + device              │
    │                                             │
    ├─→ 2. Predict energy                       │
    │      (check budget)                         │
    │                                             │
    ├─→ 3. Resolve artifact                     │
    │      (.onnx / .tflite)                     │
    │                                             │
    ├─────→ 4. POST /deploy ─────────────────→ │
    │         {model_url, ...}                   │
    │                                             │
    │                                    5. Nhận lệnh
    │                                    6. POST model_url
    │←─────────────── 7. Download ──────────────┤
    │        (streaming model file)              │
    │                                    8. Save to disk
    │                                    9. Load interpreter
    │←───── 10. Response (success) ────────────  │
    │                                             │
    │ 11. Log success                            │
    │     Update dashboard                        │
```

### 2.4 Phân Hệ Giám Sát & Phản Hồi (Monitoring & Feedback)

#### 2.4.1 Kiến Trúc Monitoring

```
Device (Agent)           Controller (Server)       Dashboard (Browser)
    │                          │                          │
    ├─ Collect metrics ─→     │                          │
    │  - CPU                   │                          │
    │  - RAM                   ├─ Aggregate data         │
    │  - Temp                  │  - Store in memory      │
    │  - Storage               │                          │
    │  - Energy (if available) │  ├─ Serve via WebSocket ┤
    │                          │  └─ Real-time updates   │
    ├─ Every 5 seconds ─→     │                          ├─ Update charts
    │  POST /metrics ────→     │                          │  - CPU usage
    │                          │                          │  - RAM usage
    │                          │                          │  - Temperature
    │                          │                          │  - Energy trend
```

#### 2.4.2 API Endpoints Giám Sát

**Endpoint: `GET /api/device/metrics`**

```python
@app.route("/api/device/metrics", methods=["GET"])
def get_device_metrics():
    """
    Lấy metrics hiện tại của device
    
    Response:
    {
        "device_ip": "192.168.1.100",
        "status": "ready",
        "model_name": "mobilenetv3_small_075",
        "metrics": {
            "cpu_usage_percent": 45.2,
            "ram_usage_mb": 512,
            "ram_usage_percent": 25,
            "temperature_c": 52.3,
            "storage_usage_mb": 2048,
            "uptime_seconds": 3600,
            "inference_active": false
        },
        "energy_metrics": {
            "budget_mwh": 50.0,
            "latest_mwh": 28.4,
            "avg_mwh": 27.8,
            "status": "ok"  # or "over_budget"
        },
        "timestamp": "2026-01-18T10:30:45.123Z"
    }
    """
    device_ip = request.args.get("device_ip")
    
    try:
        # Get telemetry từ device
        resp = requests.get(
            f"http://{device_ip}:8000/status",
            timeout=5
        )
        resp.raise_for_status()
        
        device_status = resp.json()
        
        # Format response
        return jsonify({
            "success": True,
            "device_ip": device_ip,
            "data": device_status,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "device_ip": device_ip
        }), 500
```

---

## III. Thiết Kế Cơ Sở Dữ Liệu & API

### 3.1 Cơ Sở Dữ Liệu

#### 3.1.1 File-Based JSON Storage

Do hệ thống được thiết kế cho edge/nhúng, không sử dụng database tập trung. Thay vào đó, dùng JSON files với cấu trúc:

**File: `data/deployment_logs.json`**

```json
{
  "logs": [
    {
      "timestamp": "2026-01-18T10:30:45.123Z",
      "type": "success",
      "message": "Deployment completed: mobilenetv3_small_075 → 192.168.1.100",
      "metadata": {
        "model_name": "mobilenetv3_small_075",
        "device_ip": "192.168.1.100",
        "energy_predicted_mwh": 28.4,
        "deployment_time_seconds": 45.2
      }
    },
    {
      "timestamp": "2026-01-18T10:31:20.456Z",
      "type": "error",
      "message": "Device offline: timeout connecting to 192.168.1.101",
      "metadata": {
        "device_ip": "192.168.1.101",
        "error_code": "TIMEOUT"
      }
    }
  ],
  "metadata": {
    "created_at": "2025-12-01T00:00:00Z",
    "last_updated": "2026-01-18T10:31:20.456Z",
    "total_deployments": 247,
    "success_count": 234,
    "error_count": 13,
    "success_rate": 94.7
  }
}
```

#### 3.1.2 Model Store Structure

```
ml-controller/
├── model_store/          # Model artifacts
│   ├── mobilenetv3_small_050.onnx
│   ├── mobilenetv3_small_075.onnx
│   ├── mobilenetv3_small_100.onnx
│   ├── edgenext_xx_small.onnx
│   ├── ghostnet_100.pth
│   ├── efficientnet_b0.pth
│   └── ... (14 models)
│
├── artifacts/            # Trained models
│   ├── jetson_energy_model.pkl
│   ├── jetson_scaler.pkl
│   ├── rpi5_energy_model.pkl
│   ├── rpi5_scaler.pkl
│   ├── device_specific_features.json
│   ├── device_specific_metadata.json
│   └── energy_thresholds.json
│
├── data/                 # Benchmark datasets
│   ├── 247_models_benchmark_jetson.csv
│   ├── 27_models_benchmark_rpi5.csv
│   └── deployment_logs.json
```

### 3.2 API Design

#### 3.2.1 Danh Sách Tất Cả Endpoints

| Loại | Endpoint | Mô Tả | Request | Response |
|------|----------|------|---------|----------|
| **Model Management** | | | | |
| GET | `/api/models/all` | Lấy tất cả models | `limit`, `offset` | [model_list] |
| GET | `/api/models/popular` | 15+ popular models | - | [popular_models] |
| GET | `/api/models/recommended` | Top 10 tiết kiệm nhất | `device_type` | [models] |
| POST | `/api/models/download` | Download model từ timm | `model_name` | `{success, path}` |
| **Energy Prediction** | | | | |
| POST | `/api/predict-energy` | Dự đoán năng lượng | `{device, features}` | `{energy, CI, category}` |
| GET | `/api/energy/thresholds` | Ngưỡng percentile | - | `{P10, P25, P50, P75, P90}` |
| GET | `/api/energy/metadata` | Metadata model | - | `{training_info, MAPE}` |
| **Deployment** | | | | |
| POST | `/api/deploy` | Deploy model | `{device_ip, model}` | `{success, deployment_id}` |
| GET | `/api/device/metrics` | Telemetry device | `device_ip` | `{CPU, RAM, Temp}` |
| POST | `/api/device/start` | Start inference | `{device_ip}` | `{success}` |
| **Logs & Monitoring** | | | | |
| GET | `/api/logs` | Deployment logs | `limit`, `type` | [logs] |
| POST | `/api/logs/clear` | Xóa logs | - | `{success}` |
| GET | `/api/logs/export` | Export logs | `format` | CSV/JSON |
| **Balena Integration** | | | | |
| GET | `/api/balena/devices` | Danh sách devices | `app_slug` | [devices] |
| POST | `/api/balena/deploy` | Deploy release | `{release_id}` | `{success}` |
| GET | `/api/balena/devices/<uuid>/logs` | Device logs | `limit` | [logs] |

#### 3.2.2 Ví Dụ Gọi API

**Ví dụ 1: Dự đoán năng lượng**

```bash
curl -X POST http://localhost:5000/api/predict-energy \
  -H "Content-Type: application/json" \
  -d '{
    "device_type": "jetson_nano_2gb",
    "model_name": "mobilenetv3_small_075",
    "params_m": 2.04,
    "gflops": 0.087,
    "gmacs": 0.044,
    "size_mb": 8.5,
    "latency_avg_s": 0.012,
    "throughput_iter_per_s": 83.3
  }'
```

**Response:**

```json
{
  "success": true,
  "data": {
    "model_name": "mobilenetv3_small_075",
    "device_type": "jetson_nano_2gb",
    "predicted_energy_mwh": 28.4,
    "confidence_interval_lower": 24.1,
    "confidence_interval_upper": 32.7,
    "confidence_percent": 95,
    "category": "EXCELLENT",
    "model_accuracy_mape": 18.69,
    "recommendation": "✅ EXCELLENT - Safe to deploy"
  }
}
```

**Ví dụ 2: Triển khai model**

```bash
curl -X POST http://localhost:5000/api/deploy \
  -H "Content-Type: application/json" \
  -d '{
    "bbb_ip": "192.168.1.100",
    "model_name": "mobilenetv3_small_075",
    "max_energy": 50.0
  }'
```

**Response:**

```json
{
  "success": true,
  "message": "Deployment initiated",
  "deployment_id": "deploy_20260118_103045",
  "device": {
    "ip": "192.168.1.100",
    "status": "downloading"
  },
  "model": {
    "name": "mobilenetv3_small_075",
    "predicted_energy_mwh": 28.4,
    "category": "EXCELLENT"
  },
  "timestamp": "2026-01-18T10:30:45.123Z"
}
```

---

## IV. Công Nghệ Thực Hiện

### 4.1 ML Controller Stack

| Thành Phần | Công Nghệ | Phiên Bản | Mục Đích |
|-----------|-----------|---------|---------|
| **Web Framework** | Flask | 3.0.0 | RESTful API + Dashboard |
| **Machine Learning** | scikit-learn | Latest | Gradient Boosting model |
| **Data Processing** | pandas, numpy | Latest | CSV processing, feature eng. |
| **Model Download** | PyTorch, timm | Latest | Download pre-trained models |
| **HTTP Client** | requests | 2.32.3 | Gọi API device |
| **Logging** | Built-in JSON | - | Deployment logs |

### 4.2 ML Agent Stack

| Thành Phần | Công Nghệ | Phiên Bản | Mục Đích |
|-----------|-----------|---------|---------|
| **Web Framework** | Flask | 3.0.0 | API endpoint |
| **Inference** | TFLite Runtime | 2.14.0 | Run model (.tflite) |
| **Inference** | ONNX Runtime | Latest | Run model (.onnx) - Jetson |
| **System Monitor** | psutil | Latest | CPU, RAM, Temp metrics |
| **HTTP Client** | requests | 2.32.3 | Gọi controller API |
| **Container** | Docker | Latest | Package environment |

### 4.3 Deployment Stack

| Thành Phần | Công Nghệ | Mục Đích |
|-----------|-----------|---------|
| **Containerization** | Docker | Đóng gói môi trường nhất quán |
| **Container Registry** | Docker Hub | Lưu trữ image |
| **Fleet Management** | Balena Cloud | Quản lý hàng trăm device |
| **Networking** | HTTP REST | Giao tiếp giữa controller-agent |
| **Reverse Proxy** | nginx | Load balancing (nếu cần) |

---

## V. Sơ Đồ Tương Tác Thành Phần

### 5.1 Request-Response Flow

```
┌─────────────┐
│   Browser   │
│  (User UI)  │
└──────┬──────┘
       │ HTTP GET/POST
       ▼
┌──────────────────────────────────────┐
│     ML CONTROLLER (Flask Server)     │
├──────────────────────────────────────┤
│ 1. Route request to handler          │
│ 2. Call service layer (Predictor)    │
│ 3. Load model from pickle            │
│ 4. Process features                  │
│ 5. Generate prediction + CI          │
│ 6. Return JSON response              │
└──────┬───────────────────────────────┘
       │ HTTP POST
       ▼
┌──────────────────────────────────────┐
│    ML AGENT (Device - Flask)         │
├──────────────────────────────────────┤
│ 1. Receive /deploy request           │
│ 2. Download model from controller    │
│ 3. Load model into TFLite/ONNX       │
│ 4. Update state to "ready"           │
│ 5. Return 200 OK                     │
└──────────────────────────────────────┘
```

### 5.2 Data Flow - Dự Đoán Năng Lượng

```
User Input:
- Model: mobilenetv3_small_075
- Device: Jetson Nano
- Features: params_m, gflops, ...
    ↓
[Energy Predictor Service]
    ├─ Load: jetson_energy_model.pkl
    ├─ Load: jetson_scaler.pkl
    ├─ Load: device_specific_features.json
    │
    ├─ Feature Engineering:
    │   └─ Compute: gflops_per_param, gmacs_per_mb, ...
    │
    ├─ Scaling: StandardScaler.transform()
    │
    ├─ Prediction: GradientBoosting.predict()
    │   └─ energy_pred = 28.4 mWh
    │
    ├─ Confidence Interval:
    │   └─ CI = energy_pred ± (energy_pred × MAPE × 1.96)
    │   └─ CI = [24.1, 32.7] mWh
    │
    ├─ Classification:
    │   └─ 28.4 ≤ 34.6 (P25) → "EXCELLENT"
    │
    └─ Return:
        {
          "predicted_energy_mwh": 28.4,
          "confidence_interval": [24.1, 32.7],
          "category": "EXCELLENT"
        }
```

---

## VI. Tóm Tắt Chương 2

✅ **Kiến Trúc**: Client-Server + Containerization
- ML Controller (Server): Quản lý + Web UI + API
- ML Agents (Devices): Docker containers chạy trên Jetson/RPi/BBB

✅ **4 Phân Hệ Chính:**
1. **Data Collection**: 274 models benchmark → CSV dataset
2. **Energy Prediction**: Device-aware Gradient Boosting → mWh forecast
3. **Deployment Manager**: Automate model transfer & deployment
4. **Monitoring**: Real-time telemetry collection & visualization

✅ **20+ API Endpoints** hỗ trợ:
- Model management
- Energy prediction
- Deployment control
- Device monitoring
- Logs & analytics
- Balena fleet management

✅ **Data Storage**: File-based JSON (lightweight, persistent)

✅ **Technology**: Flask + scikit-learn + TFLite/ONNX + Docker + Balena

---

**→ Tiếp theo: Chương 3 sẽ trình bày chi tiết quy trình triển khai và training model**
