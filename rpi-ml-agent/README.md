# RPI ML Agent - Raspberry Pi ML Agent Service

## Tổng quan
Service ML Agent chạy trên Raspberry Pi để nhận và triển khai models từ ML Controller.

## Kiến trúc
- **Base Image**: `balenalib/raspberry-pi-python:3.10-bullseye-run`
- **Port**: 8000
- **Runtime**: Python 3.10
- **Frameworks**: TensorFlow Lite, ONNX Runtime

## Quick Start

### 1. Deploy lên Balena Cloud (Khuyến nghị)
```powershell
cd rpi-ml-agent
.\deploy.ps1
```

Hoặc dùng CLI trực tiếp:
```bash
cd rpi-ml-agent
balena push Raspberry_Pi
```

### 2. Deploy thủ công
```bash
# Copy code lên Raspberry Pi
scp -r rpi-ml-agent/ pi@<device-ip>:/home/pi/

# SSH vào device
ssh pi@<device-ip>

# Khởi động service
cd /home/pi/rpi-ml-agent
docker-compose up -d

# Kiểm tra logs
docker-compose logs -f ml-agent
```

### 3. Development (chạy local)
```bash
cd rpi-ml-agent
pip install -r requirements.txt
python app/server.py
```

## API Endpoints

### Health Check
```bash
curl http://<device-ip>:8000/status
```

Response:
```json
{
  "status": "ready",
  "model_name": "mobilenetv3_small_100",
  "inference_active": false,
  "uptime_seconds": 3600,
  "energy_metrics": {
    "budget_mwh": 80.0,
    "latest_mwh": 5.2,
    "status": "ok"
  }
}
```

### Deploy Model
```bash
curl -X POST http://<device-ip>:8000/deploy \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "mobilenetv3_small_100",
    "model_url": "http://192.168.137.1:5000/models/mobilenetv3_small_100.tflite",
    "energy_budget_mwh": 80.0
  }'
```

### Start/Stop Inference
```bash
curl -X POST http://<device-ip>:8000/start
curl -X POST http://<device-ip>:8000/stop
```

### Telemetry
```bash
curl http://<device-ip>:8000/telemetry
```

Response:
```json
{
  "cpu_usage_percent": 25.5,
  "memory_mb": 512,
  "temperature_c": 52.3,
  "power_mw": 2500.0
}
```

## Supported Model Formats
- ✅ **TensorFlow Lite** (.tflite) - Khuyến nghị
- ✅ **ONNX** (.onnx)
- ❌ PyTorch (.pth, .pt) - Cần convert trước

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_DIR_OVERRIDE` | `/data/models` | Model storage directory |
| `PYTHONUNBUFFERED` | `1` | Disable Python output buffering |

## Volume Mounts
- `/data` - Persistent model storage
- `/sys:ro` - Read-only access to system info (for telemetry)

## Balena Features
Service yêu cầu các features:
- `balena-api` - Access Balena API
- `supervisor-api` - Access Supervisor API
- `sysfs` - Access system filesystem

## Troubleshooting

### Service không khởi động
```bash
# Check logs
balena logs <device-name>

# Or with docker-compose
docker-compose logs ml-agent
```

### Port 8000 đã được sử dụng
```bash
# Check process using port
sudo netstat -tuln | grep 8000

# Kill process
sudo kill $(sudo lsof -t -i:8000)
```

### Model download thất bại
```bash
# Test network từ device
ping 192.168.137.1  # ML Controller IP

# Test model URL
curl -I http://192.168.137.1:5000/models/mobilenetv3_small_100.tflite
```

### Memory issues
```bash
# Check memory
free -h

# Restart service
docker-compose restart ml-agent
```

## File Structure
```
/data/
  models/
    current_model.tflite    # Currently deployed model
    agent_state.json        # Service state
```

## Security
- Service chạy với `privileged: true` để access hardware sensors
- Chỉ accept HTTPS connections trong production
- Validate model checksums trước khi deploy

## Monitoring
```bash
# CPU/Memory usage
docker stats ml-agent

# Logs
docker logs -f ml-agent

# Balena dashboard
# https://dashboard.balena-cloud.com
```

## Next Steps
1. Deploy service: `.\deploy.ps1`
2. Verify status: `curl http://<device-ip>:8000/status`
3. Test deployment từ ML Controller UI
4. Monitor telemetry và energy consumption

## Differences from BBB Agent
- Raspberry Pi có nhiều RAM hơn (512MB - 8GB)
- CPU mạnh hơn (4 cores @ 1.8GHz cho RPi 5)
- Support cả WiFi và Ethernet
- Có hardware video encoder/decoder
- Nhiệt độ threshold cao hơn (80°C vs 75°C)

## Changelog
- **2026-01-04**: Initial release với TFLite và ONNX support
- Port 8000 để tránh conflict với services khác
- Auto-start inference sau deployment
- Energy budget validation
