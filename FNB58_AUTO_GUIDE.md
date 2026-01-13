# FNB58 Auto Measurement - Hướng Dẫn Hoàn Chỉnh

## ✨ Tính Năng

Script `fnb58_auto.py` **tự động:**
1. ✅ **Phát hiện cổng USB** kết nối với FNB58 (quét tất cả serial ports)
2. ✅ **Cấp quyền truy cập** cổng serial (chmod + dialout group trên Linux)
3. ✅ **Trigger agent** để đo năng lượng qua FNB58 (endpoint `/measure_energy_fnb58`)
4. ✅ **Post kết quả** về server controller để so sánh với dự đoán
5. ✅ **Xem bảng kết quả** so sánh (thực tế vs dự đoán)

**Hỗ trợ 3 chế độ:**
- **Agent Mode** (mặc định): Trigger agent -> agent tự đo & post
- **Local Mode** (`--local-measure`): Chỉ đo FNB58 cục bộ trên controller
- **Server Mode**: Post kết quả cục bộ lên server để so sánh dự đoán

---

## 📋 Chuẩn Bị

### 1. Phần Cứng
- **FNB58 USB Tester** kết nối qua cáp USB
- **Thiết bị cần đo** (Jetson Nano / RPi5 / BBB) kết nối qua USB sang FNB58

```
Controller (với Python script)
    └─ USB tới FNB58
            └─ FNB58 đo
                └─ Thiết bị (Jetson/RPi5/BBB)
```

### 2. Phần Mềm

#### Trên Controller (`ml-controller/python/`)
```bash
# Cài đặt dependencies
pip install pyserial requests

# Hoặc cập nhật requirements.txt
echo "pyserial>=3.5" >> requirements.txt
pip install -r requirements.txt
```

#### Trên Agents (Jetson / RPi5 / BBB)

Copy file `fnb58_reader.py` vào agent:
```bash
# Ví dụ cho Jetson agent
cp fnb58_reader.py jetson-ml-agent/app/

# Cập nhật requirements
echo "pyserial>=3.5" >> jetson-ml-agent/requirements.txt

# Nếu dùng Docker, cập nhật Dockerfile
# COPY fnb58_reader.py /app/
```

---

## 🚀 Cách Chạy

### Cách 1: Python Script (Mọi OS)

```bash
# Mặc định (30s, phát hiện tự động)
python fnb58_auto.py

# Đo 60 giây
python fnb58_auto.py --duration 60

# Chỉ định cổng (nếu auto-detect không hoạt động)
python fnb58_auto.py --port /dev/ttyUSB0 --duration 60

# Chỉ định IP agent
python fnb58_auto.py --agent-ip 192.168.1.50 --agent-port 8000

# Chỉ đo cục bộ (không trigger agent)
python fnb58_auto.py --local-measure

# Đầy đủ với tên model
python fnb58_auto.py \
    --duration 60 \
    --agent-ip 192.168.1.50 \
    --device-type jetson_nano \
    --model-name resnet50
```

### Cách 2: Bash Wrapper (Linux/macOS)

```bash
# Làm executable
chmod +x fnb58_auto.sh

# Chạy
./fnb58_auto.sh

# Với tùy chọn
./fnb58_auto.sh -d 60 -i 192.168.1.50 -m resnet50 -t jetson_nano

# Xem trợ giúp
./fnb58_auto.sh -h
```

### Cách 3: PowerShell Wrapper (Windows)

```powershell
# fnb58_auto.ps1
python fnb58_auto.py @args

# Dùng:
.\fnb58_auto.ps1 -duration 60 -agent_ip 192.168.1.50
```

---

## 📊 Ví Dụ Thực Tế

### Ví Dụ 1: Đo ResNet50 trên Jetson (30s)

```bash
python fnb58_auto.py \
    --agent-ip 192.168.1.50 \
    --duration 30 \
    --device-type jetson_nano \
    --model-name resnet50
```

**Output:**
```
================================================================================
FNB58 AUTO MEASUREMENT SCRIPT
================================================================================
Cấu hình:
  - Port: auto-detect
  - Agent: 192.168.1.50:8000
  - Server: http://localhost:5000
  - Thời gian đo: 30s
  - Mode: Via Agent
================================================================================

[AUTO] Tìm kiếm cổng FNB58...
[AUTO] Tìm thấy FNB58 trên: /dev/ttyUSB0
[AUTO] Cấp quyền truy cập...
[AUTO] chmod 666 /dev/ttyUSB0 ✓
[AUTO] ✓ Cấp quyền thành công

[AGENT] Gửi request tới agent: http://192.168.1.50:8000/measure_energy_fnb58
[AGENT] ✓ Agent đo xong
  - Cổng: /dev/ttyUSB0
  - Số mẫu: 1247
  - Năng lượng: 234.5 mWh
  - Công suất TB: 280.6 mW

[SERVER] Post kết quả về http://localhost:5000/api/energy/report...
[SERVER] ✓ Post thành công

[RESULT] Lấy 5 bản ghi so sánh gần nhất từ server...

Timestamp                 Model                    Thực (mWh)   Dự đoán      Sai số %
==================================================
2025-01-15 14:30:42       resnet50                 234.5        225.3        4.1%
2025-01-15 14:25:10       efficientnet_b0          156.2        158.7        1.6%
2025-01-15 14:20:05       vit_tiny_patch16_224     89.3         87.5         2.0%
2025-01-15 14:15:30       mobilenetv3_small_100    45.6         46.2         1.3%
2025-01-15 14:10:15       resnet18                 112.4        110.8        1.4%

================================================================================
XONG!
================================================================================
```

### Ví Dụ 2: Chỉ Đo Cục Bộ (Local Mode)

```bash
python fnb58_auto.py --local-measure --duration 15
```

**Dùng khi:**
- Không có agent chạy
- Muốn kiểm tra FNB58 hoạt động không
- Đo năng lượng cho device khác (không phải deployment)

---

## 🔧 Tùy Chọn Dòng Lệnh

### Python Script

```
--port PORT                   Cổng serial (tự động phát hiện nếu không có)
--agent-ip IP                 IP/hostname agent (mặc định: localhost)
--agent-port PORT             Port agent (mặc định: 8000)
--server URL                  Server URL (mặc định: http://localhost:5000)
--duration SECONDS            Thời gian đo (mặc định: 30)
--skip-permission             Bỏ qua cấp quyền
--local-measure               Chỉ đo cục bộ, không trigger agent
--post-server / --no-post     Post về server (mặc định: True)
--device-type TYPE            Loại thiết bị (jetson_nano, rpi5, bbb)
--model-name NAME             Tên model
```

### Bash Wrapper

```
-d DURATION    Thời gian đo (giây)
-p PORT        Cổng serial
-i IP          IP agent
--port PORT    Cổng agent
-s SERVER      URL server
-m MODEL       Tên model
-t DEVICE      Loại thiết bị
-l             Chỉ đo cục bộ
-h             Trợ giúp
```

---

## 🐛 Khắc Phục Sự Cố

### 1. "Không tìm thấy FNB58"
```
[ERROR] Không tìm thấy FNB58
```

**Nguyên nhân & Giải pháp:**

**Windows:**
- Kiểm tra Device Manager → Ports (COM & LPT)
- Nên thấy "USB Serial Device" hoặc "FNB58"
- Chỉ định cổng: `--port COM3`

**Linux:**
```bash
# Liệt kê serial ports
ls -la /dev/ttyUSB*

# Nếu không thấy gì:
# 1. Kiểm tra USB kết nối: lsusb
# 2. Kiểm tra driver: modprobe ch341 (hoặc ftdi_sio)
# 3. Thử cổng khác: --port /dev/ttyUSB1
```

**macOS:**
```bash
# Kiểm tra serial ports
ls -la /dev/tty.usb*

# Cài đặt driver nếu cần (CH340 hoặc PL2303)
```

### 2. "Permission denied" trên Linux

```
[ERROR] Permission denied: /dev/ttyUSB0
```

**Giải pháp:**

Script sẽ tự động cấp quyền, nhưng nếu không hoạt động:

```bash
# Cách 1: chmod (tạm thời)
sudo chmod 666 /dev/ttyUSB0

# Cách 2: Thêm user vào group dialout (vĩnh viễn)
sudo usermod -a -G dialout $USER
# Sau đó logout/login lại

# Kiểm tra:
groups $USER  # Phải có "dialout"
```

### 3. Không kết nối được agent

```
[ERROR] Không kết nối được agent: http://192.168.1.50:8000
```

**Kiểm tra:**
```bash
# 1. Agent có chạy không?
ssh user@192.168.1.50 "curl http://localhost:8000/api/model/list"

# 2. Firewall?
curl -v http://192.168.1.50:8000/api/model/list

# 3. Sai IP/port?
# Kiểm tra IP: ping 192.168.1.50
```

### 4. Agent không nhận request

**Server logs (agent):**
```bash
# SSH vào agent
ssh user@192.168.1.50

# Xem logs (nếu chạy trong Docker)
docker logs <container_id>

# Kiểm tra endpoint có không
curl -X POST http://localhost:8000/measure_energy_fnb58 \
  -H "Content-Type: application/json" \
  -d '{"duration_s": 5}'
```

---

## 📈 Quy Trình Hoàn Chỉnh

### Setup Ban Đầu (Lần 1)

```bash
# 1. Copy fnb58_reader.py vào tất cả agents
for agent in jetson-ml-agent rpi-ml-agent bbb-ml-agent; do
    cp fnb58_reader.py $agent/app/
done

# 2. Cài pyserial
pip install pyserial requests

# 3. Bắt đầu controller
cd ml-controller
python python/app.py &

# 4. SSH vào agent, start agent server
# Sau 2-3 phút
```

### Lần Chạy Sau

```bash
# 1. Đảm bảo controller & agent đang chạy
ps aux | grep "python.*app.py"

# 2. Chạy script
python fnb58_auto.py \
    --agent-ip 192.168.1.50 \
    --duration 30 \
    --model-name resnet50

# 3. Kiểm tra kết quả trên web dashboard
open http://localhost:5000
```

### Kịch Bản Kiểm Chứng (Validation)

**Đo và so sánh 5 models:**

```bash
for model in resnet18 resnet50 mobilenetv3_small_100 efficientnet_b0 vit_tiny_patch16_224; do
    echo "⏱️  Đang đo $model..."
    python fnb58_auto.py \
        --duration 30 \
        --model-name $model \
        --agent-ip 192.168.1.50
    
    # Đợi 10s giữa các lần
    sleep 10
done

echo "✓ Hoàn tất, xem kết quả tại http://localhost:5000"
```

---

## 📝 Định Dạng Kết Quả

### Energy Report (JSON)

Server lưu kết quả trong `data/energy_reports.json`:

```json
{
    "timestamp": "2025-01-15T14:30:42",
    "device_type": "jetson_nano",
    "device_uuid": "xyz123",
    "model_name": "resnet50",
    "actual_energy_mwh": 234.5,
    "predicted_mwh": 225.3,
    "abs_error_mwh": 9.2,
    "pct_error": 4.08,
    "sensor_type": "fnb58",
    "sample_count": 1247,
    "avg_power_mw": 280.6
}
```

### API Response

```json
GET /api/energy/recent?n=5

{
    "success": true,
    "total_items": 45,
    "items": [
        {
            "timestamp": "2025-01-15T14:30:42",
            "device_type": "jetson_nano",
            "model_name": "resnet50",
            "actual_energy_mwh": 234.5,
            "predicted_mwh": 225.3,
            "pct_error": 4.08,
            "sensor_type": "fnb58"
        },
        ...
    ]
}
```

---

## 💡 Mẹo & Thực Hành Tốt

### 1. Cài Đặt Alias Bash

```bash
# ~/.bashrc hoặc ~/.zshrc
alias fnb58='python ~/path/to/fnb58_auto.py'

# Sử dụng:
fnb58 --duration 60 --model-name resnet50
```

### 2. Cron Job (Đo Định Kỳ)

```bash
# Crontab: Đo mỗi giờ
0 * * * * cd /path/to/ml-controller/python && python fnb58_auto.py --duration 120 >> fnb58_measurements.log 2>&1
```

### 3. Validation Loop

```python
# validate_models.py
import subprocess
import time

models = [
    "resnet18", "resnet50", "mobilenetv3_small_100",
    "efficientnet_b0", "vit_tiny_patch16_224"
]

for model in models:
    print(f"📊 Validating {model}...")
    subprocess.run([
        "python", "fnb58_auto.py",
        "--duration", "60",
        "--model-name", model,
        "--agent-ip", "192.168.1.50"
    ])
    time.sleep(10)  # Đợi giữa các lần
```

### 4. Bảo Vệ Port FNB58

```bash
# Nếu nhiều scripts cùng truy cập FNB58, dùng lock file
# Thêm vào fnb58_auto.py:
import fcntl
with open("/tmp/fnb58.lock", "w") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)  # Chờ lock
    # Chạy measurement
```

---

## 🎯 Tóm Tắt

| Tính Năng | Đạt |
|-----------|-----|
| Auto-detect FNB58 port | ✅ |
| Auto-grant permissions | ✅ |
| Trigger agent measurement | ✅ |
| Compare vs prediction | ✅ |
| View results | ✅ |
| Support 3 modes (agent/local/server) | ✅ |
| Error handling & recovery | ✅ |
| Cross-platform (Win/Linux/macOS) | ✅ |
| Bash wrapper (Linux/macOS) | ✅ |
| Documentation & examples | ✅ |

**Tất cả tính năng đã sẵn sàng sử dụng! 🚀**
