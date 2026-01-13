# Hướng Dẫn Đo Năng Lượng Bằng FNB58 USB Tester

## 📋 Tổng Quan

Tích hợp FNB58 (hoặc FNB48 với giao thức tương tự) để đo năng lượng thực tế trong khi chạy model inference trên thiết bị IoT. Dữ liệu tự động được post về server để so sánh với dự đoán.

## 🔌 Yêu Cầu Phần Cứng

- **FNB58 USB Tester** (hoặc FNB48) - kết nối qua USB
- **Thiết bị IoT** (Jetson Nano, Raspberry Pi 5, hoặc BBB)
- **Cổng USB của thiết bị** để kết nối FNB58
- **Cáp sạc/nguồn** được kết nối qua FNB58 để đo

## 📦 Cài Đặt Phần Mềm

### Trên Server (ML Controller)

FNB58 reader đã có sẵn tại [ml-controller/python/fnb58_reader.py](ml-controller/python/fnb58_reader.py).

Cài pyserial nếu chưa có:
```bash
pip install pyserial
```

### Trên Agent (Jetson/RPi/BBB)

**Lựa chọn 1: Sao chép fnb58_reader.py vào thư mục app của agent**

```bash
# Trên máy host/build server
cp ml-controller/python/fnb58_reader.py jetson-ml-agent/app/
cp ml-controller/python/fnb58_reader.py rpi-ml-agent/app/
cp ml-controller/python/fnb58_reader.py bbb-ml-agent/app/
```

**Lựa chọn 2: Cập nhật requirements.txt của agent**

Thêm `pyserial` vào từng agent:
```bash
# jetson-ml-agent/requirements.txt
# rpi-ml-agent/requirements.txt
# bbb-ml-agent/requirements.txt
```

Thêm dòng này:
```
pyserial>=3.5
```

**Lựa chọn 3: Sử dụng Balena (Recommend)**

Trong Dockerfile của agent, thêm:
```dockerfile
RUN pip install pyserial
COPY fnb58_reader.py /app/fnb58_reader.py
```

## 🔍 Xác Định Cổng USB Của FNB58

### Trên Linux (Jetson Nano / RPi5)

```bash
# Liệt kê tất cả cổng USB tester
ls -la /dev/ttyUSB*

# Hoặc dùng lsusb
lsusb

# Hoặc dùng dmesg để xem log khi cắm FNB58
dmesg | tail -20
```

Thường FNB58 hiện ra là `/dev/ttyUSB0` hoặc `/dev/ttyUSB1`.

### Trên Windows

```powershell
# Sử dụng Device Manager hoặc command:
Get-PnpDevice | Where-Object { $_.Name -like "*USB*" } | Select-Object Name, ConfigManagerErrorCode
```

FNB58 thường là `COM3`, `COM4`, v.v.

### Tự Động Phát Hiện (Recommend)

Agent có hàm `detect_fnb58_port()` trong `fnb58_reader.py` để tự động tìm FNB58:

```python
from fnb58_reader import detect_fnb58_port
port = detect_fnb58_port()  # Trả về "/dev/ttyUSB0" nếu tìm thấy
```

## 🚀 Cách Sử Dụng

### Cách 1: Gọi Endpoint FNB58 Trực Tiếp Trên Agent

Kết nối FNB58, triển khai model lên agent, rồi gọi:

```bash
# Linux/Mac
curl -X POST http://<AGENT_IP>:8000/measure_energy_fnb58 \
  -H "Content-Type: application/json" \
  -d '{
    "duration_s": 30,
    "auto_detect": true,
    "controller_url": "http://<SERVER_IP>:5000"
  }'
```

```powershell
# PowerShell Windows
Invoke-RestMethod -Uri "http://<AGENT_IP>:8000/measure_energy_fnb58" `
  -Method POST -ContentType "application/json" `
  -Body '{
    "duration_s": 30,
    "auto_detect": true,
    "controller_url": "http://<SERVER_IP>:5000"
  }'
```

**Tham số:**
- `duration_s`: Thời gian đo (giây), mặc định 30
- `auto_detect`: true = tự tìm cổng FNB58 (khuyến nghị), false = chỉ định port
- `fnb58_port`: Cổng nếu `auto_detect: false`, ví dụ "/dev/ttyUSB0" (Linux) hoặc "COM3" (Windows)
- `controller_url`: URL server để auto-post kết quả. Tự động từ biến môi trường `CONTROLLER_URL` nếu không chỉ định

**Kết quả:**
```json
{
  "success": true,
  "sensor_type": "fnb58",
  "port": "/dev/ttyUSB0",
  "duration_s": 30.2,
  "samples_count": 45,
  "actual_energy_mwh": 17500.5,
  "avg_power_mw": 968.2,
  "posted_to_controller": true,
  "last_values": {
    "voltage_v": 5.1,
    "current_a": 2.85,
    "power_w": 14.54,
    "energy_wh": 17.5
  }
}
```

### Cách 2: Proxy Qua Server

Server có endpoint `/api/device/measure-energy` để gọi agent:

```powershell
# Gọi từ server để proxy tới agent
Invoke-RestMethod -Uri "http://localhost:5000/api/device/measure-energy" `
  -Method POST -ContentType "application/json" `
  -Body '{
    "device_url": "http://<AGENT_IP>:8000",
    "duration_s": 30
  }'
```

**Lưu ý:** Cách này cũng trigger FNB58, nhưng controller URL được server tự gắn vào.

### Cách 3: Thủ Công - Đọc FNB58 và Post Kết Quả

```python
# Trên máy host có FNB58 kết nối
from fnb58_reader import FNB58Reader
import requests

reader = FNB58Reader("/dev/ttyUSB0")  # Hoặc "COM3" trên Windows
reader.start()
time.sleep(60)  # Đo 60 giây trong khi model chạy
result = reader.stop()

# Post lên server
payload = {
    "device_type": "jetson_nano",  # hoặc "raspberry_pi5", "bbb"
    "model_name": "mobilenetv3_small_075",
    "actual_energy_mwh": result["total_energy_mwh"],
    "avg_power_mw": result["avg_power_mw"],
    "duration_s": 60,
    "sensor_type": "fnb58"
}
requests.post("http://localhost:5000/api/energy/report", json=payload)
```

## 📊 Xem Kết Quả So Sánh

### Trên Server

```powershell
# Lấy 10 bản ghi gần nhất
Invoke-RestMethod -Uri "http://localhost:5000/api/energy/recent?n=10" -Method GET | ConvertTo-Json -Depth 10
```

**Kết quả trả về:**
```json
{
  "success": true,
  "items": [
    {
      "timestamp": "2026-01-13T12:30:45.123456Z",
      "device_type": "jetson_nano",
      "model_name": "mobilenetv3_small_075",
      "sensor_type": "fnb58",
      "duration_s": 30.2,
      "actual_energy_mwh": 17500.5,
      "predicted_mwh": 17.5,
      "abs_error_mwh": 0.3,
      "pct_error": 1.7,
      "ci_lower_mwh": 14.3,
      "ci_upper_mwh": 20.9
    }
  ],
  "total": 15
}
```

### Giải Thích Trường Dữ Liệu

- `actual_energy_mwh`: Năng lượng đo từ FNB58 (mWh)
- `predicted_mwh`: Năng lượng dự đoán của model (mWh)
- `abs_error_mwh`: Sai số tuyệt đối (mWh)
- `pct_error`: Sai số phần trăm (%)
- `ci_lower_mwh`, `ci_upper_mwh`: Dải tin cậy 95% của dự đoán
- `sensor_type`: Loại cảm biến ("fnb58" cho USB tester)

## ⚙️ Cấu Hình Môi Trường (Balena)

Nếu deploy qua Balena, thêm biến môi trường ở fleet/device:

```
CONTROLLER_URL=http://<SERVER_IP>:5000
```

Khi đó agent tự động post về server mà không cần chỉ định controller_url ở request.

## 🐛 Troubleshooting

### Lỗi: "FNB58 reader không khả dụng"

**Nguyên nhân:** Chưa cài pyserial hoặc fnb58_reader.py không ở đúng thư mục.

**Giải pháp:**
```bash
pip install pyserial
cp fnb58_reader.py <app_folder>/
```

### Lỗi: "FNB58 port not found"

**Nguyên nhân:** FNB58 chưa được kết nối hoặc không được phát hiện.

**Giải pháp:**
1. Kiểm tra FNB58 có kết nối qua USB không:
   ```bash
   lsusb | grep -i "USB Tester\|FNB"
   ```
2. Chỉ định cổng thủ công:
   ```bash
   curl -X POST http://<AGENT_IP>:8000/measure_energy_fnb58 \
     -d '{"fnb58_port": "/dev/ttyUSB0", "duration_s": 30}'
   ```

### Lỗi: "Không kết nối được /dev/ttyUSB0"

**Nguyên nhân:** Quyền truy cập cổng serial bị từ chối.

**Giải pháp (Linux):**
```bash
# Cấp quyền cho user
sudo usermod -a -G dialout $USER

# Hoặc cấp quyền cho cổng
sudo chmod 666 /dev/ttyUSB0
```

**Trên Balena (container):**
```dockerfile
# Dockerfile
RUN usermod -a -G dialout root
```

### Lỗi: "Failed to post to controller"

**Nguyên nhân:** Server không kết nối được hoặc URL sai.

**Giải pháp:**
- Kiểm tra server đang chạy: `http://localhost:5000` (hoặc IP server đúng)
- Kiểm tra kết nối mạng giữa agent và server
- Xem log của request:
  ```python
  # Trong response, trường "post_warning" sẽ chứa chi tiết lỗi
  ```

## 📖 Ví Dụ Thực Tế: Đo Năng Lượng Model Jetson

### Bước 1: Chuẩn Bị

```bash
# Trên Jetson Nano
ssh jetson@192.168.1.50

# Kiểm tra FNB58
ls /dev/ttyUSB*
# Output: /dev/ttyUSB0

# Deploy model (hoặc từ dashboard)
curl -X POST http://localhost:8000/deploy \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "mobilenetv3_small_075",
    "model_url": "http://192.168.1.100:5000/models/mobilenetv3_small_075.tflite",
    "energy_budget_mwh": 50
  }'
```

### Bước 2: Đo Năng Lượng 60 Giây

```bash
# Gọi endpoint FNB58 từ server hoặc Jetson
curl -X POST http://192.168.1.50:8000/measure_energy_fnb58 \
  -H "Content-Type: application/json" \
  -d '{
    "duration_s": 60,
    "auto_detect": true,
    "controller_url": "http://192.168.1.100:5000"
  }'
```

### Bước 3: Xem Kết Quả So Sánh

```powershell
# Từ máy Windows/Linux có server
Invoke-RestMethod -Uri "http://192.168.1.100:5000/api/energy/recent?n=5" -Method GET
```

**Output:**
```
timestamp              : 2026-01-13T12:45:30.123Z
device_type           : jetson_nano
model_name            : mobilenetv3_small_075
actual_energy_mwh     : 15.2
predicted_mwh         : 17.5
abs_error_mwh         : 2.3
pct_error             : 13.1
sensor_type           : fnb58
ci_lower_mwh          : 14.3
ci_upper_mwh          : 20.9
```

**Kết luận:** Dự đoán cao hơn thực tế 13%, nằm trong dải tin cậy → Mô hình tốt.

## 🎯 Các Bước Tiếp Theo

1. **Benchmark multiple models:** Chạy FNB58 cho vài model khác nhau để xây dựng tập dữ liệu validation
2. **Cải thiện mô hình:** Nếu MAPE cao, thu thập thêm data từ đo thực tế, retrain model
3. **Tối ưu hóa deployment:** Dùng kết quả để chọn model tốt nhất (năng lượng thấp, accuracy cao)
4. **Tích hợp CI/CD:** Tự động test model trước khi deploy bằng FNB58

---

**Hỗ Trợ Thêm**

Nếu bạn muốn mở rộng cho loại cảm biến khác (INA219, tegrastats, v.v.), hãy:
1. Thêm reader module tương tự `fnb58_reader.py`
2. Thêm endpoint `/measure_energy_<sensor_type>` vào agent
3. Đặt `sensor_type: "<sensor_type>"` khi post về server

Tất cả endpoint đều sử dụng chung endpoint `/api/energy/report` của server, nên so sánh dễ dàng.
