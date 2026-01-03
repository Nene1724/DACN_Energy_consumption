# Hướng Dẫn Sử Dụng IoT ML Energy Manager

Hệ thống triển khai mô hình ML tới thiết bị IoT với giám sát năng lượng tiêu thụ.

---

## Khởi Động Hệ Thống

### 1. Cài đặt và chạy server

```bash
# Di chuyển vào thư mục python
cd d:\DACN\ml-controller\python

# Chạy Flask server
python app.py
```

Server sẽ khởi động tại: **http://localhost:5000**

### 2. Mở trình duyệt web

Truy cập: **http://localhost:5000**

Giao diện **IoT ML Energy Manager** sẽ hiển thị.

---

## Cấu Trúc Giao Diện

Giao diện có 2 chế độ xem chính:

### Tab 1: Deployment (Triển khai)
- Quản lý models
- Deploy models lên thiết bị
- Giám sát năng lượng
- Xem deployment logs

### Tab 2: Monitoring (Giám sát)
- Biểu đồ time-series
- Health status
- Phân bố năng lượng
- Advanced analytics

---

## Hướng Dẫn Sử Dụng Chi Tiết

### BƯỚC 1: Tải Danh Sách Models

**Có 2 cách tải models:**

#### Cách 1: Tải models đề xuất (Recommended)
```
1. Nhấn nút "Đề xuất năng lượng thấp" ở góc trên phải
2. Hệ thống sẽ tải Top 10 models tiết kiệm năng lượng nhất
3. Các models được sắp xếp theo năng lượng tăng dần
```

**Ưu điểm:**
- Chỉ hiển thị models phù hợp với thiết bị IoT
- Đã lọc theo tiêu chí: < 100 mWh, < 100 MB, latency < 0.5s
- Có label "Recommended" và lý do đề xuất

#### Cách 2: Tải tất cả models
```
1. Nhấn nút "Tải tất cả" ở góc trên phải
2. Hệ thống tải toàn bộ 126 models từ benchmark
3. Có thể lọc bằng các filter tabs
```

### BƯỚC 2: Lọc Models (Tùy chọn)

Sử dụng các filter tabs bên dưới tiêu đề trang:

- **Tất cả**: Hiển thị toàn bộ models đã tải
- **Recommended**: Chỉ hiển thị models được đề xuất
- **< 50 mWh**: Models cực kỳ tiết kiệm năng lượng
- **< 30 MB**: Models nhẹ, phù hợp RAM hạn chế

### BƯỚC 3: Chọn Model

```
1. Cuộn danh sách "Model library" ở cột bên phải
2. Click vào model bạn muốn deploy
3. Model được chọn sẽ có viền màu xanh
4. Thông tin chi tiết hiển thị ở "Selection preview" bên dưới
```

**Thông tin hiển thị:**
- Tên model
- Năng lượng tiêu thụ (mWh)
- Kích thước file (MB)
- Độ trễ inference (s)
- Số lượng parameters (M)
- Lý do recommend (nếu có)

**Cảnh báo năng lượng:**
- Nếu model vượt ngưỡng 100 mWh → Hiển thị "Vượt ngưỡng!" màu đỏ
- Bạn vẫn có thể deploy nhưng cần xác nhận

### BƯỚC 4: Cấu Hình Thiết Bị

**Panel "Thiết bị & Constraints" (cột trái):**

```
1. Nhập IP thiết bị IoT vào ô "Địa chỉ IP thiết bị"
   - Mặc định: 192.168.137.10
   - Có thể thay đổi theo IP thực tế của bạn

2. Xem trạng thái thiết bị:
   - Thiết bị IoT: Màu dot cho biết trạng thái (IDLE/READY/RUNNING)
   - Model hiện tại: Model đang chạy trên thiết bị
```

**3 Cách kết nối thiết bị:**

#### Cách 1: Nhập IP thủ công
- Nhập IP vào ô input
- Hệ thống tự động kiểm tra kết nối mỗi 5 giây

#### Cách 2: Sử dụng Balena Fleet (Nâng cao)
```
1. Scroll xuống panel "Balena fleet"
2. Nhấn "Làm mới" để tải danh sách thiết bị từ Balena Cloud
3. Chọn thiết bị ONLINE
4. Nhấn nút "Connect" bên cạnh thiết bị
5. IP sẽ tự động điền vào ô input
```

#### Cách 3: Local network discovery (Tùy chọn)
- Sử dụng công cụ như `arp -a` để tìm IP thiết bị
- Nhập IP vào ô input thủ công

### BƯỚC 5: Deploy Model

```
1. Đảm bảo đã chọn model (step 3)
2. Đảm bảo đã nhập IP thiết bị (step 4)
3. Nhấn nút "Deploy model đã chọn" (màu xanh) ở góc trên phải panel "Model library"
```

**Quy trình deploy:**

1. **Kiểm tra năng lượng:**
   - Nếu model vượt ngưỡng 100 mWh → Hiện popup xác nhận
   - Chọn "OK" để tiếp tục hoặc "Cancel" để hủy

2. **Download model (tự động):**
   - Hệ thống check xem model artifact đã có trong `model_store/` chưa
   - Nếu chưa có → Tự động download từ timm
   - Hiển thị thông báo "Đang tiến hành deploy model..."

3. **Transfer tới thiết bị:**
   - Controller gửi model file tới thiết bị IoT
   - Thiết bị download model qua endpoint `/models/<filename>`
   - Hiển thị progress trong "Deployment log"

4. **Kết quả:**
   - Thành công: Thông báo màu xanh "Deploy thành công!"
   - Thất bại: Thông báo màu đỏ với lý do lỗi

### BƯỚC 6: Giám Sát Năng Lượng

**Panel "Energy watch" (cột trái, giữa):**

Sau khi deploy thành công, panel này tự động cập nhật:

**4 Metrics chính:**
1. **Ngân sách (mWh)**: Ngưỡng năng lượng đã đặt (100 mWh)
2. **Đo mới nhất (mWh)**: Lần đo năng lượng gần nhất
3. **Trung bình (mWh)**: Năng lượng trung bình của các lần đo
4. **Trạng thái**: 
   - "Ổn định" (màu xanh): Trong ngưỡng
   - "Vượt ngưỡng" (màu đỏ): Vượt quá ngân sách

**Lịch sử đo:**
- Hiển thị 6 lần đo gần nhất
- Mỗi dòng: Thời gian + Năng lượng (mWh)
- Tự động cập nhật real-time

### BƯỚC 7: Xem Deployment Logs

**Panel "Deployment log" (cột trái, dưới cùng):**

Logs hiển thị theo thời gian thực:
- Màu xanh (info): Thông tin deploy
- Màu đỏ (error): Lỗi deploy
- Màu xanh lá (success): Deploy thành công

**Ví dụ logs:**
```
14:23:45 - Bắt đầu deploy mobilenetv3_small_075 lên thiết bị 192.168.137.10
14:23:46 - Ngân sách năng lượng: 100 mWh
14:23:52 - Hoàn tất deploy mobilenetv3_small_075 (58.32 mWh)
14:23:52 - Đang giám sát với ngân sách 100 mWh
```

---

## Tính Năng Nâng Cao

### 1. Energy Predictor (Dự đoán năng lượng)

**Panel "Energy predictor" (cột phải, giữa):**

Dùng để dự đoán năng lượng cho models chưa benchmark.

**Cách sử dụng:**

#### Cách 1: Điền tay
```
1. Nhập các thông số model:
   - Params (M): Số lượng parameters (triệu)
   - GFLOPs: Floating point operations (tỷ)
   - GMACs: Multiply-accumulate operations (tỷ)
   - Size (MB): Kích thước file model
   - Latency (s): Thời gian inference
   - Throughput (iter/s): Số iterations mỗi giây

2. Nhấn nút "Dự đoán"

3. Kết quả hiển thị:
   - Energy Est: Năng lượng dự đoán (mWh)
   - CI: Confidence interval (khoảng tin cậy)
```

#### Cách 2: Dùng model đang chọn
```
1. Chọn model trong danh sách
2. Nhấn "Dùng model đang chọn"
3. Hệ thống tự động điền các thông số có sẵn
4. Bổ sung GMACs nếu thiếu
5. Nhấn "Dự đoán"
```

**Kết quả:**
```
Energy Est: 58.32 mWh
CI: 40.82 - 75.82 mWh
Model: mobilenetv3_small_075
```

### 2. Benchmark Snapshot

**Panel "Benchmark snapshot" (cột phải, dưới cùng):**

Hiển thị thống kê nhanh:
- **Tổng models**: Số models trong dataset
- **Dưới 50 mWh**: Số models cực kỳ tiết kiệm năng lượng
- **Dưới 100 mWh**: Số models phù hợp IoT

### 3. Balena Fleet Management

**Panel "Balena fleet" (cột trái, giữa-dưới):**

Quản lý nhiều thiết bị IoT qua Balena Cloud.

**Chức năng:**
1. **Lọc theo app:** Nhập app slug để lọc thiết bị theo project
2. **Chỉ hiển thị online:** Checkbox để ẩn thiết bị offline
3. **Làm mới:** Reload danh sách thiết bị

**Mỗi thiết bị hiển thị:**
- Tên thiết bị
- Status badge: ONLINE (xanh) / OFFLINE (xám)
- UUID, IP, OS version, Device type
- Nút "Webconsole": Mở web terminal
- Nút "Connect": Dùng IP này để deploy

---

## Chế Độ Monitoring

**Chuyển sang tab "Monitoring" ở đầu trang.**

### 4 Loại Biểu Đồ:

#### 1. Time-series
- **Line/Area chart**: Năng lượng theo thời gian
- **Multi-axis**: Energy vs Latency & Throughput
- Hiển thị xu hướng tiêu thụ năng lượng

#### 2. Status / Health
- **Gauge**: Phần trăm ngân sách đã dùng
- **Uptime**: Tỷ lệ hoạt động
- **KPIs**: Latency avg, Memory used, Thermal, Success rate

#### 3. Phân bố (Distribution)
- **Histogram**: Phân bố tiêu thụ năng lượng
- **Box summary**: Min, P25, P50, P75, Max

#### 4. Nâng cao (Advanced)
- **Pareto/Scatter**: Trade-off năng lượng vs latency
- **Heatmap slot**: Tiêu thụ theo khung giờ

---

## Xử Lý Lỗi Thường Gặp

### Lỗi 1: "Model artifact not found"

**Nguyên nhân:** File model chưa có trong `model_store/`

**Giải pháp:**
```bash
# Tự động: Hệ thống sẽ download khi deploy
# Hoặc download thủ công:
cd d:\DACN\ml-controller\python
python download_models.py <model_name>

# Ví dụ:
python download_models.py mobilenetv3_small_075
```

### Lỗi 2: "Không thể kết nối BBB"

**Nguyên nhân:** IP sai hoặc thiết bị không online

**Giải pháp:**
1. Kiểm tra thiết bị đã bật chưa
2. Ping IP để test kết nối: `ping 192.168.137.10`
3. Kiểm tra thiết bị và máy tính cùng mạng
4. Thử IP khác từ Balena Fleet

### Lỗi 3: "Deploy bị hủy do vượt ngưỡng năng lượng"

**Nguyên nhân:** Model cần năng lượng > 100 mWh

**Giải pháp:**
1. Chọn model khác nhẹ hơn (< 100 mWh)
2. Hoặc chấp nhận deploy: Click "OK" khi popup xác nhận

### Lỗi 4: "BALENA_API_TOKEN chưa được cấu hình"

**Nguyên nhân:** Chưa cấu hình token Balena Cloud

**Giải pháp:**
- Token đã được hardcode trong `app.py` (development mode)
- Hoặc set environment variable:
```bash
export BALENA_API_TOKEN=your_token_here
```

### Lỗi 5: "Không thể tải danh sách models"

**Nguyên nhân:** Server chưa chạy hoặc file CSV bị lỗi

**Giải pháp:**
1. Kiểm tra server đang chạy: http://localhost:5000
2. Check file CSV tồn tại: `ml-controller/data/124_models_benchmark_jetson.csv`
3. Restart server: Ctrl+C rồi `python app.py`

---

## Tips & Best Practices

### 1. Chọn Model Phù Hợp

**Cho thiết bị battery-powered:**
- Chọn models < 50 mWh (ghostnet_100, mnasnet_small)
- Ưu tiên "Recommended" models

**Cho thiết bị có nguồn điện:**
- Có thể chọn models 100-300 mWh
- Cân nhắc trade-off: Năng lượng vs Accuracy

### 2. Quản Lý Năng Lượng

- Set ngân sách phù hợp với thiết bị
- Giám sát "Energy watch" thường xuyên
- Nếu "Vượt ngưỡng" → Rollback về model nhẹ hơn

### 3. Testing Workflow

**Quy trình test model mới:**
```
1. Tải models đề xuất
2. Chọn model nhẹ nhất (ghostnet_100)
3. Deploy lên thiết bị test
4. Giám sát năng lượng 5-10 phút
5. Nếu ổn định → Deploy lên production
```

### 4. Monitoring Workflow

```
1. Deploy model
2. Chuyển sang tab "Monitoring"
3. Xem tab "Time-series" → Check xu hướng
4. Xem tab "Status/Health" → Check metrics
5. Nếu có bất thường → Xem logs → Debug
```

### 5. Backup & Rollback

- Ghi chú model và năng lượng trong logs
- Nếu model mới tốn năng lượng → Deploy lại model cũ
- Sử dụng Balena Fleet để rollback nhiều thiết bị cùng lúc

---

## API Endpoints (Cho Advanced Users)

Nếu muốn tích hợp vào hệ thống khác:

### GET /api/models/all
Lấy danh sách tất cả 126 models
```bash
curl http://localhost:5000/api/models/all
```

### GET /api/models/recommended
Lấy models đề xuất
```bash
curl "http://localhost:5000/api/models/recommended?device_type=BBB&max_energy=100"
```

### POST /api/predict-energy
Dự đoán năng lượng
```bash
curl -X POST http://localhost:5000/api/predict-energy \
  -H "Content-Type: application/json" \
  -d '{
    "params_m": 5.0,
    "gflops": 1.5,
    "gmacs": 0.75,
    "size_mb": 20.0,
    "latency_avg_s": 0.05,
    "throughput_iter_per_s": 20.0
  }'
```

### POST /api/deploy
Deploy model
```bash
curl -X POST http://localhost:5000/api/deploy \
  -H "Content-Type: application/json" \
  -d '{
    "bbb_ip": "192.168.137.10",
    "model_name": "mobilenetv3_small_075",
    "max_energy": 100
  }'
```

---

## Tài Liệu Tham Khảo

- **README.md**: Tổng quan project và API documentation
- **requirements.txt**: Danh sách dependencies
- **124_models_benchmark_jetson.csv**: Dataset benchmark Jetson Nano (124 models)
- **27_models_benchmark_rpi5.csv**: Dataset benchmark Raspberry Pi 5 (27 models)
- **energy_prediction_model.ipynb**: Chi tiết training model

---

## Liên Hệ & Hỗ Trợ

Nếu gặp vấn đề:
1. Check logs trong "Deployment log" panel
2. Check terminal console (nơi chạy `python app.py`)
3. Xem file README.md section "Troubleshooting"

---

**Chúc bạn sử dụng thành công IoT ML Energy Manager!**
