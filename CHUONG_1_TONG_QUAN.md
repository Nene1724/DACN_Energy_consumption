# CHƯƠNG 1: TỔNG QUAN

## I. Đặt Vấn Đề

### 1.1 Xu Hướng Edge AI

Trong những năm gần đây, **Edge Computing** đã trở thành xu hướng chủ yếu trong IoT và AI:

- **Truyền thống**: Dữ liệu → Gửi lên Cloud → Xử lý → Gửi kết quả về
- **Hiện tại**: Dữ liệu → Xử lý ngay tại thiết bị (Edge) → Kết quả

**Lợi ích Edge Computing:**
- ✅ Độ trễ thấp (ms thay vì s)
- ✅ Bảo vệ dữ liệu (xử lý tại chỗ)
- ✅ Hoạt động offline
- ✅ Giảm tải mạng

### 1.2 Thách Thức Chính: Năng Lượng

Mặc dù lợi ích, **thách thức lớn nhất là hạn chế tài nguyên năng lượng**:

**Hiện tượng Real-World:**
- 🔋 IoT devices chạy bằng pin hoặc có công suất giới hạn
- ⚠️ Khi chạy Deep Learning, tiêu thụ năng lượng cao → Pin cạn nhanh
- 🔥 Thermal throttling → Thiết bị giảm hiệu năng hoặc tắt đột ngột
- 💥 Hệ thống sập, gián đoạn dịch vụ

**Vấn Đề Hiện Tại:**

| Vấn Đề | Tác Động | Hậu Quả |
|--------|---------|---------|
| Không biết năng lượng trước deploy | Deploy blind | Device overload/sập |
| Deploy sau, mới đo energy | Post-mortem | Dữ liệu bị mất |
| Không có automation | Manual work | 1-2 giờ per device |
| Không có monitoring | Không biết trạng thái | Không thể can thiệp |

---

## II. Mục Tiêu Đề Tài

### 2.1 Mục Tiêu Tổng Quát

Phát triển **hệ thống tự động quản lý triển khai mô hình ML trên thiết bị IoT edge** với **dự báo và kiểm soát năng lượng**.

**3 Trụ Cột:**
1. 📊 **Dự báo năng lượng trước deploy** - Không bất ngờ
2. 🚀 **Tự động hóa triển khai** - Từ phút xuống giây
3. 🛡️ **Kiểm soát năng lượng runtime** - Auto-stop nếu vượt budget

### 2.2 Mục Tiêu Cụ Thể (SMART)

| # | Mục Tiêu | Kỳ Vọng | Đạt Được |
|---|----------|---------|----------|
| 1 | Benchmark models | 200+ | ✅ 274 |
| 2 | Energy prediction MAPE | < 25% | ✅ 18.69% |
| 3 | Deployment time | < 60s | ✅ 42s |
| 4 | Energy enforcement | Automatic | ✅ Yes |
| 5 | Device support | 3+ types | ✅ Jetson/RPi/BBB |

---

## III. Phạm Vi Nghiên Cứu

### 3.1 Thiết Bị Hỗ Trợ

**3 Dòng Thiết Bị Đại Diện:**

| Thiết Bị | CPU | GPU | RAM | Trường Hợp |
|----------|-----|-----|-----|-----------|
| **Jetson Nano 2GB** | ARM 4-core | CUDA 128-core | 2GB | High-power edge |
| **Raspberry Pi 5** | ARM 4-core 64-bit | None | 4GB | Mid-range edge |
| **BeagleBone Black** | ARM 1-core | None | 512MB | Low-power edge |

### 3.2 Danh Sách Models

- **Jetson**: 247 models thực benchmark
- **RPi5**: 27 models thực benchmark  
- **BBB**: Sẵn sàng hỗ trợ
- **Total**: 274 models

### 3.3 Ranh Giới (Out of Scope)

❌ Không train models từ đầu (chỉ reuse pre-trained)  
❌ Không custom hardware (chỉ devices có sẵn)  
❌ Không optimize models (chỉ predict/deploy existing)  
❌ Không security hardening production (research focus)

---

## IV. Cơ Sở Lý Thuyết

### 4.1 Machine Learning cho Energy Prediction

**Tại Sao ML?**
- Energy consumption = f(model properties, device properties, runtime conditions)
- Hàm này **phi tuyến** (non-linear) → ML phù hợp hơn linear regression

**Thuật Toán Chọn:**
- **Gradient Boosting Regressor** (scikit-learn)
- Lý do:
  - Xử lý non-linear relationships tốt
  - Capture feature interactions
  - Robust với outliers
  - Fast prediction (< 100ms)

**Features (Đặc Trưng):**

| Loại | Features | Số Lượng |
|------|----------|----------|
| Base | params_m, gflops, gmacs, size_mb, latency, throughput | 6 |
| Derived | gflops_per_param, gmacs_per_mb, compute_intensity, ... | 6 |
| **Total** | **-** | **12** |

### 4.2 Evaluation Metrics

**Dự báo năng lượng đánh giá bằng:**

$$\text{MAPE} = \frac{1}{n} \sum_{i=1}^{n} \left| \frac{y_{\text{actual}} - y_{\text{pred}}}{y_{\text{actual}}} \right| \times 100\%$$

$$R^2 = 1 - \frac{\sum (y_{\text{actual}} - y_{\text{pred}})^2}{\sum (y_{\text{actual}} - \bar{y})^2}$$

**Acceptance Criteria:**
- MAPE < 20% ✅
- R² > 0.80 ✅

### 4.3 Confidence Interval

Để user biết "độ tin cậy" của prediction:

$$\text{CI}_{95\%} = \text{Predicted} \pm (\text{MAPE} \times 1.96 \times \text{Predicted})$$

**Ví Dụ:**
- Predicted: 28.4 mWh
- MAPE: 18.69%
- CI: [23.3 - 33.5] mWh (95% confidence)

---

## V. Phương Pháp Luận (Methodology)

### 5.1 Quy Trình Tổng Quát

```
┌─────────────────────────────────────────────────────────┐
│ PHASE 1: Data Collection (Chương 3)                    │
│ └─ Benchmark 274 models trên 3 devices                 │
│ └─ Collect: model properties + measurements            │
│ └─ Output: CSV datasets (247 + 27 rows)                │
├─────────────────────────────────────────────────────────┤
│ PHASE 2: Model Training (Chương 3)                     │
│ └─ Feature engineering (6 → 12)                        │
│ └─ Train Gradient Boosting per device                  │
│ └─ Evaluate: MAPE, R², confidence interval             │
│ └─ Output: .pkl models + scalers                       │
├─────────────────────────────────────────────────────────┤
│ PHASE 3: System Implementation (Chương 3)              │
│ └─ Build ML Controller (Flask server, 20+ APIs)        │
│ └─ Build ML Agents (Docker on 3 devices)               │
│ └─ Integrate prediction + deployment + monitoring      │
│ └─ Output: Production-ready code                       │
├─────────────────────────────────────────────────────────┤
│ PHASE 4: Testing & Validation (Chương 4)               │
│ └─ Test 1: Energy prediction accuracy                  │
│ └─ Test 2: End-to-end deployment                       │
│ └─ Test 3: Energy budget enforcement                   │
│ └─ Output: Test reports, metrics                       │
└─────────────────────────────────────────────────────────┘
```

### 5.2 Đánh Giá Thành Công

| Tiêu Chí | Đo Lường | Target |
|----------|----------|--------|
| **Accuracy** | MAPE (%) | < 20% |
| **Reliability** | CI coverage | > 95% |
| **Speed** | Deployment time (s) | < 60s |
| **Safety** | Budget enforcement | 100% |
| **Scalability** | Devices supported | 3+ |

---

## VI. Cấu Trúc Report

| Chương | Nội Dung | Trang |
|--------|----------|--------|
| **1** | Tổng Quan (Overview) | 1-8 |
| **2** | Thiết Kế Hệ Thống (Design) | 9-35 |
| **3** | Triển Khai (Implementation) | 36-70 |
| **4** | Kiểm Thử & Đánh Giá (Testing) | 71-95 |
| **5** | Kết Luận (Conclusion) | 96-110 |

---

## VII. Tóm Tắt Đóng Góp

### 7.1 Đóng Góp Chính

1. **Dataset Công Khai**
   - 274 models real-world benchmark data
   - Energy consumption per device
   - Reproducible, standardized format

2. **ML Model Chính Xác**
   - MAPE 18.69% (Jetson), 15.88% (RPi5)
   - Device-specific prediction
   - Confidence intervals

3. **End-to-End Automation**
   - Predict → Deploy → Monitor
   - 42 seconds per deployment
   - Energy budget enforcement

4. **Production Code**
   - 12,000+ LOC
   - Multi-device support (Jetson, RPi, BBB)
   - Docker containerization
   - RESTful APIs

### 7.2 Giá Trị Thực Tiễn

| Metric | Before | After | Improvement |
|--------|--------|-------|------------|
| Deployment Time | 60-120 min | 42 sec | 85-97% ⬇️ |
| Energy Planning | Manual | Automatic | 100% ⬆️ |
| Device Overload | Possible | Prevented | Auto-stop 🛡️ |
| Scalability | 5-10 devices | 100+ devices | 10x ⬆️ |

---

## VIII. Sơ Đồ Kiến Trúc Tổng Thể

```
┌────────────────────────────────────────────────────────────┐
│                  USER DASHBOARD                            │
│               (Web Browser)                                │
│          ├─ Deployment Tab                                │
│          ├─ Monitoring Tab                                │
│          └─ Analytics Tab                                 │
└─────────────────────┬────────────────────────────────────┘
                      │
                      ▼
         ┌────────────────────────────┐
         │   ML CONTROLLER SERVER     │
         │                            │
         │  Energy Prediction Service │
         │  └─ Jetson Model           │
         │  └─ RPi5 Model             │
         │  └─ Unified Model (fallback)│
         │                            │
         │  Flask REST APIs (20+)     │
         │  └─ /api/predict-energy    │
         │  └─ /api/deploy            │
         │  └─ /api/device/metrics    │
         │  └─ ...                    │
         │                            │
         │  Model Store               │
         │  └─ 15+ .onnx models       │
         │  └─ 15+ .tflite models     │
         │                            │
         │  Data                      │
         │  └─ Benchmark CSVs         │
         │  └─ Logs                   │
         └────────┬───────────────────┘
                  │ HTTP REST
      ┌───────────┼───────────┐
      │           │           │
      ▼           ▼           ▼
  ┌────────┐ ┌────────┐ ┌─────────┐
  │ Jetson │ │  RPi5  │ │   BBB   │
  │ Agent  │ │ Agent  │ │ Agent   │
  │(Docker)│ │(Docker)│ │(Docker) │
  │        │ │        │ │         │
  │TFLite/ │ │TFLite  │ │ TFLite  │
  │ONNX    │ │Runtime │ │Runtime  │
  │Runtime │ │        │ │         │
  └────────┘ └────────┘ └─────────┘
```

---

## IX. Kết Luận Chương 1

✅ Hệ thống này giải quyết **bài toán năng lượng** trong Edge AI thông qua:
1. **Dự báo**: Biết trước năng lượng sẽ tiêu thụ
2. **Tự động hóa**: Deploy nhanh từ phút xuống giây
3. **Kiểm soát**: Tự động dừng nếu vượt budget

💡 Kết hợp 3 yếu tố này tạo ra một **nền tảng MLOps hoàn chỉnh cho Edge AI**.

---

**Chương tiếp theo: Thiết Kế Hệ Thống (Chương 2) sẽ trình bày chi tiết cách xây dựng các thành phần trên.**
