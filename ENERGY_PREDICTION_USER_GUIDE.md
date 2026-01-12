# Energy Prediction - Hướng Dẫn Sử Dụng

## 🎯 Tính Năng Là Gì?

**Energy Prediction** giúp bạn dự đoán lượng năng lượng mà một AI model sẽ tiêu thụ TRƯỚC KHI deploy lên edge device (Jetson Nano hoặc Raspberry Pi 5).

### Tại Sao Cần?

✅ **Tiết kiệm pin/điện năng** cho thiết bị IoT  
✅ **Chọn model phù hợp** với khả năng thiết bị  
✅ **Tránh deploy nhầm** model quá nặng  
✅ **Deploy nhanh** với models đã tải sẵn

---

## 🚀 Cách Sử Dụng (5 Bước)

### Bước 1: Mở Trang Deployment

1. Truy cập: http://localhost:5000
2. Click tab **"Deployment"** trên menu

### Bước 2: Chọn Device

Chọn thiết bị bạn muốn deploy:

```
📱 Target Device
┌──────────────────────────────────────────┐
│ Jetson Nano (MAPE: 18.7%)        ▼      │
└──────────────────────────────────────────┘
```

- **Jetson Nano**: Độ chính xác 18.7% (train trên 247 models)
- **Raspberry Pi 5**: Độ chính xác 15.9% (train trên 27 models)

### Bước 3: Chọn Model

Chọn model từ danh sách 15+ popular models:

```
📦 Select popular model
┌──────────────────────────────────────────┐
│ EdgeNeXt XX-Small (1.33M params)  ▼     │
└──────────────────────────────────────────┘
```

**Các nhóm model:**
- 📱 **MobileNet** (5 models) - Siêu nhẹ cho mobile
- 🎯 **EfficientNet** (3 models) - Cân bằng hiệu suất/năng lượng
- 🏗️ **ResNet** (3 models) - Độ chính xác cao
- 🔧 **Others** (4 models) - SqueezeNet, ShuffleNet, etc.

Khi chọn xong, thông số sẽ TỰ ĐỘNG điền vào các ô bên dưới.

### Bước 4: Predict Energy

Nhấn nút **"Predict Energy"** màu xanh lớn.

Hệ thống sẽ hiển thị kết quả:

```
┌─────────────────────────────────────────┐
│        🟢 17.59 mWh                     │
│     EXCELLENT • Model MAPE: 18.7%       │
├─────────────────────────────────────────┤
│ Confidence Interval (95%): 14.3-20.9    │
│ Model Used: GradientBoostingRegressor   │
│ Status: ✅ EXCELLENT                    │
├─────────────────────────────────────────┤
│ 💡 Recommendation:                      │
│ Energy consumption is within excellent  │
│ range. Safe to deploy!                  │
├─────────────────────────────────────────┤
│    [🚀 Deploy edgenext_xx_small]       │
└─────────────────────────────────────────┘
```

### Bước 5: Deploy (Nếu Muốn)

Nếu kết quả là 🟢 **EXCELLENT** hoặc 🟡 **GOOD**, bạn có thể:

1. Nhấn nút **"🚀 Deploy [model_name]"**
2. Model sẽ được deploy lên device đã chọn
3. Kiểm tra kết quả trong tab **"Monitoring"**

---

## 📊 Hiểu Kết Quả

### Energy Categories

Hệ thống phân loại models thành 4 mức:

| Icon | Category | Jetson Nano | Raspberry Pi 5 | Ý Nghĩa |
|------|----------|-------------|----------------|---------|
| 🟢 | **EXCELLENT** | < 34.6 mWh | < 11.1 mWh | ✅ **DEPLOY NGAY** - Tiêu thụ thấp nhất |
| 🟡 | **GOOD** | 34.6-104.6 mWh | 11.1-18.0 mWh | ✅ **DEPLOY OK** - Tiêu thụ vừa phải |
| 🟠 | **ACCEPTABLE** | 104.6-235.3 mWh | 18.0-30.7 mWh | ⚠️ **CÂN NHẮC** - Tiêu thụ cao |
| 🔴 | **HIGH** | > 235.3 mWh | > 30.7 mWh | ❌ **KHÔNG KHUYẾN NGHỊ** - Quá nặng |

### Recommendations

- **deploy**: Model phù hợp, deploy ngay!
- **deploy_with_caution**: Model hơi nặng, cân nhắc tối ưu
- **not_recommend**: Model quá nặng, chọn model khác

### Model Downloaded Badge

- ✅ **DOWNLOADED**: Model đã có sẵn trong `model_store`, deploy ngay được
- ⚠️ **NOT DOWNLOADED**: Model chưa có, cần download trước (chỉ EXCELLENT models được tải sẵn)

---

## 📝 Ví Dụ Thực Tế

### Ví Dụ 1: EdgeNeXt XX-Small (EXCELLENT)

```
Device: Jetson Nano
Model: EdgeNeXt XX-Small (1.33M params)

Kết Quả:
🟢 17.59 mWh - EXCELLENT
Confidence Interval: 14.3 - 20.9 mWh
Status: ✅ DOWNLOADED

→ ĐỀ XUẤT: Deploy ngay! Model rất nhẹ, phù hợp cho edge device.
```

### Ví Dụ 2: ResNet-18 (GOOD)

```
Device: Jetson Nano
Model: ResNet-18 (11.69M params)

Kết Quả:
🟡 54.4 mWh - GOOD
Confidence Interval: 44.2 - 64.6 mWh
Status: ✅ DOWNLOADED

→ ĐỀ XUẤT: Có thể deploy, nhưng tiêu thụ cao hơn MobileNet.
```

### Ví Dụ 3: VGG-16 (HIGH)

```
Device: Jetson Nano
Model: VGG-16 (138.36M params)

Kết Quả:
🔴 607.6 mWh - HIGH
Confidence Interval: 494.0 - 721.2 mWh
Status: ⚠️ NOT DOWNLOADED

→ ĐỀ XUẤT: KHÔNG deploy! Model quá nặng cho Jetson Nano.
   Đề xuất: Dùng MobileNetV3 hoặc EfficientNet thay thế.
```

---

## 🎨 Popular Models Cheat Sheet

### 🟢 EXCELLENT Models (Jetson Nano)

| Model | Params | Energy | Use Case |
|-------|--------|--------|----------|
| MobileNetV3 Small 0.5x | 1.53M | 11.8 mWh | IoT, Real-time |
| EdgeNeXt XX-Small | 1.33M | 17.6 mWh | Modern, Efficient |
| SqueezeNet 1.0 | 1.25M | 28.5 mWh | Compact, Fast |
| MobileNetV3 Small 1.0x | 2.54M | 11.5 mWh | Balanced |
| MobileNetV2 1.0x | 3.50M | 20.2 mWh | Standard |
| ShuffleNetV2 0.5x | 1.37M | 12.0 mWh | Lightweight |
| MobileNetV3 Large 1.0x | 5.48M | 22.3 mWh | More Accurate |
| EfficientNet-Lite0 | 4.65M | 30.0 mWh | Edge-optimized |

### 🟡 GOOD Models

| Model | Params | Energy | Use Case |
|-------|--------|--------|----------|
| ResNet-18 | 11.69M | 54.4 mWh | Classic, Accurate |
| EfficientNet-B0 | 5.29M | 55.2 mWh | Efficient |

### 🟠 ACCEPTABLE Models

| Model | Params | Energy | Use Case |
|-------|--------|--------|----------|
| ResNet-34 | 21.80M | 110.1 mWh | Medium ResNet |
| ResNet-50 | 25.56M | 110.5 mWh | Standard ResNet |
| DenseNet-121 | 7.98M | 107.8 mWh | Dense connections |

### 🔴 HIGH Models (Tránh Deploy)

| Model | Params | Energy | Why Not? |
|-------|--------|--------|----------|
| VGG-16 | 138.36M | 607.6 mWh | Cực nặng, không tối ưu |

---

## ❓ FAQs

### Q: Làm sao để thêm model mới?

**A:** Contact admin để thêm model vào `popular_models_metadata.json`. Hoặc tự thêm specs và chạy:
```bash
python generate_popular_models_metadata.py
```

### Q: Tại sao một số models không có nút Deploy?

**A:** Chỉ models có category EXCELLENT mới được tải sẵn trong `model_store`. Models khác cần download manual.

### Q: Làm sao để deploy model khác thiết bị hiện tại?

**A:** Chọn device khác trong dropdown "Target Device" rồi predict lại.

### Q: Kết quả có chính xác không?

**A:** 
- Jetson Nano: MAPE 18.7% (sai số trung bình ~19%)
- Raspberry Pi 5: MAPE 15.9% (sai số trung bình ~16%)
- Confidence Interval 95% được hiển thị để thấy range có thể

### Q: Model nào tốt nhất?

**A:** Tùy use case:
- **Real-time, Low Power**: MobileNetV3 Small 0.5x (11.8 mWh)
- **Balanced**: EdgeNeXt XX-Small (17.6 mWh)
- **More Accurate**: MobileNetV3 Large (22.3 mWh)
- **Classic**: ResNet-18 (54.4 mWh - nếu không ngại tốn năng lượng)

---

## 🛠️ Troubleshooting

### Lỗi: "Energy prediction failed"

**Nguyên nhân:** API lỗi hoặc model chưa load

**Cách sửa:**
1. Kiểm tra Flask server đang chạy
2. Reload trang
3. Thử model khác

### Lỗi: "Model not found in model_store"

**Nguyên nhân:** Model chưa được download

**Cách sửa:**
```bash
cd ml-controller/python
python download_excellent_models.py
```

### Kết quả "Loading popular models..."

**Nguyên nhân:** API `/api/models/popular` lỗi

**Cách sửa:**
1. Check browser console (F12)
2. Kiểm tra file `popular_models_metadata.json` có tồn tại
3. Restart Flask server

---

## 📞 Support

- Technical Guide: `ENERGY_PREDICTION_TECHNICAL_GUIDE.md`
- Main README: `README.md`
- Notebook: `ml-controller/notebooks/energy_prediction_model.ipynb`
- GitHub Issues: [Link to your repo]

---

**Chúc bạn deploy thành công! 🚀**
