# Hướng Dẫn Huấn Luyện Lại Energy Prediction Models

## 📋 Tổng Quan

Tính năng Energy Prediction hiện có **MAPE 15-19%** (tốt hơn bạn nghĩ!), nhưng có thể cải thiện thêm.

### Hiện trạng:
- ✅ **Jetson Nano**: MAPE 18.69%, R²=0.86 (247 models)
- ✅ **Raspberry Pi 5**: MAPE 15.88%, R²=0.95 (27 models)
- ⚠️ **Vấn đề**: RPi5 có quá ít data (27 models), Jetson có thể cải thiện thêm

---

## 🚀 Bước 1: Chuẩn Bị Môi Trường

### 1.1. Chuyển vào thư mục dự án
```powershell
cd D:\DACN_BACKUP\DACN_Energy_consumption\ml-controller
```

### 1.2. Cài đặt dependencies
```powershell
pip install -r requirements.txt

# Hoặc cài thủ công:
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
pip install xgboost lightgbm  # Optional: để thử thuật toán mới
```

### 1.3. Mở Jupyter Notebook
```powershell
jupyter notebook notebooks/energy_prediction_model.ipynb
```

Hoặc mở trực tiếp trong VS Code:
```powershell
code notebooks/energy_prediction_model.ipynb
```

---

## 📊 Bước 2: Thu Thập Thêm Dữ Liệu (Khuyến nghị!)

### 2.1. Benchmark thêm models trên Raspberry Pi 5

**Cần:** Tăng từ 27 → 80-100 models để cải thiện độ chính xác

**Cách làm:**
1. Chạy benchmark framework trên RPi5:
```bash
# Trên Raspberry Pi 5
cd /path/to/benchmark
python benchmark_models.py --device rpi5 --models all
```

2. Export kết quả ra CSV với format:
```csv
model,params_m,gflops,gmacs,size_mb,latency_avg_s,throughput_iter_per_s,energy_avg_mwh
mobilenetv3_small_075,2.04,0.044,0.022,8.2,0.0234,42.7,8.5
efficientnet_b0,5.29,0.39,0.19,21.1,0.0567,17.6,15.2
...
```

3. Append vào file:
```powershell
# Merge với file hiện tại
# data/27_models_benchmark_rpi5.csv
```

### 2.2. (Optional) Benchmark thêm models trên Jetson Nano

Jetson đã có 247 models (đủ), nhưng nếu muốn thêm:
```bash
# Trên Jetson Nano
python benchmark_models.py --device jetson --models new_architectures
```

---

## 🔧 Bước 3: Retrain Models

### 3.1. Run toàn bộ Notebook

**Trong VS Code hoặc Jupyter:**
1. Mở [energy_prediction_model.ipynb](notebooks/energy_prediction_model.ipynb)
2. Click **"Run All"** hoặc `Ctrl+Shift+Enter` cho từng cell
3. Đợi khoảng 5-10 phút

### 3.2. Kiểm tra kết quả training

Sau khi run, xem metrics:

```
=== Jetson Nano Model ===
Test MAPE: 18.69%  (target: < 15%)
Test R²: 0.860      (target: > 0.90)

=== Raspberry Pi 5 Model ===
LOO MAPE: 15.88%   (target: < 12%)
LOO R²: 0.946      (target: > 0.95)
```

### 3.3. Export models mới

Notebook sẽ tự động lưu models vào `artifacts/`:
```
✅ Saved: artifacts/jetson_energy_model.pkl
✅ Saved: artifacts/jetson_scaler.pkl
✅ Saved: artifacts/rpi5_energy_model.pkl
✅ Saved: artifacts/rpi5_scaler.pkl
✅ Saved: artifacts/device_specific_metadata.json
```

---

## ⚡ Bước 4: Cải Thiện Nâng Cao

### 4.1. Thử Thuật Toán Khác (XGBoost, LightGBM)

**Thêm cell mới vào notebook:**

```python
# Cell mới: Thử XGBoost
import xgboost as xgb

# Train XGBoost cho Jetson
xgb_model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    random_state=42
)
xgb_model.fit(X_train_scaled, y_train)
xgb_pred = xgb_model.predict(X_test_scaled)

# Đánh giá
from sklearn.metrics import mean_absolute_percentage_error, r2_score
xgb_mape = mean_absolute_percentage_error(y_test, xgb_pred) * 100
xgb_r2 = r2_score(y_test, xgb_pred)

print(f"XGBoost MAPE: {xgb_mape:.2f}%")
print(f"XGBoost R²: {xgb_r2:.3f}")
```

```python
# Cell mới: Thử LightGBM
import lightgbm as lgb

lgb_model = lgb.LGBMRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    random_state=42
)
lgb_model.fit(X_train_scaled, y_train)
lgb_pred = lgb_model.predict(X_test_scaled)

lgb_mape = mean_absolute_percentage_error(y_test, lgb_pred) * 100
lgb_r2 = r2_score(y_test, lgb_pred)

print(f"LightGBM MAPE: {lgb_mape:.2f}%")
print(f"LightGBM R²: {lgb_r2:.3f}")
```

### 4.2. Hyperparameter Tuning

**Thêm cell tuning:**

```python
from sklearn.model_selection import GridSearchCV

# Grid search cho GradientBoostingRegressor
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'min_samples_split': [2, 5, 10],
    'subsample': [0.8, 0.9, 1.0]
}

grid_search = GridSearchCV(
    GradientBoostingRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring='neg_mean_absolute_percentage_error',
    n_jobs=-1,
    verbose=2
)

print("🔍 Searching for best hyperparameters...")
grid_search.fit(X_train_scaled, y_train)

print(f"✅ Best params: {grid_search.best_params_}")
print(f"✅ Best CV score: {-grid_search.best_score_:.2f}%")

# Sử dụng best model
best_model = grid_search.best_estimator_
```

### 4.3. Feature Engineering Mới

**Thêm features vào cell Feature Engineering:**

```python
# Thêm vào section "Derived Features"

# 1. Architecture complexity
df['arch_complexity'] = df['params_m'] * df['gflops'] / (df['size_mb'] + 1e-6)

# 2. Efficiency score
df['efficiency_score'] = df['throughput_iter_per_s'] / (df['energy_avg_mwh'] + 1e-6)

# 3. Memory bandwidth requirement
df['memory_bandwidth'] = df['size_mb'] / (df['latency_avg_s'] + 1e-6)

# 4. FLOPs per second
df['flops_per_second'] = df['gflops'] * 1e9 * df['throughput_iter_per_s']

# 5. Energy per FLOP
df['energy_per_gflop'] = df['energy_avg_mwh'] / (df['gflops'] + 1e-6)

# Update feature list
feature_cols = [
    'params_m', 'gflops', 'gmacs', 'size_mb', 
    'latency_avg_s', 'throughput_iter_per_s',
    'params_per_gflop', 'gflops_per_mb', 'computational_density',
    'arch_complexity', 'efficiency_score', 'memory_bandwidth',
    'flops_per_second', 'energy_per_gflop'
]
```

### 4.4. Ensemble Models

**Thêm cell ensemble:**

```python
# Ensemble: Kết hợp nhiều models
from sklearn.ensemble import VotingRegressor

ensemble = VotingRegressor([
    ('gb', GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05)),
    ('xgb', xgb.XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05)),
    ('lgb', lgb.LGBMRegressor(n_estimators=200, max_depth=5, learning_rate=0.05))
])

ensemble.fit(X_train_scaled, y_train)
ensemble_pred = ensemble.predict(X_test_scaled)

ensemble_mape = mean_absolute_percentage_error(y_test, ensemble_pred) * 100
ensemble_r2 = r2_score(y_test, ensemble_pred)

print(f"Ensemble MAPE: {ensemble_mape:.2f}%")
print(f"Ensemble R²: {ensemble_r2:.3f}")
```

---

## ✅ Bước 5: Test Models Mới

### 5.1. Restart Flask server

```powershell
cd D:\DACN_BACKUP\DACN_Energy_consumption\ml-controller\python
python app.py
```

### 5.2. Test qua web dashboard

1. Mở http://localhost:5000
2. Chọn device (Jetson hoặc RPi5)
3. Chọn model từ popular list
4. Click **"Predict Energy"**
5. Xem kết quả và MAPE mới

### 5.3. Test qua API

```powershell
# Test Jetson
Invoke-RestMethod -Uri "http://localhost:5000/api/predict" -Method POST -ContentType "application/json" -Body '{
  "payloads": [{
    "device_type": "jetson_nano",
    "model": "mobilenetv3_small_075",
    "params_m": 2.04,
    "gflops": 0.044,
    "gmacs": 0.022,
    "size_mb": 8.2,
    "latency_avg_s": 0.0234,
    "throughput_iter_per_s": 42.7
  }]
}'
```

---

## 📈 Bước 6: So Sánh Kết Quả

### 6.1. Metrics cũ (hiện tại):
```
Jetson: MAPE 18.69%, R² 0.860
RPi5:   MAPE 15.88%, R² 0.946
```

### 6.2. Metrics mới (sau khi retrain):
```
# Ghi lại kết quả sau khi chạy xong notebook
Jetson: MAPE ___%, R² ___
RPi5:   MAPE ___%, R² ___
```

### 6.3. Target benchmarks:
```
✅ EXCELLENT: MAPE < 12%, R² > 0.95
✅ GOOD:      MAPE < 18%, R² > 0.90
⚠️ ACCEPTABLE: MAPE < 25%, R² > 0.80
```

---

## 🎯 Checklist Cải Thiện

- [ ] Thu thập thêm 50-70 models cho RPi5 (quan trọng nhất!)
- [ ] Run lại toàn bộ notebook hiện tại
- [ ] Thử XGBoost
- [ ] Thử LightGBM
- [ ] Hyperparameter tuning với GridSearchCV
- [ ] Thêm features mới
- [ ] Ensemble models
- [ ] Test và so sánh kết quả
- [ ] Update metadata file
- [ ] Document improvements

---

## 🔍 Troubleshooting

### Lỗi: "ModuleNotFoundError: No module named 'sklearn'"
```powershell
pip install scikit-learn
```

### Lỗi: "FileNotFoundError: data/247_models_benchmark_jetson.csv"
```powershell
# Kiểm tra path
cd D:\DACN_BACKUP\DACN_Energy_consumption\ml-controller
ls data\
```

### Notebook chạy chậm
- Giảm `n_estimators` xuống 100-150
- Tắt GridSearchCV (chạy riêng sau)
- Sử dụng subset data để test nhanh

### MAPE vẫn cao sau retrain
- **Nguyên nhân 1**: Data quality (outliers, missing values)
- **Nguyên nhân 2**: Feature engineering chưa tốt
- **Nguyên nhân 3**: Quá ít data (đặc biệt RPi5)
- **Giải pháp**: Thu thập thêm data là quan trọng nhất!

---

## 📚 Tài Liệu Tham Khảo

- [Notebook Training](notebooks/energy_prediction_model.ipynb)
- [Energy Predictor Service](python/energy_predictor_service.py)
- [Current Metadata](artifacts/device_specific_metadata.json)
- [User Guide](ENERGY_PREDICTION_USER_GUIDE.md)

---

**🎓 Tips:**
1. **Data > Algorithm**: Thu thập thêm data hiệu quả hơn tune model
2. **Start Simple**: Chạy lại notebook hiện tại trước khi thử advanced techniques
3. **Validate Carefully**: Luôn test trên unseen data
4. **Document Everything**: Ghi lại mọi thay đổi và kết quả
