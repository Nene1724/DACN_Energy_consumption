import sys
sys.path.insert(0, "ml-controller/python")
from energy_predictor_service import EnergyPredictorService

svc = EnergyPredictorService("ml-controller/artifacts")
print("Jetson MAPE:", round(svc.jetson_cfg["mape"]*100, 1), "%")
print("RPi5 MAPE:", round(svc.rpi5_cfg["mape"]*100, 1), "%")
print("Features:", len(svc.feature_names), svc.feature_names[:5])

result = svc.predict([{
    "device_type": "jetson_nano",
    "model_name": "lcnet050",
    "params_m": 1.88,
    "gflops": 0.045,
    "gmacs": 0.022,
    "size_mb": 7.2,
    "latency_avg_s": 0.012,
    "throughput_iter_per_s": 83.0,
}])
p = result[0]
print("Prediction:", p.get("prediction_mwh"), "mWh")
print("CI:", p.get("ci_lower_mwh"), "-", p.get("ci_upper_mwh"))
print("Model used:", p.get("model_used"))
print("MAPE pct:", p.get("mape_pct"))
