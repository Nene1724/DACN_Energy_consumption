"""
Retrain device-specific production models using ONLY features computable at prediction time.
These 23 features can be extracted from ONNX/TFLite + runtime benchmark results.

Run: .venv\Scripts\python retrain_production_models.py
"""
import os, json, pickle, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
warnings.filterwarnings("ignore")

ARTIFACTS_DIR = "ml-controller/artifacts"
DATA_DIR = "ml-controller/data"

# Features computable at prediction time from model metadata + quick benchmark
PRODUCTION_FEATURES = [
    "params_m", "gflops", "gmacs", "size_mb",
    "latency_avg_s", "throughput_iter_per_s",
    # Derived
    "gflops_per_param", "gmacs_per_mb", "latency_throughput_ratio",
    "compute_intensity", "model_complexity", "computational_density",
    # Log transforms
    "log_params_m", "log_gflops", "log_size_mb", "log_gmacs",
    "log_latency", "log_throughput", "log_model_complexity", "log_compute_intensity",
    # Interactions
    "log_params_x_log_latency", "log_gflops_x_log_latency",
    # Device flag
    "batch_high_power",
]

def build_features(df: pd.DataFrame, device_type: str) -> pd.DataFrame:
    out = df.copy()
    eps = 1e-6
    out["gflops_per_param"] = out["gflops"] / (out["params_m"] + eps)
    out["gmacs_per_mb"] = out["gmacs"] / (out["size_mb"] + eps)
    out["latency_throughput_ratio"] = out["latency_avg_s"] * out["throughput_iter_per_s"]
    out["compute_intensity"] = out["gflops"] * out["latency_avg_s"]
    out["model_complexity"] = out["params_m"] * out["gflops"]
    out["computational_density"] = out["gflops"] / (out["size_mb"] + eps)
    out["log_params_m"] = np.log1p(out["params_m"])
    out["log_gflops"] = np.log1p(out["gflops"])
    out["log_size_mb"] = np.log1p(out["size_mb"])
    out["log_gmacs"] = np.log1p(out["gmacs"])
    out["log_latency"] = np.log1p(out["latency_avg_s"])
    out["log_throughput"] = np.log1p(out["throughput_iter_per_s"])
    out["log_model_complexity"] = np.log1p(out["model_complexity"])
    out["log_compute_intensity"] = np.log1p(out["compute_intensity"])
    out["log_params_x_log_latency"] = out["log_params_m"] * out["log_latency"]
    out["log_gflops_x_log_latency"] = out["log_gflops"] * out["log_latency"]
    out["batch_high_power"] = 1 if device_type == "jetson_nano" else 0
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    out.fillna(0, inplace=True)
    return out


def mape(y_true, y_pred):
    mask = np.abs(y_true) > 1e-6
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))) * 100


def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-10))


def train_device_model(df: pd.DataFrame, device_name: str, device_type: str):
    print(f"\n{'='*60}")
    print(f"Training {device_name} ({len(df)} samples)")
    df_feat = build_features(df, device_type)
    feat_cols = [c for c in PRODUCTION_FEATURES if c in df_feat.columns]
    X = df_feat[feat_cols].values
    y = df["energy_avg_mwh"].values

    # Stratified CV using energy quantile bins (avoids bad splits on skewed data)
    from sklearn.model_selection import cross_val_predict
    n_bins = min(5, len(y) // 10)
    quantile_bins = pd.qcut(pd.Series(y), q=n_bins, labels=False, duplicates="drop")

    # Best hyperparams (known from prior experiments)
    best = ExtraTreesRegressor(
        n_estimators=500, max_features=0.6, min_samples_leaf=1,
        max_depth=None, random_state=42, n_jobs=-1
    )

    # 5-fold CV evaluation
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_preds = cross_val_predict(best, X, y, cv=kf)

    rmse_val = float(np.sqrt(mean_squared_error(y, cv_preds)))
    mae_val = float(mean_absolute_error(y, cv_preds))
    mape_val = mape(y, cv_preds)
    r2_val = r2(y, cv_preds)
    print(f"5-Fold CV RMSE: {rmse_val:.3f} mWh")
    print(f"5-Fold CV MAE:  {mae_val:.3f} mWh")
    print(f"5-Fold CV MAPE: {mape_val:.2f}%")
    print(f"5-Fold CV R2:   {r2_val:.4f}")

    # Train final model on ALL data for production
    best.fit(X, y)
    print(f"Final model trained on all {len(X)} samples")

    # Scaler (kept for compatibility)
    scaler = StandardScaler()
    scaler.fit(X)

    return best, scaler, feat_cols, {
        "rmse": rmse_val,
        "mae": mae_val,
        "mape": mape_val,
        "r2": r2_val,
        "n_cv": int(len(y)),
        "log_transform_target": False,
    }


def main():
    # Load data
    jetson = pd.read_csv(os.path.join(DATA_DIR, "360_models_benchmark_jetson.csv"))
    rpi5 = pd.read_csv(os.path.join(DATA_DIR, "253_models_benchmark_rpi5.csv"))
    print(f"Jetson: {len(jetson)} samples, RPi5: {len(rpi5)} samples")

    # Train Jetson model
    j_model, j_scaler, j_feats, j_metrics = train_device_model(jetson, "Jetson Nano", "jetson_nano")

    # Train RPi5 model
    r_model, r_scaler, r_feats, r_metrics = train_device_model(rpi5, "Raspberry Pi 5", "raspberry_pi5")

    # Save models
    with open(os.path.join(ARTIFACTS_DIR, "jetson_energy_model.pkl"), "wb") as f:
        pickle.dump(j_model, f)
    with open(os.path.join(ARTIFACTS_DIR, "jetson_scaler.pkl"), "wb") as f:
        pickle.dump(j_scaler, f)
    with open(os.path.join(ARTIFACTS_DIR, "rpi5_energy_model.pkl"), "wb") as f:
        pickle.dump(r_model, f)
    with open(os.path.join(ARTIFACTS_DIR, "rpi5_scaler.pkl"), "wb") as f:
        pickle.dump(r_scaler, f)

    # Save feature list as a FLAT LIST (required by EnergyPredictorService)
    with open(os.path.join(ARTIFACTS_DIR, "device_specific_features.json"), "w") as f:
        json.dump(PRODUCTION_FEATURES, f, indent=2)

    # Save metadata
    metadata = {
        "jetson_model": {
            "model_name": "ExtraTreesRegressor",
            "metrics": {
                "cv_rmse": j_metrics["rmse"],
                "cv_mae": j_metrics["mae"],
                "cv_mape_eps": j_metrics["mape"],
                "cv_r2": j_metrics["r2"],
            },
            "log_transform_target": False,
            "feature_count": len(j_feats),
        },
        "rpi5_model": {
            "model_name": "ExtraTreesRegressor",
            "metrics": {
                "cv_rmse": r_metrics["rmse"],
                "cv_mae": r_metrics["mae"],
                "cv_mape_eps": r_metrics["mape"],
                "cv_r2": r_metrics["r2"],
            },
            "log_transform_target": False,
            "feature_count": len(r_feats),
        },
    }
    with open(os.path.join(ARTIFACTS_DIR, "device_specific_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Compute energy thresholds
    all_energy = pd.concat([jetson["energy_avg_mwh"], rpi5["energy_avg_mwh"]]).dropna()
    j_energy = jetson["energy_avg_mwh"].dropna()
    r_energy = rpi5["energy_avg_mwh"].dropna()
    thresholds = {
        "jetson_nano": {
            "p25": float(np.percentile(j_energy, 25)),
            "p50": float(np.percentile(j_energy, 50)),
            "p75": float(np.percentile(j_energy, 75)),
        },
        "raspberry_pi5": {
            "p25": float(np.percentile(r_energy, 25)),
            "p50": float(np.percentile(r_energy, 50)),
            "p75": float(np.percentile(r_energy, 75)),
        },
        "unified": {
            "p25": float(np.percentile(all_energy, 25)),
            "p50": float(np.percentile(all_energy, 50)),
            "p75": float(np.percentile(all_energy, 75)),
        }
    }
    with open(os.path.join(ARTIFACTS_DIR, "energy_thresholds.json"), "w") as f:
        json.dump(thresholds, f, indent=2)

    print("\nAll artifacts saved to", ARTIFACTS_DIR)
    print(f"\nJetson: RMSE={j_metrics['rmse']:.1f}, MAPE={j_metrics['mape']:.2f}%, R²={j_metrics['r2']:.4f}")
    print(f"RPi5:   RMSE={r_metrics['rmse']:.1f}, MAPE={r_metrics['mape']:.2f}%, R²={r_metrics['r2']:.4f}")


if __name__ == "__main__":
    main()
