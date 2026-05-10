"""
Quick experiment to find the best energy prediction model.
Tests: ExtraTrees baseline, ExtraTrees+log, XGBoost+log, Stacking.
Uses 5-fold CV on full dataset. Saves best models to artifacts/.
"""
import os, json, pickle, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
warnings.filterwarnings("ignore")

ARTIFACTS_DIR = "ml-controller/artifacts"
DATA_DIR = "ml-controller/data"

# ── Extended feature set (25 features, all prediction-time computable) ──
FEATURES = [
    "params_m", "gflops", "gmacs", "size_mb",
    "latency_avg_s", "throughput_iter_per_s",
    # Derived ratios
    "gflops_per_param", "gmacs_per_mb", "latency_throughput_ratio",
    "compute_intensity", "model_complexity", "computational_density",
    "latency_per_gflop", "size_per_param",
    # Log transforms
    "log_params_m", "log_gflops", "log_size_mb", "log_gmacs",
    "log_latency", "log_throughput", "log_model_complexity", "log_compute_intensity",
    "log_latency_per_gflop",
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
    out["latency_per_gflop"] = out["latency_avg_s"] / (out["gflops"] + eps)
    out["size_per_param"] = out["size_mb"] / (out["params_m"] + eps)
    out["log_params_m"] = np.log1p(out["params_m"])
    out["log_gflops"] = np.log1p(out["gflops"])
    out["log_size_mb"] = np.log1p(out["size_mb"])
    out["log_gmacs"] = np.log1p(out["gmacs"])
    out["log_latency"] = np.log1p(out["latency_avg_s"])
    out["log_throughput"] = np.log1p(out["throughput_iter_per_s"])
    out["log_model_complexity"] = np.log1p(out["model_complexity"])
    out["log_compute_intensity"] = np.log1p(out["compute_intensity"])
    out["log_latency_per_gflop"] = np.log1p(out["latency_per_gflop"])
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


def eval_cv(model, X, y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    preds = cross_val_predict(model, X, y, cv=kf)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y, preds))),
        "mae": float(mean_absolute_error(y, preds)),
        "mape": mape(y, preds),
        "r2": r2(y, preds),
    }


def eval_cv_log(model, X, y_raw, n_splits=5):
    """CV with log-transformed target; metrics reported in original scale."""
    y_log = np.log1p(y_raw)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    preds_log = cross_val_predict(model, X, y_log, cv=kf)
    preds = np.expm1(preds_log)
    preds = np.clip(preds, 0, None)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_raw, preds))),
        "mae": float(mean_absolute_error(y_raw, preds)),
        "mape": mape(y_raw, preds),
        "r2": r2(y_raw, preds),
    }


def print_metrics(name, m):
    print(f"  {name:40s}  RMSE={m['rmse']:8.3f}  MAE={m['mae']:7.3f}  MAPE={m['mape']:6.2f}%  R2={m['r2']:.4f}")


def experiment_device(df, device_name, device_type):
    print(f"\n{'='*70}")
    print(f"Device: {device_name}  ({len(df)} samples)")
    print(f"  Energy range: {df['energy_avg_mwh'].min():.2f} – {df['energy_avg_mwh'].max():.2f} mWh")

    df_feat = build_features(df, device_type)
    feat_cols = [c for c in FEATURES if c in df_feat.columns]
    X = df_feat[feat_cols].values
    y = df["energy_avg_mwh"].values

    # ── Models ──
    et = ExtraTreesRegressor(n_estimators=500, max_features=0.6, min_samples_leaf=1, random_state=42, n_jobs=-1)
    rf = RandomForestRegressor(n_estimators=300, max_features=0.6, min_samples_leaf=2, random_state=42, n_jobs=-1)
    xgb = XGBRegressor(n_estimators=600, learning_rate=0.04, max_depth=6,
                       subsample=0.8, colsample_bytree=0.8,
                       reg_alpha=0.05, reg_lambda=1.0, min_child_weight=3,
                       random_state=42, n_jobs=-1, verbosity=0)
    xgb_deep = XGBRegressor(n_estimators=800, learning_rate=0.03, max_depth=8,
                            subsample=0.75, colsample_bytree=0.7,
                            reg_alpha=0.1, reg_lambda=2.0, min_child_weight=2,
                            random_state=42, n_jobs=-1, verbosity=0)

    results = {}
    print("\n  Raw target:")
    for name, mdl in [("ExtraTrees", et), ("XGBoost", xgb)]:
        m = eval_cv(mdl, X, y)
        results[name] = m
        print_metrics(name, m)

    print("\n  Log1p target (metrics in original mWh):")
    for name, mdl in [("ExtraTrees+log", et), ("XGBoost+log", xgb), ("XGBoost-deep+log", xgb_deep)]:
        m = eval_cv_log(mdl, X, y)
        results[name] = m
        print_metrics(name, m)

    # Stacking with log target
    estimators = [("et", ExtraTreesRegressor(n_estimators=300, max_features=0.6, random_state=42, n_jobs=-1)),
                  ("xgb", XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=6, random_state=42, verbosity=0, n_jobs=-1))]
    stack = StackingRegressor(estimators=estimators, final_estimator=Ridge(alpha=1.0), cv=3, n_jobs=-1)
    m_stack = eval_cv_log(stack, X, y)
    results["Stacking+log"] = m_stack
    print_metrics("Stacking(ET+XGB)+log", m_stack)

    # Pick best by MAPE
    best_name = min(results, key=lambda k: results[k]["mape"])
    best_m = results[best_name]
    print(f"\n  Best: {best_name}  ->  MAPE={best_m['mape']:.2f}%  R2={best_m['r2']:.4f}")
    return results, best_name, feat_cols


if __name__ == "__main__":
    jetson = pd.read_csv(os.path.join(DATA_DIR, "360_models_benchmark_jetson.csv"))
    rpi5   = pd.read_csv(os.path.join(DATA_DIR, "253_models_benchmark_rpi5.csv"))

    j_results, j_best, j_feats = experiment_device(jetson, "Jetson Nano", "jetson_nano")
    r_results, r_best, r_feats = experiment_device(rpi5, "Raspberry Pi 5", "raspberry_pi5")

    print("\n" + "="*70)
    print("SUMMARY")
    print(f"  Jetson best: {j_best}  MAPE={j_results[j_best]['mape']:.2f}%  RMSE={j_results[j_best]['rmse']:.3f}  R2={j_results[j_best]['r2']:.4f}")
    print(f"  RPi5   best: {r_best}  MAPE={r_results[r_best]['mape']:.2f}%  RMSE={r_results[r_best]['rmse']:.3f}  R2={r_results[r_best]['r2']:.4f}")
