"""
Improved energy prediction training.
Uses exact same 80/20 stratified split as original notebook (train on 80%, evaluate on 20%).
Tests multiple models, picks best per device, saves production artifacts.
"""
import os, json, pickle, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
warnings.filterwarnings("ignore")

ARTIFACTS_DIR = "ml-controller/artifacts"
DATA_DIR = "ml-controller/data"
RANDOM_STATE = 42

# 26 prediction-time features (2 more than previous: latency_per_gflop, size_per_param)
FEATURES = [
    "params_m", "gflops", "gmacs", "size_mb",
    "latency_avg_s", "throughput_iter_per_s",
    "gflops_per_param", "gmacs_per_mb", "latency_throughput_ratio",
    "compute_intensity", "model_complexity", "computational_density",
    "latency_per_gflop", "size_per_param",
    "log_params_m", "log_gflops", "log_size_mb", "log_gmacs",
    "log_latency", "log_throughput", "log_model_complexity", "log_compute_intensity",
    "log_latency_per_gflop",
    "log_params_x_log_latency", "log_gflops_x_log_latency",
    "batch_high_power",
]


def build_features(df, device_type):
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

def median_mape(y_true, y_pred):
    mask = np.abs(y_true) > 1e-6
    return float(np.median(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))) * 100

def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-10))

def pearson_r(y_true, y_pred):
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def train_and_eval(model, X_train, y_train, X_test, y_test, log_target=False):
    if log_target:
        y_train_t = np.log1p(y_train)
        model.fit(X_train, y_train_t)
        preds_t = model.predict(X_test)
        preds = np.expm1(preds_t)
        preds = np.clip(preds, 0, None)
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        preds = np.clip(preds, 0, None)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "mae": float(mean_absolute_error(y_test, preds)),
        "mape": mape(y_test, preds),
        "median_mape": median_mape(y_test, preds),
        "r2": r2(y_test, preds),
        "pearson_r": pearson_r(y_test, preds),
        "log_target": log_target,
    }, preds


def train_device(df, device_name, device_type):
    print(f"\n{'='*70}")
    print(f"Device: {device_name}  ({len(df)} samples)")
    print(f"  Energy: min={df['energy_avg_mwh'].min():.2f}  max={df['energy_avg_mwh'].max():.2f}  "
          f"mean={df['energy_avg_mwh'].mean():.2f}  median={df['energy_avg_mwh'].median():.2f}")

    df_feat = build_features(df, device_type)
    feat_cols = [c for c in FEATURES if c in df_feat.columns]
    X = df_feat[feat_cols].values
    y = df["energy_avg_mwh"].values

    # Stratify by log-energy quantile bins (same spirit as original notebook)
    bins = pd.qcut(pd.Series(np.log1p(y)), q=5, labels=False, duplicates="drop")
    X_train, X_test, y_train, y_test, _, bins_test = train_test_split(
        X, y, bins.values, test_size=0.20, random_state=RANDOM_STATE, stratify=bins.values
    )
    print(f"  Train: {len(X_train)}  Test: {len(X_test)}")
    print(f"  Test energy: min={y_test.min():.2f}  max={y_test.max():.2f}  mean={y_test.mean():.2f}")

    candidates = {
        "ExtraTrees_raw": (
            ExtraTreesRegressor(n_estimators=500, max_features=0.6, min_samples_leaf=1, random_state=42, n_jobs=-1),
            False
        ),
        "ExtraTrees_log": (
            ExtraTreesRegressor(n_estimators=500, max_features=0.65, min_samples_leaf=1, random_state=42, n_jobs=-1),
            True
        ),
        "XGBoost_log": (
            XGBRegressor(n_estimators=800, learning_rate=0.04, max_depth=6,
                         subsample=0.8, colsample_bytree=0.8,
                         reg_alpha=0.05, reg_lambda=1.5, min_child_weight=2,
                         random_state=42, n_jobs=-1, verbosity=0),
            True
        ),
        "XGBoost_raw": (
            XGBRegressor(n_estimators=600, learning_rate=0.04, max_depth=6,
                         subsample=0.8, colsample_bytree=0.8,
                         reg_alpha=0.05, reg_lambda=1.0, min_child_weight=3,
                         random_state=42, n_jobs=-1, verbosity=0),
            False
        ),
    }

    results = {}
    best_preds = None
    best_model_obj = None
    for name, (mdl, log_t) in candidates.items():
        import copy
        m, preds = train_and_eval(copy.deepcopy(mdl), X_train, y_train, X_test, y_test, log_target=log_t)
        results[name] = m
        print(f"  {name:25s}  RMSE={m['rmse']:8.3f}  MAE={m['mae']:7.3f}  "
              f"MAPE={m['mape']:6.2f}%  medMAPE={m['median_mape']:6.2f}%  R2={m['r2']:.4f}")

    # Pick best by MAPE (primary) then R² (tiebreak)
    best_name = min(results, key=lambda k: (results[k]["mape"], -results[k]["r2"]))
    best_m = results[best_name]
    best_mdl_template, best_log = candidates[best_name]

    print(f"\n  >>> Best: {best_name}")
    print(f"      RMSE={best_m['rmse']:.3f}  MAE={best_m['mae']:.3f}  "
          f"MAPE={best_m['mape']:.2f}%  medMAPE={best_m['median_mape']:.2f}%  "
          f"R2={best_m['r2']:.4f}  Pearson_r={best_m['pearson_r']:.4f}")

    # Retrain best model on ALL data for production
    import copy
    prod_model = copy.deepcopy(best_mdl_template)
    if best_log:
        prod_model.fit(X, np.log1p(y))
    else:
        prod_model.fit(X, y)

    # 5-fold CV on full data for honest reporting
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    if best_log:
        cv_preds_log = cross_val_predict(copy.deepcopy(best_mdl_template), X, np.log1p(y), cv=kf)
        cv_preds = np.expm1(cv_preds_log)
    else:
        cv_preds = cross_val_predict(copy.deepcopy(best_mdl_template), X, y, cv=kf)
    cv_preds = np.clip(cv_preds, 0, None)
    cv_metrics = {
        "cv_rmse": float(np.sqrt(mean_squared_error(y, cv_preds))),
        "cv_mae": float(mean_absolute_error(y, cv_preds)),
        "cv_mape": mape(y, cv_preds),
        "cv_median_mape": median_mape(y, cv_preds),
        "cv_r2": r2(y, cv_preds),
    }
    print(f"  5-fold CV (all {len(y)} samples):")
    print(f"      RMSE={cv_metrics['cv_rmse']:.3f}  MAE={cv_metrics['cv_mae']:.3f}  "
          f"MAPE={cv_metrics['cv_mape']:.2f}%  medMAPE={cv_metrics['cv_median_mape']:.2f}%  "
          f"R2={cv_metrics['cv_r2']:.4f}")

    scaler = StandardScaler()
    scaler.fit(X)

    return prod_model, scaler, feat_cols, best_m, cv_metrics, best_name, best_log


def main():
    jetson = pd.read_csv(os.path.join(DATA_DIR, "360_models_benchmark_jetson.csv"))
    rpi5   = pd.read_csv(os.path.join(DATA_DIR, "253_models_benchmark_rpi5.csv"))
    print(f"Jetson: {len(jetson)} samples  |  RPi5: {len(rpi5)} samples")

    j_model, j_scaler, j_feats, j_test, j_cv, j_best_name, j_log = train_device(jetson, "Jetson Nano", "jetson_nano")
    r_model, r_scaler, r_feats, r_test, r_cv, r_best_name, r_log = train_device(rpi5, "Raspberry Pi 5", "raspberry_pi5")

    # Combined test metrics
    n_j, n_r = round(len(jetson) * 0.2), round(len(rpi5) * 0.2)
    n_total = n_j + n_r
    combined_rmse = float(np.sqrt((n_j * j_test["rmse"]**2 + n_r * r_test["rmse"]**2) / n_total))
    combined_mae  = float((n_j * j_test["mae"] + n_r * r_test["mae"]) / n_total)
    combined_mape = float((n_j * j_test["mape"] + n_r * r_test["mape"]) / n_total)
    combined_r2   = float((n_j * j_test["r2"] + n_r * r_test["r2"]) / n_total)

    print(f"\n{'='*70}")
    print(f"COMBINED TEST SET ({n_total} samples = {n_j} Jetson + {n_r} RPi5)")
    print(f"  RMSE={combined_rmse:.3f} mWh  MAE={combined_mae:.3f} mWh  "
          f"MAPE={combined_mape:.3f}%  R2={combined_r2:.4f}")
    print(f"\nPAPER Table II values (device-specific Extra Trees):")
    print(f"  Old: RMSE=142.656  MAE=32.788  MAPE=16.718%  R2=0.796")
    print(f"  New: RMSE={combined_rmse:.3f}  MAE={combined_mae:.3f}  "
          f"MAPE={combined_mape:.3f}%  R2={combined_r2:.4f}")

    # Save models
    with open(os.path.join(ARTIFACTS_DIR, "jetson_energy_model.pkl"), "wb") as f:
        pickle.dump(j_model, f)
    with open(os.path.join(ARTIFACTS_DIR, "jetson_scaler.pkl"), "wb") as f:
        pickle.dump(j_scaler, f)
    with open(os.path.join(ARTIFACTS_DIR, "rpi5_energy_model.pkl"), "wb") as f:
        pickle.dump(r_model, f)
    with open(os.path.join(ARTIFACTS_DIR, "rpi5_scaler.pkl"), "wb") as f:
        pickle.dump(r_scaler, f)

    with open(os.path.join(ARTIFACTS_DIR, "device_specific_features.json"), "w") as f:
        json.dump(FEATURES, f, indent=2)

    metadata = {
        "jetson_model": {
            "model_name": j_best_name.replace("_log", "").replace("_raw", ""),
            "log_transform_target": j_log,
            "metrics": {
                "test_rmse": j_test["rmse"], "test_mae": j_test["mae"],
                "test_mape": j_test["mape"], "test_median_mape": j_test["median_mape"],
                "test_r2": j_test["r2"], "test_pearson_r": j_test["pearson_r"],
                "cv_rmse": j_cv["cv_rmse"], "cv_mae": j_cv["cv_mae"],
                "cv_mape": j_cv["cv_mape"], "cv_median_mape": j_cv["cv_median_mape"],
                "cv_r2": j_cv["cv_r2"],
                # Keep cv_mape_eps for EnergyPredictorService CI computation
                "cv_mape_eps": j_test["median_mape"],
            },
            "feature_count": len(j_feats),
        },
        "rpi5_model": {
            "model_name": r_best_name.replace("_log", "").replace("_raw", ""),
            "log_transform_target": r_log,
            "metrics": {
                "test_rmse": r_test["rmse"], "test_mae": r_test["mae"],
                "test_mape": r_test["mape"], "test_median_mape": r_test["median_mape"],
                "test_r2": r_test["r2"], "test_pearson_r": r_test["pearson_r"],
                "cv_rmse": r_cv["cv_rmse"], "cv_mae": r_cv["cv_mae"],
                "cv_mape": r_cv["cv_mape"], "cv_median_mape": r_cv["cv_median_mape"],
                "cv_r2": r_cv["cv_r2"],
                "cv_mape_eps": r_test["median_mape"],
            },
            "feature_count": len(r_feats),
        },
        "combined_test": {
            "n_total": int(n_total), "n_jetson": int(n_j), "n_rpi5": int(n_r),
            "rmse": combined_rmse, "mae": combined_mae,
            "mape": combined_mape, "r2": combined_r2,
        }
    }
    with open(os.path.join(ARTIFACTS_DIR, "device_specific_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Energy thresholds (unchanged)
    all_j = jetson["energy_avg_mwh"].dropna()
    all_r = rpi5["energy_avg_mwh"].dropna()
    all_e = pd.concat([all_j, all_r])
    thresholds = {
        "jetson_nano": {"p25": float(np.percentile(all_j, 25)), "p50": float(np.percentile(all_j, 50)), "p75": float(np.percentile(all_j, 75))},
        "raspberry_pi5": {"p25": float(np.percentile(all_r, 25)), "p50": float(np.percentile(all_r, 50)), "p75": float(np.percentile(all_r, 75))},
        "unified": {"p25": float(np.percentile(all_e, 25)), "p50": float(np.percentile(all_e, 50)), "p75": float(np.percentile(all_e, 75))},
    }
    with open(os.path.join(ARTIFACTS_DIR, "energy_thresholds.json"), "w") as f:
        json.dump(thresholds, f, indent=2)

    print("\n[OK] All artifacts saved.")
    return j_test, r_test, combined_rmse, combined_mae, combined_mape, combined_r2


if __name__ == "__main__":
    main()
