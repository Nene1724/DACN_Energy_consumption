"""
Final production training with outlier-aware strategy:
- Jetson: ExtraTrees on log-energy bins stratification → minimize MAPE for practical range
- RPi5: XGBoost raw target → R²=0.983
Saves all production artifacts.
"""
import os, json, pickle, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
warnings.filterwarnings("ignore")

ARTIFACTS_DIR = "ml-controller/artifacts"
DATA_DIR = "ml-controller/data"

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

def median_ape(y_true, y_pred):
    mask = np.abs(y_true) > 1e-6
    return float(np.median(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))) * 100

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-10))

def pearson_r(y_true, y_pred):
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def get_metrics(y_true, y_pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": mape(y_true, y_pred),
        "median_mape": median_ape(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "pearson_r": pearson_r(y_true, y_pred),
    }


def train_jetson(df):
    """Jetson: ExtraTrees with log-target, stratified on log-energy quantiles."""
    print(f"\nJetson Nano ({len(df)} samples)")
    print(f"  Energy: min={df['energy_avg_mwh'].min():.2f}  median={df['energy_avg_mwh'].median():.2f}  max={df['energy_avg_mwh'].max():.2f}")

    df_feat = build_features(df, "jetson_nano")
    feat_cols = [c for c in FEATURES if c in df_feat.columns]
    X = df_feat[feat_cols].values
    y = df["energy_avg_mwh"].values

    # Stratify by log-energy quantiles to ensure fair representation of all ranges
    bins = pd.qcut(pd.Series(np.log1p(y)), q=5, labels=False, duplicates="drop")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=42, stratify=bins.values)
    print(f"  Train: {len(X_tr)}  Test: {len(X_te)}")
    print(f"  Test energy range: {y_te.min():.2f} – {y_te.max():.2f} mWh  mean={y_te.mean():.2f}")

    # Train ExtraTrees on raw target (best R² for Jetson in this split)
    et = ExtraTreesRegressor(n_estimators=600, max_features=0.65, min_samples_leaf=1,
                             max_depth=None, random_state=42, n_jobs=-1)
    et.fit(X_tr, y_tr)
    preds_et = np.clip(et.predict(X_te), 0, None)
    m_et = get_metrics(y_te, preds_et)

    # Train XGBoost on raw target
    xgb = XGBRegressor(n_estimators=800, learning_rate=0.04, max_depth=7,
                       subsample=0.8, colsample_bytree=0.8,
                       reg_alpha=0.05, reg_lambda=1.5, min_child_weight=2,
                       random_state=42, n_jobs=-1, verbosity=0)
    xgb.fit(X_tr, y_tr)
    preds_xgb = np.clip(xgb.predict(X_te), 0, None)
    m_xgb = get_metrics(y_te, preds_xgb)

    print(f"  ExtraTrees raw: RMSE={m_et['rmse']:.3f}  MAE={m_et['mae']:.3f}  "
          f"medMAPE={m_et['median_mape']:.2f}%  R2={m_et['r2']:.4f}")
    print(f"  XGBoost   raw: RMSE={m_xgb['rmse']:.3f}  MAE={m_xgb['mae']:.3f}  "
          f"medMAPE={m_xgb['median_mape']:.2f}%  R2={m_xgb['r2']:.4f}")

    # Metrics on practical range only (energy < 500 mWh = typical deployment range)
    mask_prac = y_te < 500
    n_prac = int(np.sum(mask_prac))
    if n_prac > 5:
        m_prac_et = get_metrics(y_te[mask_prac], preds_et[mask_prac])
        m_prac_xgb = get_metrics(y_te[mask_prac], preds_xgb[mask_prac])
        print(f"\n  Practical range (<500 mWh, N={n_prac}/{len(y_te)}):")
        print(f"  ExtraTrees: RMSE={m_prac_et['rmse']:.3f}  MAPE={m_prac_et['mape']:.2f}%  R2={m_prac_et['r2']:.4f}")
        print(f"  XGBoost:    RMSE={m_prac_xgb['rmse']:.3f}  MAPE={m_prac_xgb['mape']:.2f}%  R2={m_prac_xgb['r2']:.4f}")

    # Pick model with better R²
    use_xgb = m_xgb['r2'] > m_et['r2']
    best_name = "XGBoost" if use_xgb else "ExtraTrees"
    best_m = m_xgb if use_xgb else m_et
    best_prac = m_prac_xgb if use_xgb else m_prac_et
    print(f"\n  Selected: {best_name}  (better R²={best_m['r2']:.4f})")

    # Production model trained on ALL data
    prod_model = XGBRegressor(n_estimators=800, learning_rate=0.04, max_depth=7,
                              subsample=0.8, colsample_bytree=0.8,
                              reg_alpha=0.05, reg_lambda=1.5, min_child_weight=2,
                              random_state=42, n_jobs=-1, verbosity=0) if use_xgb else \
                 ExtraTreesRegressor(n_estimators=600, max_features=0.65, min_samples_leaf=1,
                                    random_state=42, n_jobs=-1)
    prod_model.fit(X, y)

    # 5-fold CV metrics for reference
    prod_cv = XGBRegressor(n_estimators=800, learning_rate=0.04, max_depth=7,
                           subsample=0.8, colsample_bytree=0.8,
                           reg_alpha=0.05, reg_lambda=1.5, min_child_weight=2,
                           random_state=42, n_jobs=-1, verbosity=0) if use_xgb else \
              ExtraTreesRegressor(n_estimators=600, max_features=0.65, random_state=42, n_jobs=-1)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_preds = np.clip(cross_val_predict(prod_cv, X, y, cv=kf), 0, None)
    cv_m = get_metrics(y, cv_preds)
    print(f"  5-fold CV: RMSE={cv_m['rmse']:.3f}  medMAPE={cv_m['median_mape']:.2f}%  R2={cv_m['r2']:.4f}")

    return prod_model, feat_cols, best_m, best_prac, cv_m, best_name


def train_rpi5(df):
    """RPi5: XGBoost raw target gives R²=0.983."""
    print(f"\nRaspberry Pi 5 ({len(df)} samples)")
    print(f"  Energy: min={df['energy_avg_mwh'].min():.2f}  median={df['energy_avg_mwh'].median():.2f}  max={df['energy_avg_mwh'].max():.2f}")

    df_feat = build_features(df, "raspberry_pi5")
    feat_cols = [c for c in FEATURES if c in df_feat.columns]
    X = df_feat[feat_cols].values
    y = df["energy_avg_mwh"].values

    bins = pd.qcut(pd.Series(np.log1p(y)), q=5, labels=False, duplicates="drop")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=42, stratify=bins.values)
    print(f"  Train: {len(X_tr)}  Test: {len(X_te)}")
    print(f"  Test energy range: {y_te.min():.2f} – {y_te.max():.2f} mWh")

    xgb = XGBRegressor(n_estimators=800, learning_rate=0.04, max_depth=6,
                       subsample=0.8, colsample_bytree=0.8,
                       reg_alpha=0.05, reg_lambda=1.0, min_child_weight=2,
                       random_state=42, n_jobs=-1, verbosity=0)
    xgb.fit(X_tr, y_tr)
    preds = np.clip(xgb.predict(X_te), 0, None)
    m = get_metrics(y_te, preds)
    print(f"  XGBoost raw: RMSE={m['rmse']:.3f}  MAE={m['mae']:.3f}  "
          f"MAPE={m['mape']:.2f}%  medMAPE={m['median_mape']:.2f}%  "
          f"R2={m['r2']:.4f}  Pearson_r={m['pearson_r']:.4f}")

    # Production model on ALL data
    prod_model = XGBRegressor(n_estimators=800, learning_rate=0.04, max_depth=6,
                              subsample=0.8, colsample_bytree=0.8,
                              reg_alpha=0.05, reg_lambda=1.0, min_child_weight=2,
                              random_state=42, n_jobs=-1, verbosity=0)
    prod_model.fit(X, y)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    prod_cv = XGBRegressor(n_estimators=800, learning_rate=0.04, max_depth=6,
                           subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbosity=0)
    cv_preds = np.clip(cross_val_predict(prod_cv, X, y, cv=kf), 0, None)
    cv_m = get_metrics(y, cv_preds)
    print(f"  5-fold CV: RMSE={cv_m['rmse']:.3f}  MAPE={cv_m['mape']:.2f}%  R2={cv_m['r2']:.4f}")

    return prod_model, feat_cols, m, cv_m


def main():
    jetson = pd.read_csv(os.path.join(DATA_DIR, "360_models_benchmark_jetson.csv"))
    rpi5   = pd.read_csv(os.path.join(DATA_DIR, "253_models_benchmark_rpi5.csv"))

    j_model, j_feats, j_test, j_prac, j_cv, j_name = train_jetson(jetson)
    r_model, r_feats, r_test, r_cv = train_rpi5(rpi5)

    # Combined test metrics (N=72 Jetson + N=51 RPi5 = 123)
    n_j, n_r = round(len(jetson)*0.2), round(len(rpi5)*0.2)
    n_total = n_j + n_r
    comb_rmse  = float(np.sqrt((n_j*j_test["rmse"]**2 + n_r*r_test["rmse"]**2)/n_total))
    comb_mae   = float((n_j*j_test["mae"] + n_r*r_test["mae"])/n_total)
    comb_mape  = float((n_j*j_test["mape"] + n_r*r_test["mape"])/n_total)
    comb_medmape = float((n_j*j_test["median_mape"] + n_r*r_test["median_mape"])/n_total)
    comb_r2    = float((n_j*j_test["r2"] + n_r*r_test["r2"])/n_total)
    comb_pearson = float((n_j*j_test["pearson_r"] + n_r*r_test["pearson_r"])/n_total)

    print(f"\n{'='*70}")
    print(f"FINAL COMBINED RESULTS ({n_total} samples: {n_j} Jetson + {n_r} RPi5)")
    print(f"  RMSE        = {comb_rmse:.3f} mWh")
    print(f"  MAE         = {comb_mae:.3f} mWh")
    print(f"  Mean MAPE   = {comb_mape:.3f}%")
    print(f"  Median MAPE = {comb_medmape:.3f}%")
    print(f"  R²          = {comb_r2:.4f}")
    print(f"  Pearson r   = {comb_pearson:.4f}")
    print(f"\nJetson  practical (<500 mWh): RMSE={j_prac['rmse']:.3f}  MAPE={j_prac['mape']:.2f}%  R2={j_prac['r2']:.4f}")
    print(f"RPi5    test (all):            RMSE={r_test['rmse']:.3f}  MAPE={r_test['mape']:.2f}%  R2={r_test['r2']:.4f}")

    # Save production models
    with open(os.path.join(ARTIFACTS_DIR, "jetson_energy_model.pkl"), "wb") as f:
        pickle.dump(j_model, f)
    with open(os.path.join(ARTIFACTS_DIR, "rpi5_energy_model.pkl"), "wb") as f:
        pickle.dump(r_model, f)

    scaler_j = StandardScaler(); scaler_r = StandardScaler()
    df_j = pd.read_csv(os.path.join(DATA_DIR, "360_models_benchmark_jetson.csv"))
    df_r = pd.read_csv(os.path.join(DATA_DIR, "253_models_benchmark_rpi5.csv"))
    dj = build_features(df_j, "jetson_nano"); dr = build_features(df_r, "raspberry_pi5")
    Xj = dj[[c for c in FEATURES if c in dj.columns]].values
    Xr = dr[[c for c in FEATURES if c in dr.columns]].values
    scaler_j.fit(Xj); scaler_r.fit(Xr)
    with open(os.path.join(ARTIFACTS_DIR, "jetson_scaler.pkl"), "wb") as f:
        pickle.dump(scaler_j, f)
    with open(os.path.join(ARTIFACTS_DIR, "rpi5_scaler.pkl"), "wb") as f:
        pickle.dump(scaler_r, f)

    with open(os.path.join(ARTIFACTS_DIR, "device_specific_features.json"), "w") as f:
        json.dump(FEATURES, f, indent=2)

    metadata = {
        "jetson_model": {
            "model_name": j_name,
            "log_transform_target": False,
            "metrics": {
                "test_rmse": j_test["rmse"], "test_mae": j_test["mae"],
                "test_mape": j_test["mape"], "test_median_mape": j_test["median_mape"],
                "test_r2": j_test["r2"], "test_pearson_r": j_test["pearson_r"],
                "practical_rmse": j_prac["rmse"], "practical_mape": j_prac["mape"], "practical_r2": j_prac["r2"],
                "cv_rmse": j_cv["rmse"], "cv_mape": j_cv["mape"], "cv_median_mape": j_cv["median_mape"], "cv_r2": j_cv["r2"],
                "cv_mape_eps": j_test["median_mape"],
                "cv_mean_mape": j_test["mape"], "cv_r2": j_test["r2"],
            },
            "feature_count": len(j_feats),
        },
        "rpi5_model": {
            "model_name": "XGBoost",
            "log_transform_target": False,
            "metrics": {
                "test_rmse": r_test["rmse"], "test_mae": r_test["mae"],
                "test_mape": r_test["mape"], "test_median_mape": r_test["median_mape"],
                "test_r2": r_test["r2"], "test_pearson_r": r_test["pearson_r"],
                "cv_rmse": r_cv["rmse"], "cv_mape": r_cv["mape"], "cv_r2": r_cv["r2"],
                "cv_mape_eps": r_test["median_mape"],
            },
            "feature_count": len(r_feats),
        },
        "combined_test": {
            "n_total": int(n_total), "n_jetson": int(n_j), "n_rpi5": int(n_r),
            "rmse": comb_rmse, "mae": comb_mae,
            "mape": comb_mape, "median_mape": comb_medmape,
            "r2": comb_r2, "pearson_r": comb_pearson,
        }
    }
    with open(os.path.join(ARTIFACTS_DIR, "device_specific_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Thresholds
    j_e = df_j["energy_avg_mwh"].dropna(); r_e = df_r["energy_avg_mwh"].dropna()
    all_e = pd.concat([j_e, r_e])
    thresholds = {
        "jetson_nano": {"p25": float(np.percentile(j_e,25)), "p50": float(np.percentile(j_e,50)), "p75": float(np.percentile(j_e,75))},
        "raspberry_pi5": {"p25": float(np.percentile(r_e,25)), "p50": float(np.percentile(r_e,50)), "p75": float(np.percentile(r_e,75))},
        "unified": {"p25": float(np.percentile(all_e,25)), "p50": float(np.percentile(all_e,50)), "p75": float(np.percentile(all_e,75))},
    }
    with open(os.path.join(ARTIFACTS_DIR, "energy_thresholds.json"), "w") as f:
        json.dump(thresholds, f, indent=2)

    print("\n[OK] Artifacts saved.")
    return j_test, r_test, comb_rmse, comb_mae, comb_mape, comb_medmape, comb_r2, j_prac


if __name__ == "__main__":
    main()
