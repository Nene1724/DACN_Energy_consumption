import os
import pickle
import json
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd


class EnergyPredictorService:
    """
    Device-aware energy predictor service.
    
    Supports:
    - Device-specific models (Jetson Nano, Raspberry Pi 5) - PRODUCTION READY
    - Unified fallback model for unknown devices
    
    Expects artifacts in ml-controller/artifacts:
    - jetson_energy_model.pkl + jetson_scaler.pkl
    - rpi5_energy_model.pkl + rpi5_scaler.pkl
    - device_specific_features.json
    - device_specific_metadata.json
    """

    def __init__(self, artifacts_dir: str):
        self.artifacts_dir = artifacts_dir
        
        # Load device-specific models (PRODUCTION)
        self.jetson_model = self._load_pickle("jetson_energy_model.pkl")
        self.jetson_scaler = self._load_pickle("jetson_scaler.pkl")
        self.rpi5_model = self._load_pickle("rpi5_energy_model.pkl")
        self.rpi5_scaler = self._load_pickle("rpi5_scaler.pkl")
        
        # Load unified model (FALLBACK)
        self.unified_model = self._load_pickle("energy_predictor.pkl")
        self.unified_scaler = self._load_pickle("energy_scaler.pkl")
        
        # Load features
        self.feature_names = self._load_json("device_specific_features.json", default=[])
        if not self.feature_names:  # Fallback to old features
            self.feature_names = self._load_pickle("feature_names.pkl", default=[])
        
        # Load metadata
        self.metadata = self._load_json("device_specific_metadata.json", default={})

        # Confidence intervals + target transform configs from metadata
        self.unified_mape = 0.50  # Conservative fallback MAPE
        self.jetson_cfg = self._build_device_config(
            device_key="jetson_model",
            model=self.jetson_model,
            scaler=self.jetson_scaler,
            default_mape_pct=22.0
        )
        self.rpi5_cfg = self._build_device_config(
            device_key="rpi5_model",
            model=self.rpi5_model,
            scaler=self.rpi5_scaler,
            default_mape_pct=15.0
        )

    def _load_pickle(self, filename: str, default=None):
        path = os.path.join(self.artifacts_dir, filename)
        if not os.path.exists(path):
            return default
        with open(path, "rb") as f:
            return pickle.load(f)
    
    def _load_json(self, filename: str, default=None):
        path = os.path.join(self.artifacts_dir, filename)
        if not os.path.exists(path):
            return default
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _extract_mape_ratio(self, metrics: Dict[str, Any], default_mape_pct: float) -> float:
        """Extract mape ratio from various metadata formats."""
        for key in ("cv_mape_eps", "test_mape", "loo_mape", "mape", "cv_smape"):
            val = metrics.get(key)
            if val is None:
                continue
            try:
                val_f = float(val)
                if np.isfinite(val_f) and val_f >= 0:
                    return val_f / 100.0
            except (TypeError, ValueError):
                continue
        return float(default_mape_pct) / 100.0

    def _build_device_config(self, device_key: str, model: Any, scaler: Any, default_mape_pct: float) -> Dict[str, Any]:
        meta = self.metadata.get(device_key, {}) if isinstance(self.metadata, dict) else {}
        metrics = meta.get("metrics", {}) if isinstance(meta, dict) else {}
        return {
            "model": model,
            "scaler": scaler,
            "mape": self._extract_mape_ratio(metrics, default_mape_pct),
            "log_transform_target": bool(meta.get("log_transform_target", False)),
            "model_name": meta.get("model_name") or (type(model).__name__ if model is not None else "unknown"),
        }

    def _infer_batch_high_power(self, payload: Dict[str, Any], device_type: str) -> int:
        """Infer batch flag used by Jetson model if caller does not provide it."""
        device_lower = (device_type or "").lower()
        if "jetson" not in device_lower and "nano" not in device_lower:
            return 0

        explicit = payload.get("batch_high_power")
        if explicit is not None:
            try:
                return int(float(explicit) > 0)
            except (TypeError, ValueError):
                pass

        power_mode = str(payload.get("power_mode") or payload.get("nvpmodel_mode") or "").lower().strip()
        if "maxn" in power_mode:
            return 1
        if "5w" in power_mode:
            return 0

        return 0

    def _build_feature_row(self, payload: Dict[str, Any], device_type: str) -> pd.DataFrame:
        """
        Build a single-row DataFrame for device-specific features.
        
                Device-specific models use 12 features (NO device encoding):
                - params_m, gflops, gmacs, size_mb, latency_avg_s, throughput_iter_per_s
                - gflops_per_param, gmacs_per_mb, latency_throughput_ratio,
                    compute_intensity, model_complexity, computational_density
        """
        try:
            params_m = float(payload["params_m"])
            gflops = float(payload["gflops"])
            gmacs = float(payload["gmacs"])
            size_mb = float(payload["size_mb"])
            latency_avg_s = float(payload["latency_avg_s"])
            throughput_iter_per_s = float(payload["throughput_iter_per_s"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "Thiếu hoặc sai định dạng các field bắt buộc: "
                "params_m, gflops, gmacs, size_mb, latency_avg_s, throughput_iter_per_s"
            ) from exc

        # Compute derived features (same as notebook)
        derived = {
            "gflops_per_param": gflops / (params_m + 1e-6),
            "gmacs_per_mb": gmacs / (size_mb + 1e-6),
            "latency_throughput_ratio": latency_avg_s * throughput_iter_per_s,
            "compute_intensity": gflops * latency_avg_s,
            "model_complexity": params_m * gflops,
            "computational_density": gflops / (size_mb + 1e-6)
        }

        log_features = {
            "log_params_m": np.log1p(params_m),
            "log_gflops": np.log1p(gflops),
            "log_size_mb": np.log1p(size_mb),
            "log_gmacs": np.log1p(gmacs),
            "log_latency": np.log1p(latency_avg_s),
            "log_throughput": np.log1p(throughput_iter_per_s),
            "log_model_complexity": np.log1p(derived["model_complexity"]),
            "log_compute_intensity": np.log1p(derived["compute_intensity"]),
        }

        interactions = {
            "log_params_x_log_latency": log_features["log_params_m"] * log_features["log_latency"],
            "log_gflops_x_log_latency": log_features["log_gflops"] * log_features["log_latency"],
        }

        batch_high_power = self._infer_batch_high_power(payload, device_type)

        features = {
            "params_m": params_m,
            "gflops": gflops,
            "gmacs": gmacs,
            "size_mb": size_mb,
            "latency_avg_s": latency_avg_s,
            "throughput_iter_per_s": throughput_iter_per_s,
            **derived,
            **log_features,
            **interactions,
            "batch_high_power": batch_high_power,
        }

        # Build row with exact feature order from self.feature_names
        row = {name: features.get(name, np.nan) for name in self.feature_names}
        df_row = pd.DataFrame([row])
        
        # Handle inf/nan - fill with 0 instead of median (since we only have 1 row)
        df_row.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_row.fillna(0, inplace=True)
        
        return df_row

    def _predict_value(self, model: Any, scaler: Any, features_df: pd.DataFrame, use_log_target: bool) -> float:
        """Predict with proper preprocessing, avoiding double scaling for Pipeline models."""
        if model is None:
            raise ValueError("Model chưa được load")

        if hasattr(model, "named_steps"):
            raw_pred = float(model.predict(features_df)[0])
        elif scaler is not None:
            scaled = scaler.transform(features_df)
            raw_pred = float(model.predict(scaled)[0])
        else:
            raw_pred = float(model.predict(features_df)[0])

        if use_log_target:
            return float(np.expm1(raw_pred))
        return raw_pred

    def _select_model_config(self, device_type: str) -> Dict[str, Any]:
        """
        Select appropriate model, scaler, and MAPE based on device type.
        
        Returns:
            Config dict with model/scaler/mape/log_transform_target/model_name
        """
        device_lower = device_type.lower().strip()
        
        # Jetson Nano routing
        if any(keyword in device_lower for keyword in ["jetson", "nano"]):
            if self.jetson_cfg["model"] is not None:
                return self.jetson_cfg
        
        # Raspberry Pi 5 routing
        if any(keyword in device_lower for keyword in ["raspberry", "rpi", "pi"]):
            if self.rpi5_cfg["model"] is not None:
                return self.rpi5_cfg
        
        # Fallback to unified model
        if self.unified_model is not None:
            return {
                "model": self.unified_model,
                "scaler": self.unified_scaler,
                "mape": self.unified_mape,
                "log_transform_target": False,
                "model_name": type(self.unified_model).__name__,
            }
        
        # No model available
        raise ValueError(f"No model available for device type: {device_type}")
    
    def predict(self, payloads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Run batch prediction with device-aware routing.
        
        Each payload must include:
        - device_type or device: str (e.g., "jetson_nano", "raspberry_pi5")
        - params_m, gflops, gmacs, size_mb, latency_avg_s, throughput_iter_per_s: float
        
        Returns:
            List of predictions with confidence intervals
        """
        outputs = []
        for item in payloads:
            # Get device type
            device_type = item.get("device_type") or item.get("device") or "unknown"
            
            try:
                # Select device-specific model
                cfg = self._select_model_config(device_type)
                model = cfg["model"]
                scaler = cfg["scaler"]
                mape = cfg["mape"]
                log_transform_target = cfg["log_transform_target"]
                
                # Build features
                features_df = self._build_feature_row(item, device_type)
                
                # Predict with correct preprocessing + inverse transform for log-target models
                pred = self._predict_value(
                    model=model,
                    scaler=scaler,
                    features_df=features_df,
                    use_log_target=log_transform_target,
                )
                
                # Confidence interval using device-specific MAPE
                lower = max(pred * (1 - mape), 0.0)
                upper = pred * (1 + mape)
                
                outputs.append({
                    "model_name": item.get("model") or item.get("name"),
                    "device_type": device_type,
                    "prediction_mwh": pred,
                    "ci_lower_mwh": lower,
                    "ci_upper_mwh": upper,
                    "model_used": cfg["model_name"],
                    "mape_pct": mape * 100,
                    "features_used": {k: features_df.iloc[0][k] for k in self.feature_names}
                })
            except Exception as e:
                outputs.append({
                    "model_name": item.get("model") or item.get("name"),
                    "device_type": device_type,
                    "error": str(e),
                    "prediction_mwh": None
                })
        
        return outputs
