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
        
        # Get MAPE for confidence intervals
        jetson_metrics = self.metadata.get("jetson_model", {}).get("metrics", {})
        rpi5_metrics = self.metadata.get("rpi5_model", {}).get("metrics", {})
        self.jetson_mape = jetson_metrics.get("test_mape", 22.0) / 100  # 21.54% → 0.2154
        self.rpi5_mape = rpi5_metrics.get("loo_mape", 15.0) / 100  # 14.21% → 0.1421
        self.unified_mape = 0.50  # Conservative fallback MAPE

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
        with open(path, "r") as f:
            return json.load(f)

    def _build_feature_row(self, payload: Dict[str, Any]) -> pd.DataFrame:
        """
        Build a single-row DataFrame for device-specific features.
        
        Device-specific models use 9 features (NO device encoding):
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

        features = {
            "params_m": params_m,
            "gflops": gflops,
            "gmacs": gmacs,
            "size_mb": size_mb,
            "latency_avg_s": latency_avg_s,
            "throughput_iter_per_s": throughput_iter_per_s,
            **derived
        }

        # Build row with exact feature order from self.feature_names
        row = {name: features.get(name, np.nan) for name in self.feature_names}
        df_row = pd.DataFrame([row])
        
        # Handle inf/nan
        df_row.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_row.fillna(df_row.median(numeric_only=True), inplace=True)
        
        return df_row

    def _select_model_and_scaler(self, device_type: str) -> Tuple[Any, Any, float]:
        """
        Select appropriate model, scaler, and MAPE based on device type.
        
        Returns:
            (model, scaler, mape) tuple
        """
        device_lower = device_type.lower().strip()
        
        # Jetson Nano routing
        if any(keyword in device_lower for keyword in ["jetson", "nano"]):
            if self.jetson_model is not None and self.jetson_scaler is not None:
                return self.jetson_model, self.jetson_scaler, self.jetson_mape
        
        # Raspberry Pi 5 routing
        if any(keyword in device_lower for keyword in ["raspberry", "rpi", "pi"]):
            if self.rpi5_model is not None and self.rpi5_scaler is not None:
                return self.rpi5_model, self.rpi5_scaler, self.rpi5_mape
        
        # Fallback to unified model
        if self.unified_model is not None and self.unified_scaler is not None:
            return self.unified_model, self.unified_scaler, self.unified_mape
        
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
                model, scaler, mape = self._select_model_and_scaler(device_type)
                
                # Build features
                features_df = self._build_feature_row(item)
                
                # Scale and predict
                scaled = scaler.transform(features_df)
                pred = float(model.predict(scaled)[0])
                
                # Confidence interval using device-specific MAPE
                lower = max(pred * (1 - mape), 0.0)
                upper = pred * (1 + mape)
                
                outputs.append({
                    "model_name": item.get("model") or item.get("name"),
                    "device_type": device_type,
                    "prediction_mwh": pred,
                    "ci_lower_mwh": lower,
                    "ci_upper_mwh": upper,
                    "model_used": type(model).__name__,
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
