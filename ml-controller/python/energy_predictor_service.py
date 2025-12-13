import os
import pickle
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd


class EnergyPredictorService:
    """
    Lightweight wrapper to load the trained energy predictor and run inference.
    Expects artifacts generated under ml-controller/artifacts.
    """

    def __init__(self, artifacts_dir: str):
        self.artifacts_dir = artifacts_dir
        self.model = self._load_pickle("energy_predictor.pkl")
        self.scaler = self._load_pickle("energy_scaler.pkl")
        self.feature_names: List[str] = self._load_pickle("feature_names.pkl")
        self.model_info: Dict[str, Any] = self._load_pickle(
            "model_info.pkl", default={}
        )
        self.default_mape = abs(
            (
                (self.model_info.get("results") or {})
                .get(self.model_info.get("best_model"), {})
                .get("mape", 0.15)
            )
        )

    def _load_pickle(self, filename: str, default=None):
        path = os.path.join(self.artifacts_dir, filename)
        if not os.path.exists(path):
            return default
        with open(path, "rb") as f:
            return pickle.load(f)

    def _build_feature_row(self, payload: Dict[str, Any]) -> pd.DataFrame:
        """
        Build a single-row DataFrame in the exact feature order expected by the model.
        Required base fields: params_m, gflops, gmacs, size_mb, latency_avg_s, throughput_iter_per_s.
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

        derived = {
            "gflops_per_param": gflops / (params_m + 1e-6),
            "gmacs_per_mb": gmacs / (size_mb + 1e-6),
            "latency_throughput_ratio": latency_avg_s * throughput_iter_per_s,
            "compute_intensity": gflops * latency_avg_s,
            "model_complexity": params_m * gflops,
        }

        features = {
            "params_m": params_m,
            "gflops": gflops,
            "gmacs": gmacs,
            "size_mb": size_mb,
            "latency_avg_s": latency_avg_s,
            "throughput_iter_per_s": throughput_iter_per_s,
            **derived,
        }

        row = {name: features.get(name, np.nan) for name in self.feature_names}
        df_row = pd.DataFrame([row])
        df_row.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_row.fillna(df_row.median(numeric_only=True), inplace=True)
        return df_row

    def predict(self, payloads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Run batch prediction. Returns list of dicts with prediction and basic CI.
        """
        outputs = []
        for item in payloads:
            features_df = self._build_feature_row(item)
            scaled = self.scaler.transform(features_df)
            pred = float(self.model.predict(scaled)[0])

            # Simple CI approximation using validation MAPE
            rel_err = max(self.default_mape, 0.1)
            lower = max(pred * (1 - rel_err), 0.0)
            upper = pred * (1 + rel_err)

            outputs.append(
                {
                    "model_name": item.get("model") or item.get("name"),
                    "prediction_mwh": pred,
                    "ci_lower_mwh": lower,
                    "ci_upper_mwh": upper,
                    "features_used": {k: features_df.iloc[0][k] for k in self.feature_names},
                }
            )
        return outputs
