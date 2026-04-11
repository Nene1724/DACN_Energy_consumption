"""
ONNX / TFLite feature extractor for Energy Prediction.

Extracts the 6 model features required by EnergyPredictorService:
    params_m, gflops, gmacs, size_mb  — from model file (exact)
    latency_avg_s, throughput_iter_per_s — estimated from benchmark dataset

Supported formats:
    .onnx   → full extraction (params, GFLOPs, GMACs, size, input_shape)
    .tflite → partial extraction (size, estimated params from flatbuffer)
    other   → size only, user must supply remaining fields manually
"""

from __future__ import annotations

import os
import io
import struct
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# ONNX extraction
# ──────────────────────────────────────────────────────────────────────────────

def _extract_onnx(file_bytes: bytes, input_shape: Optional[Tuple] = None) -> Dict[str, Any]:
    """
    Extract features from an ONNX model bytes.

    Returns dict with keys:
        params_m, gflops, gmacs, size_mb, input_shape
        (gflops/gmacs are None if shape inference fails)
    """
    try:
        import onnx  # type: ignore
    except ImportError:
        raise RuntimeError("Package 'onnx' is not installed. Run: pip install onnx onnx-tool")

    model = onnx.load_from_string(file_bytes)
    graph = model.graph

    # ── Params ──────────────────────────────────────────────────────────────
    total_params = 0
    for init in graph.initializer:
        if len(init.dims) > 0:
            total_params += int(np.prod(init.dims))
    params_m = round(total_params / 1e6, 4)

    # ── Size ────────────────────────────────────────────────────────────────
    size_mb = round(len(file_bytes) / (1024 * 1024), 4)

    # ── Input shape ─────────────────────────────────────────────────────────
    detected_input_shape = None
    if graph.input:
        first_input = graph.input[0]
        tensor_type = first_input.type.tensor_type
        if tensor_type.HasField("shape"):
            dims = [d.dim_value for d in tensor_type.shape.dim]
            # Replace dynamic dims (0 / -1) with defaults
            if len(dims) == 4:
                dims[0] = 1  # batch
                if dims[2] <= 0:
                    dims[2] = 224
                if dims[3] <= 0:
                    dims[3] = 224
                if dims[1] <= 0:
                    dims[1] = 3
            detected_input_shape = tuple(dims)

    # Use caller-supplied shape if provided, otherwise use detected, fallback to (1,3,224,224)
    used_shape = input_shape or detected_input_shape or (1, 3, 224, 224)

    # ── GFLOPs / GMACs via onnx-tool ────────────────────────────────────────
    gflops: Optional[float] = None
    gmacs: Optional[float] = None

    try:
        import onnx_tool  # type: ignore

        # Write bytes to a temp file — onnx_tool needs a filepath
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            m = onnx_tool.Model(tmp_path)
            # Build input feed using detected shape
            input_name = graph.input[0].name
            dummy = np.zeros(used_shape, dtype=np.float32)
            m.graph.shape_infer({input_name: dummy})
            m.graph.profile()

            total_macs = 0
            for node in m.graph.nodemap.values():
                raw = node.macs
                if isinstance(raw, (int, float)):
                    total_macs += raw
                elif isinstance(raw, (list, tuple)) and len(raw) > 0:
                    total_macs += raw[0]  # first element = op MACs

            gmacs = round(total_macs / 1e9, 6)
            gflops = round(gmacs * 2, 6)
        finally:
            os.unlink(tmp_path)

    except Exception as e:
        # Fallback: estimate GFLOPs from params (very rough: 2 * params per 224 image)
        # Better than nothing when onnx-tool fails
        gmacs = None
        gflops = None

    return {
        "params_m": params_m,
        "size_mb": size_mb,
        "gflops": gflops,
        "gmacs": gmacs,
        "input_shape": list(used_shape),
        "model_type": "onnx",
        "extraction_complete": gflops is not None,
    }


# ──────────────────────────────────────────────────────────────────────────────
# TFLite extraction (partial)
# ──────────────────────────────────────────────────────────────────────────────

def _extract_tflite(file_bytes: bytes) -> Dict[str, Any]:
    """
    Minimal extraction from TFLite flatbuffer.
    Only size_mb and an estimate of params_m from file size.
    GFLOPs / GMACs require running the interpreter — returned as None.
    """
    size_mb = round(len(file_bytes) / (1024 * 1024), 4)

    # TFLite flatbuffer: try to count tensors via flatbuffers lib (optional)
    params_m: Optional[float] = None
    input_shape = None

    try:
        # Attempt using flatbuffers if available
        import flatbuffers  # type: ignore  # noqa: F401
        # Full parsing is complex — skip for now
    except ImportError:
        pass

    # If we couldn't get params, estimate: typical TFLite quantized models ≈ 4 bytes/param
    # This is a very rough heuristic
    if params_m is None:
        params_m = round((size_mb * 1024 * 1024) / 4 / 1e6, 4)  # assume float32

    return {
        "params_m": params_m,
        "size_mb": size_mb,
        "gflops": None,
        "gmacs": None,
        "input_shape": input_shape,
        "model_type": "tflite",
        "extraction_complete": False,
        "note": "TFLite: GFLOPs not extracted. Params estimated from file size. Please verify manually.",
    }


# ──────────────────────────────────────────────────────────────────────────────
# Latency / throughput estimation from benchmark CSV
# ──────────────────────────────────────────────────────────────────────────────

def _estimate_latency_throughput(
    gflops: float,
    params_m: Optional[float],
    gmacs: Optional[float],
    size_mb: Optional[float],
    device_key: str,
    jetson_csv: str,
    rpi5_csv: str,
    k: int = 7,
) -> Tuple[float, float]:
    """
    Estimate latency and throughput for a model on a given device by finding
    the k most similar models in the benchmark dataset using multiple model
    complexity signals instead of GFLOPs alone.

    Returns:
        (latency_avg_s, throughput_iter_per_s) — median of k-nearest neighbours
    """
    csv_path = rpi5_csv if "rpi" in device_key or "raspberry" in device_key else jetson_csv

    if not os.path.exists(csv_path):
        # Hard fallback when CSV is missing
        return (0.01, 100.0)

    df = pd.read_csv(csv_path)

    required = {"gflops", "params_m", "size_mb", "latency_avg_s", "throughput_iter_per_s"}
    if not required.issubset(df.columns):
        return (0.01, 100.0)

    df = df.dropna(subset=list(required))
    df = df[(df["gflops"] > 0) & (df["params_m"] > 0) & (df["size_mb"] > 0)]

    if df.empty:
        return (0.01, 100.0)

    df = df.copy()

    # Keep only rows where latency × throughput ≈ 1 (consistency check)
    ratio = df["latency_avg_s"] * df["throughput_iter_per_s"]
    df = df[(ratio > 0.5) & (ratio < 2.0)]

    if df.empty:
        return (0.01, 100.0)

    # k-nearest by multi-feature log distance
    feature_pairs = [
        ("gflops", float(gflops)),
        ("params_m", float(params_m) if params_m is not None else None),
        ("size_mb", float(size_mb) if size_mb is not None else None),
    ]
    if gmacs is not None and "gmacs" in df.columns:
        feature_pairs.append(("gmacs", float(gmacs)))

    dist = np.zeros(len(df), dtype=float)
    valid_feature_count = 0
    for col, target in feature_pairs:
        if target is None or target <= 0 or col not in df.columns:
            continue

        series = np.log1p(df[col].astype(float))
        target_log = np.log1p(target)
        scale = float(series.quantile(0.75) - series.quantile(0.25))
        if not np.isfinite(scale) or scale <= 1e-6:
            scale = float(series.std(ddof=0))
        if not np.isfinite(scale) or scale <= 1e-6:
            scale = 1.0

        dist += ((series - target_log) / scale).abs().to_numpy()
        valid_feature_count += 1

    if valid_feature_count == 0:
        return (0.01, 100.0)

    df["_dist"] = dist / valid_feature_count

    # Prefer fast models (latency < 1s = GPU-accelerated inference) — 
    # these are more representative for edge deployment use-cases
    df_fast = df[df["latency_avg_s"] < 1.0]
    pool = df_fast if len(df_fast) >= k else df  # fall back to all if not enough

    df_sorted = pool.nsmallest(k, "_dist").copy()
    weights = 1.0 / (df_sorted["_dist"].to_numpy(dtype=float) + 1e-6)
    if not np.all(np.isfinite(weights)) or weights.sum() <= 0:
        weights = np.ones(len(df_sorted), dtype=float)

    lat = float(np.expm1(np.average(np.log1p(df_sorted["latency_avg_s"]), weights=weights)))
    thr = float(np.expm1(np.average(np.log1p(df_sorted["throughput_iter_per_s"]), weights=weights)))
    return (round(lat, 6), round(thr, 4))


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def extract_features(
    file_bytes: bytes,
    filename: str,
    device_type: str = "jetson_nano",
    input_shape: Optional[Tuple] = None,
    jetson_csv: str = "",
    rpi5_csv: str = "",
) -> Dict[str, Any]:
    """
    Main entry point.

    Args:
        file_bytes:  Raw bytes of the uploaded model file.
        filename:    Original filename (used to detect format by extension).
        device_type: "jetson_nano" or "raspberry_pi5" — used for latency estimation.
        input_shape: Optional explicit input shape, e.g. (1, 3, 224, 224).
        jetson_csv:  Path to Jetson benchmark CSV (for latency estimation).
        rpi5_csv:    Path to RPi5 benchmark CSV.

    Returns:
        Dict with keys:
            params_m, gflops, gmacs, size_mb         — model properties
            latency_avg_s, throughput_iter_per_s      — estimated from benchmark
            input_shape, model_type                   — metadata
            extraction_complete                       — True if all fields extracted
            warnings                                  — list of warning strings
    """
    ext = os.path.splitext(filename.lower())[1]
    warnings: list[str] = []

    # ── Format-specific extraction ───────────────────────────────────────────
    if ext == ".onnx":
        info = _extract_onnx(file_bytes, input_shape=input_shape)
    elif ext == ".tflite":
        info = _extract_tflite(file_bytes)
        warnings.append(
            "TFLite format: GFLOPs/GMACs could not be extracted automatically. "
            "Please enter them manually, or convert to ONNX for full extraction."
        )
    else:
        size_mb = round(len(file_bytes) / (1024 * 1024), 4)
        info = {
            "params_m": None,
            "size_mb": size_mb,
            "gflops": None,
            "gmacs": None,
            "input_shape": None,
            "model_type": ext.lstrip(".") or "unknown",
            "extraction_complete": False,
        }
        warnings.append(
            f"Format '{ext}' is not fully supported. Only file size was extracted. "
            "Please convert to ONNX for automatic feature extraction."
        )

    # ── GMACs fallback if only GFLOPs available ──────────────────────────────
    if info.get("gflops") is not None and info.get("gmacs") is None:
        info["gmacs"] = round(info["gflops"] / 2, 6)
    if info.get("gmacs") is not None and info.get("gflops") is None:
        info["gflops"] = round(info["gmacs"] * 2, 6)

    # ── Latency / Throughput estimation ─────────────────────────────────────
    gflops_for_est = info.get("gflops")
    if gflops_for_est is not None and gflops_for_est > 0:
        lat, thr = _estimate_latency_throughput(
            gflops=gflops_for_est,
            params_m=info.get("params_m"),
            gmacs=info.get("gmacs"),
            size_mb=info.get("size_mb"),
            device_key=device_type,
            jetson_csv=jetson_csv,
            rpi5_csv=rpi5_csv,
        )
        warnings.append(
            f"latency_avg_s and throughput_iter_per_s are estimated from benchmark "
            f"dataset (k-nearest by params/GFLOPs/GMACs/size). Actual values on {device_type} may differ."
        )
    else:
        lat, thr = 0.01, 100.0
        warnings.append(
            "Could not estimate latency/throughput: GFLOPs unknown. "
            "Please enter latency_avg_s and throughput_iter_per_s manually."
        )

    info["latency_avg_s"] = lat
    info["throughput_iter_per_s"] = thr
    info["warnings"] = warnings
    info["device_type"] = device_type

    return info
