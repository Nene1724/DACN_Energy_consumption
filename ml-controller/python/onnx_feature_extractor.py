"""
ONNX / TFLite feature extractor for energy prediction.

Extracts the model features required by EnergyPredictorService:
    params_m, gflops, gmacs, size_mb      from the model artifact
    latency_avg_s, throughput_iter_per_s  estimated from benchmark CSVs

Supported formats:
    .onnx   -> full extraction using onnx + onnx-tool when available
    .tflite -> flatbuffer-based extraction for params, GMACs and GFLOPs
    other   -> size only, user must provide the remaining fields manually
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


def _shape_to_list(shape_like: Any) -> list[int]:
    """Convert numpy / flatbuffer shape containers to a clean Python list."""
    if shape_like is None:
        return []
    if hasattr(shape_like, "tolist"):
        raw = shape_like.tolist()
    elif isinstance(shape_like, (list, tuple)):
        raw = list(shape_like)
    else:
        raw = [shape_like]
    out: list[int] = []
    for value in raw:
        try:
            out.append(int(value))
        except (TypeError, ValueError):
            continue
    return out


def _num_elements(shape: Any) -> int:
    dims = _shape_to_list(shape)
    if not dims:
        return 0
    total = 1
    for dim in dims:
        if dim <= 0:
            return 0
        total *= int(dim)
    return int(total)


def _extract_onnx(file_bytes: bytes, input_shape: Optional[Tuple] = None) -> Dict[str, Any]:
    """
    Extract features from ONNX bytes.

    Returns:
        params_m, gflops, gmacs, size_mb, input_shape
    """
    try:
        import onnx  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Package 'onnx' is not installed. Run: pip install onnx onnx-tool") from exc

    model = onnx.load_from_string(file_bytes)
    graph = model.graph

    total_params = 0
    for init in graph.initializer:
        if len(init.dims) > 0:
            total_params += int(np.prod(init.dims))
    params_m = round(total_params / 1e6, 6)
    size_mb = round(len(file_bytes) / (1024 * 1024), 6)

    detected_input_shape: Optional[Tuple[int, ...]] = None
    if graph.input:
        first_input = graph.input[0]
        tensor_type = first_input.type.tensor_type
        if tensor_type.HasField("shape"):
            dims = [int(d.dim_value) for d in tensor_type.shape.dim]
            if len(dims) == 4:
                dims[0] = 1
                dims[1] = dims[1] if dims[1] > 0 else 3
                dims[2] = dims[2] if dims[2] > 0 else 224
                dims[3] = dims[3] if dims[3] > 0 else 224
            detected_input_shape = tuple(dims)

    used_shape = tuple(input_shape or detected_input_shape or (1, 3, 224, 224))

    gflops: Optional[float] = None
    gmacs: Optional[float] = None
    try:
        import tempfile
        import onnx_tool  # type: ignore

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            model_tool = onnx_tool.Model(tmp_path)
            input_name = graph.input[0].name
            dummy = np.zeros(used_shape, dtype=np.float32)
            model_tool.graph.shape_infer({input_name: dummy})
            model_tool.graph.profile()

            total_macs = 0.0
            for node in model_tool.graph.nodemap.values():
                raw = node.macs
                if isinstance(raw, (int, float)):
                    total_macs += float(raw)
                elif isinstance(raw, (list, tuple)) and raw:
                    total_macs += float(raw[0])

            gmacs = round(total_macs / 1e9, 6)
            gflops = round(gmacs * 2, 6)
        finally:
            os.unlink(tmp_path)
    except Exception:
        gflops = None
        gmacs = None

    return {
        "params_m": params_m,
        "size_mb": size_mb,
        "gflops": gflops,
        "gmacs": gmacs,
        "input_shape": list(used_shape),
        "model_type": "onnx",
        "extraction_complete": gflops is not None,
    }


def _tflite_tensor_shape(tensor: Any) -> list[int]:
    shape = _shape_to_list(getattr(tensor, "ShapeAsNumpy", lambda: None)())
    if shape:
        return shape
    shape_sig_fn = getattr(tensor, "ShapeSignatureAsNumpy", None)
    if callable(shape_sig_fn):
        return _shape_to_list(shape_sig_fn())
    return []


def _tflite_bytes_per_type(tensor_type: int) -> Optional[int]:
    # See TensorType enum in the generated tflite schema.
    size_map = {
        0: 4,   # FLOAT32
        1: 2,   # FLOAT16
        2: 4,   # INT32
        3: 1,   # UINT8
        4: 8,   # INT64
        6: 1,   # BOOL
        7: 2,   # INT16
        9: 8,   # COMPLEX64 stores 2x float32, but rarely appears as weight
        10: 8,  # INT8? not used here, kept conservative
        11: 8,  # FLOAT64
        15: 1,  # INT8
        16: 2,  # FLOAT16 alternative enum value in newer schemas
        17: 8,  # FLOAT64 alternative enum value
        18: 2,  # UINT16
        19: 4,  # COMPLEX128 not expected, kept approximate
        20: 4,  # UINT32
        21: 8,  # UINT64
        22: 8,  # RESOURCE / VARIANT unknown, not used for params
    }
    return size_map.get(int(tensor_type))


def _tflite_buffer_nbytes(buffer_obj: Any) -> int:
    if buffer_obj is None:
        return 0
    data = getattr(buffer_obj, "DataAsNumpy", lambda: None)()
    if data is None:
        return 0
    if hasattr(data, "size"):
        return int(data.size)
    try:
        return int(len(data))
    except TypeError:
        try:
            return 1 if int(data) >= 0 else 0
        except (TypeError, ValueError):
            return 0


def _tflite_builtin_name(tflite_mod: Any, builtin_code: int) -> str:
    mapping = getattr(_tflite_builtin_name, "_mapping", None)
    if mapping is None:
        mapping = {
            int(getattr(tflite_mod.BuiltinOperator, name)): name
            for name in dir(tflite_mod.BuiltinOperator)
            if name.isupper()
        }
        setattr(_tflite_builtin_name, "_mapping", mapping)
    return mapping.get(int(builtin_code), f"OP_{builtin_code}")


def _build_tflite_producer_map(subgraph: Any) -> dict[int, int]:
    producer_map: dict[int, int] = {}
    for op_idx in range(subgraph.OperatorsLength()):
        operator = subgraph.Operators(op_idx)
        for output_idx in range(operator.OutputsLength()):
            tensor_idx = int(operator.Outputs(output_idx))
            if tensor_idx >= 0:
                producer_map[tensor_idx] = op_idx
    return producer_map


def _resolve_tflite_const_tensor_idx(
    model: Any,
    subgraph: Any,
    producer_map: dict[int, int],
    tensor_idx: int,
    tflite_mod: Any,
    depth: int = 0,
) -> Optional[int]:
    if tensor_idx < 0 or depth > 8:
        return None

    tensor = subgraph.Tensors(tensor_idx)
    if int(tensor.Buffer()) > 0:
        return tensor_idx

    op_idx = producer_map.get(int(tensor_idx))
    if op_idx is None:
        return None

    operator = subgraph.Operators(op_idx)
    opcode = model.OperatorCodes(operator.OpcodeIndex())
    op_name = _tflite_builtin_name(tflite_mod, int(opcode.BuiltinCode()))
    passthrough_ops = {"DEQUANTIZE", "QUANTIZE", "CAST", "RESHAPE"}
    if op_name not in passthrough_ops or operator.InputsLength() < 1:
        return None

    parent_tensor_idx = int(operator.Inputs(0))
    if parent_tensor_idx < 0:
        return None

    return _resolve_tflite_const_tensor_idx(
        model=model,
        subgraph=subgraph,
        producer_map=producer_map,
        tensor_idx=parent_tensor_idx,
        tflite_mod=tflite_mod,
        depth=depth + 1,
    )


def _estimate_tflite_heavy_op_macs(op_name: str, input_shapes: list[list[int]], output_shapes: list[list[int]]) -> float:
    if not output_shapes or not output_shapes[0]:
        return 0.0

    output_shape = output_shapes[0]

    if op_name == "CONV_2D" and len(input_shapes) >= 2:
        weight_shape = input_shapes[1]
        if len(output_shape) == 4 and len(weight_shape) == 4:
            batch, out_h, out_w, out_c = output_shape
            _, kernel_h, kernel_w, in_c = weight_shape
            if min(batch, out_h, out_w, out_c, kernel_h, kernel_w, in_c) > 0:
                return float(batch * out_h * out_w * out_c * kernel_h * kernel_w * in_c)

    if op_name == "DEPTHWISE_CONV_2D" and len(input_shapes) >= 2:
        weight_shape = input_shapes[1]
        if len(output_shape) == 4 and len(weight_shape) == 4:
            batch, out_h, out_w, out_c = output_shape
            _, kernel_h, kernel_w, _ = weight_shape
            if min(batch, out_h, out_w, out_c, kernel_h, kernel_w) > 0:
                return float(batch * out_h * out_w * out_c * kernel_h * kernel_w)

    if op_name in {"FULLY_CONNECTED", "BATCH_MATMUL"} and len(input_shapes) >= 2:
        left_shape = input_shapes[0]
        right_shape = input_shapes[1]
        if len(left_shape) >= 2 and len(right_shape) >= 2:
            batch = int(np.prod(left_shape[:-1])) if len(left_shape) > 1 else 1
            inner_dim = left_shape[-1]
            output_width = right_shape[0] if op_name == "FULLY_CONNECTED" else right_shape[-1]
            if batch > 0 and inner_dim > 0 and output_width > 0:
                return float(batch * inner_dim * output_width)

    if op_name == "TRANSPOSE_CONV" and len(input_shapes) >= 2:
        weight_shape = input_shapes[1]
        if len(output_shape) == 4 and len(weight_shape) == 4:
            batch, out_h, out_w, out_c = output_shape
            _, kernel_h, kernel_w, in_c = weight_shape
            if min(batch, out_h, out_w, out_c, kernel_h, kernel_w, in_c) > 0:
                return float(batch * out_h * out_w * out_c * kernel_h * kernel_w * in_c)

    if op_name == "CONV_3D" and len(input_shapes) >= 2:
        weight_shape = input_shapes[1]
        if len(output_shape) == 5 and len(weight_shape) == 5:
            batch, out_d, out_h, out_w, out_c = output_shape
            _, kernel_d, kernel_h, kernel_w, in_c = weight_shape
            if min(batch, out_d, out_h, out_w, out_c, kernel_d, kernel_h, kernel_w, in_c) > 0:
                return float(batch * out_d * out_h * out_w * out_c * kernel_d * kernel_h * kernel_w * in_c)

    return 0.0


def _estimate_tflite_misc_ops(op_name: str, output_shapes: list[list[int]]) -> float:
    if not output_shapes or not output_shapes[0]:
        return 0.0

    output_elems = _num_elements(output_shapes[0])
    if output_elems <= 0:
        return 0.0

    weighted_ops = {
        "ADD": 1.0,
        "SUB": 1.0,
        "MUL": 1.0,
        "DIV": 1.0,
        "MAXIMUM": 1.0,
        "MINIMUM": 1.0,
        "MEAN": 1.0,
        "SUM": 1.0,
        "RESHAPE": 0.15,
        "SQUEEZE": 0.15,
        "PACK": 0.15,
        "UNPACK": 0.15,
        "TRANSPOSE": 0.5,
        "SLICE": 0.25,
        "STRIDED_SLICE": 0.25,
        "PAD": 0.25,
        "PADV2": 0.25,
        "RESIZE_BILINEAR": 4.0,
        "CONCATENATION": 0.2,
        "CAST": 0.1,
        "QUANTIZE": 0.1,
        "DEQUANTIZE": 0.1,
        "LOGISTIC": 4.0,
        "TANH": 4.0,
        "SOFTMAX": 5.0,
        "RELU": 0.25,
        "RELU6": 0.35,
        "LEAKY_RELU": 0.5,
        "PRELU": 0.5,
        "GATHER": 0.3,
        "GATHER_ND": 0.3,
    }
    factor = weighted_ops.get(op_name)
    if factor is None:
        return 0.0
    return float(output_elems * factor)


def _extract_tflite(file_bytes: bytes) -> Dict[str, Any]:
    """
    Extract params, input shape, GMACs and approximate GFLOPs from a TFLite flatbuffer.

    This is intentionally static and lightweight: it does not execute the model,
    but it profiles the graph structure well enough for downstream energy prediction.
    """
    size_mb = round(len(file_bytes) / (1024 * 1024), 6)

    try:
        import tflite  # type: ignore
    except ImportError:
        params_m = round((size_mb * 1024 * 1024) / 4 / 1e6, 6)
        return {
            "params_m": params_m,
            "size_mb": size_mb,
            "gflops": None,
            "gmacs": None,
            "input_shape": None,
            "model_type": "tflite",
            "extraction_complete": False,
            "note": "tflite schema package is missing, so only a file-size estimate was used.",
        }

    model = tflite.Model.GetRootAsModel(file_bytes, 0)
    if model.SubgraphsLength() == 0:
        return {
            "params_m": None,
            "size_mb": size_mb,
            "gflops": None,
            "gmacs": None,
            "input_shape": None,
            "model_type": "tflite",
            "extraction_complete": False,
            "note": "No subgraph found in the TFLite flatbuffer.",
        }

    subgraph = model.Subgraphs(0)
    producer_map = _build_tflite_producer_map(subgraph)

    input_shape = None
    if subgraph.InputsLength() > 0:
        input_tensor_idx = int(subgraph.Inputs(0))
        input_tensor = subgraph.Tensors(input_tensor_idx)
        input_shape = _tflite_tensor_shape(input_tensor) or None

    total_macs = 0.0
    total_misc_ops = 0.0
    op_histogram: dict[str, int] = {}
    heavy_profiled_ops = 0
    param_buffer_counts: dict[int, int] = {}
    heavy_param_ops = {"CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED", "TRANSPOSE_CONV", "BATCH_MATMUL", "CONV_3D"}

    for op_idx in range(subgraph.OperatorsLength()):
        operator = subgraph.Operators(op_idx)
        opcode = model.OperatorCodes(operator.OpcodeIndex())
        builtin_code = int(opcode.BuiltinCode())
        op_name = _tflite_builtin_name(tflite, builtin_code)
        op_histogram[op_name] = op_histogram.get(op_name, 0) + 1

        input_shapes = []
        for input_idx in range(operator.InputsLength()):
            tensor_idx = int(operator.Inputs(input_idx))
            if tensor_idx < 0:
                input_shapes.append([])
                continue
            input_shapes.append(_tflite_tensor_shape(subgraph.Tensors(tensor_idx)))

        output_shapes = []
        for output_idx in range(operator.OutputsLength()):
            tensor_idx = int(operator.Outputs(output_idx))
            if tensor_idx < 0:
                output_shapes.append([])
                continue
            output_shapes.append(_tflite_tensor_shape(subgraph.Tensors(tensor_idx)))

        op_macs = _estimate_tflite_heavy_op_macs(op_name, input_shapes, output_shapes)
        if op_macs > 0:
            total_macs += op_macs
            heavy_profiled_ops += 1

        if op_name in heavy_param_ops:
            for input_pos in range(1, operator.InputsLength()):
                raw_tensor_idx = int(operator.Inputs(input_pos))
                source_tensor_idx = _resolve_tflite_const_tensor_idx(
                    model=model,
                    subgraph=subgraph,
                    producer_map=producer_map,
                    tensor_idx=raw_tensor_idx,
                    tflite_mod=tflite,
                )
                if source_tensor_idx is None:
                    continue

                tensor = subgraph.Tensors(source_tensor_idx)
                buffer_idx = int(tensor.Buffer())
                if buffer_idx <= 0 or buffer_idx in param_buffer_counts:
                    continue

                element_count = _num_elements(_tflite_tensor_shape(tensor))
                if element_count <= 0:
                    buffer_obj = model.Buffers(buffer_idx)
                    nbytes = _tflite_buffer_nbytes(buffer_obj)
                    bytes_per_type = _tflite_bytes_per_type(int(tensor.Type()))
                    if nbytes > 0 and bytes_per_type and bytes_per_type > 0:
                        element_count = max(1, nbytes // bytes_per_type)

                if element_count > 0:
                    param_buffer_counts[buffer_idx] = int(element_count)

        total_misc_ops += _estimate_tflite_misc_ops(op_name, output_shapes)

    if not param_buffer_counts:
        for tensor_idx in range(subgraph.TensorsLength()):
            tensor = subgraph.Tensors(tensor_idx)
            buffer_idx = int(tensor.Buffer())
            if buffer_idx <= 0 or buffer_idx in param_buffer_counts:
                continue

            buffer_obj = model.Buffers(buffer_idx)
            nbytes = _tflite_buffer_nbytes(buffer_obj)
            if nbytes <= 0:
                continue

            element_count = _num_elements(_tflite_tensor_shape(tensor))
            if element_count <= 0:
                continue
            param_buffer_counts[buffer_idx] = int(element_count)

    total_params = int(sum(param_buffer_counts.values()))

    gmacs = round(total_macs / 1e9, 6) if total_macs > 0 else None
    gflops = round(((total_macs * 2.0) + total_misc_ops) / 1e9, 6) if total_macs > 0 else None

    note_parts = []
    if heavy_profiled_ops == 0:
        note_parts.append("No heavy arithmetic ops were recognized from the flatbuffer.")
    else:
        note_parts.append(
            f"Profiled {heavy_profiled_ops} heavy ops from {subgraph.OperatorsLength()} operators in the TFLite graph."
        )
    if total_misc_ops > 0:
        note_parts.append("GFLOPs includes lightweight elementwise-op estimates in addition to 2 x GMACs.")

    return {
        "params_m": round(total_params / 1e6, 6) if total_params > 0 else None,
        "size_mb": size_mb,
        "gflops": gflops,
        "gmacs": gmacs,
        "input_shape": input_shape,
        "model_type": "tflite",
        "extraction_complete": bool(total_params > 0 and gmacs is not None),
        "ops_profiled": dict(sorted(op_histogram.items())),
        "note": " ".join(note_parts).strip() or None,
    }


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
    """
    csv_path = rpi5_csv if "rpi" in device_key or "raspberry" in device_key else jetson_csv
    if not os.path.exists(csv_path):
        return (0.01, 100.0)

    df = pd.read_csv(csv_path)
    required = {"gflops", "params_m", "size_mb", "latency_avg_s", "throughput_iter_per_s"}
    if not required.issubset(df.columns):
        return (0.01, 100.0)

    df = df.dropna(subset=list(required))
    df = df[(df["gflops"] > 0) & (df["params_m"] > 0) & (df["size_mb"] > 0)]
    if df.empty:
        return (0.01, 100.0)

    ratio = df["latency_avg_s"] * df["throughput_iter_per_s"]
    df = df[(ratio > 0.5) & (ratio < 2.0)]
    if df.empty:
        return (0.01, 100.0)

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

    df = df.copy()
    df["_dist"] = dist / valid_feature_count

    df_fast = df[df["latency_avg_s"] < 1.0]
    pool = df_fast if len(df_fast) >= k else df

    df_sorted = pool.nsmallest(k, "_dist").copy()
    weights = 1.0 / (df_sorted["_dist"].to_numpy(dtype=float) + 1e-6)
    if not np.all(np.isfinite(weights)) or weights.sum() <= 0:
        weights = np.ones(len(df_sorted), dtype=float)

    latency = float(np.expm1(np.average(np.log1p(df_sorted["latency_avg_s"]), weights=weights)))
    throughput = float(np.expm1(np.average(np.log1p(df_sorted["throughput_iter_per_s"]), weights=weights)))
    return (round(latency, 6), round(throughput, 4))


def extract_features(
    file_bytes: bytes,
    filename: str,
    device_type: str = "jetson_nano",
    input_shape: Optional[Tuple] = None,
    jetson_csv: str = "",
    rpi5_csv: str = "",
) -> Dict[str, Any]:
    """
    Public entry point for model feature extraction.
    """
    ext = os.path.splitext(filename.lower())[1]
    warnings: list[str] = []

    if ext == ".onnx":
        info = _extract_onnx(file_bytes, input_shape=input_shape)
    elif ext == ".tflite":
        info = _extract_tflite(file_bytes)
        if info.get("gflops") is None or info.get("gmacs") is None:
            warnings.append(
                "TFLite format: static graph profiling could not infer complete GFLOPs/GMACs. "
                "Please verify the extracted complexity before using the prediction in production."
            )
    else:
        size_mb = round(len(file_bytes) / (1024 * 1024), 6)
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
            "Please convert to ONNX or TFLite for automatic profiling."
        )

    if info.get("gflops") is not None and info.get("gmacs") is None:
        info["gmacs"] = round(float(info["gflops"]) / 2.0, 6)
    if info.get("gmacs") is not None and info.get("gflops") is None:
        info["gflops"] = round(float(info["gmacs"]) * 2.0, 6)

    gflops_for_est = info.get("gflops")
    if gflops_for_est is not None and gflops_for_est > 0:
        latency_avg_s, throughput_iter_per_s = _estimate_latency_throughput(
            gflops=float(gflops_for_est),
            params_m=info.get("params_m"),
            gmacs=info.get("gmacs"),
            size_mb=info.get("size_mb"),
            device_key=device_type,
            jetson_csv=jetson_csv,
            rpi5_csv=rpi5_csv,
        )
        warnings.append(
            "latency_avg_s and throughput_iter_per_s are estimated from the benchmark dataset "
            "using nearest-neighbor matching on params, GFLOPs, GMACs and size. Real device values may differ."
        )
    else:
        latency_avg_s, throughput_iter_per_s = 0.01, 100.0
        warnings.append(
            "Could not estimate latency/throughput because GFLOPs is unknown. "
            "Please provide latency_avg_s and throughput_iter_per_s manually."
        )

    info["latency_avg_s"] = latency_avg_s
    info["throughput_iter_per_s"] = throughput_iter_per_s
    info["warnings"] = warnings
    info["device_type"] = device_type
    return info
