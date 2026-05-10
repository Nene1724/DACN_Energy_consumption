from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
CONTROLLER_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = CONTROLLER_ROOT / "data"
ARTIFACTS_DIR = CONTROLLER_ROOT / "artifacts"
RESULTS_DIR = REPO_ROOT / "results"
RAW_RESULTS_DIR = RESULTS_DIR / "raw"
PROCESSED_RESULTS_DIR = RESULTS_DIR / "processed"
FIGURES_RESULTS_DIR = RESULTS_DIR / "figures"
REPORTS_RESULTS_DIR = RESULTS_DIR / "reports"
DEFAULT_EXPERIMENT_LOG_PATH = RAW_RESULTS_DIR / "experiment_log.csv"

STATIC_MODE = "static"
ADAPTIVE_MODE = "adaptive"
VALID_DEPLOYMENT_MODES = {STATIC_MODE, ADAPTIVE_MODE}

EXPERIMENT_COLUMNS = [
    "experiment_run_id",
    "timestamp",
    "scenario",
    "deployment_mode",
    "event_source",
    "device_name",
    "device_id",
    "device_ip",
    "device_type",
    "model_name",
    "model_family",
    "model_format",
    "model_size_mb",
    "input_resolution",
    "latency_ms",
    "energy_mwh",
    "predicted_energy_mwh",
    "cpu_usage",
    "ram_usage",
    "temperature",
    "accuracy",
    "benchmark_runs",
    "throughput_iter_per_s",
    "sensor_type",
    "prediction_error_pct",
    "notes",
]

FAMILY_PATTERNS = [
    "convnextv2",
    "efficientformer",
    "efficientvit",
    "mobilenet",
    "efficientnet",
    "convnext",
    "coatnet",
    "shufflenet",
    "squeezenet",
    "ghostnet",
    "mnasnet",
    "densenet",
    "regnet",
    "resnet",
    "seresnet",
    "deit",
    "beit",
    "crossvit",
    "convit",
    "maxvit",
    "fastvit",
    "mobilevit",
    "edgenext",
    "coat",
    "cait",
    "xcit",
    "swin",
    "vit",
    "vgg",
    "inception",
    "rexnet",
    "fbnetv",
    "lcnet",
    "tinynet",
    "hrnet",
    "darknet",
    "dla",
    "dpn",
    "movenet",
    "yolo",
    "cnn",
]

NUMERIC_COLUMNS = {
    "model_size_mb",
    "latency_ms",
    "energy_mwh",
    "predicted_energy_mwh",
    "cpu_usage",
    "ram_usage",
    "temperature",
    "accuracy",
    "benchmark_runs",
    "throughput_iter_per_s",
    "prediction_error_pct",
}


def ensure_results_structure() -> Dict[str, Path]:
    RAW_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return {
        "root": RESULTS_DIR,
        "raw": RAW_RESULTS_DIR,
        "processed": PROCESSED_RESULTS_DIR,
        "figures": FIGURES_RESULTS_DIR,
        "reports": REPORTS_RESULTS_DIR,
    }


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def coerce_float(value: Any) -> Optional[float]:
    try:
        if value in ("", None):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def coerce_int(value: Any) -> Optional[int]:
    number = coerce_float(value)
    if number is None or pd.isna(number):
        return None
    return int(round(number))


def normalize_deployment_mode(value: Any) -> str:
    text = str(value or STATIC_MODE).strip().lower()
    if text in VALID_DEPLOYMENT_MODES:
        return text
    if "adapt" in text:
        return ADAPTIVE_MODE
    return STATIC_MODE


def infer_model_family(model_name: Any) -> str:
    name = str(model_name or "").strip().lower()
    if not name:
        return "unknown"
    for pattern in FAMILY_PATTERNS:
        if pattern in name:
            return pattern
    prefix = "".join(ch for ch in name.split("_")[0] if ch.isalpha())
    return prefix or "other"


def normalize_input_resolution(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        items = [str(item) for item in value]
        if len(items) >= 2:
            return f"{items[-2]}x{items[-1]}"
        return "x".join(items)

    text = str(value).strip().lower()
    if not text:
        return ""

    cleaned = text.replace("[", "").replace("]", "").replace("(", "").replace(")", "")
    cleaned = cleaned.replace(",", "x").replace(" ", "")
    parts = [part for part in cleaned.split("x") if part]
    if len(parts) >= 2:
        return f"{parts[-2]}x{parts[-1]}"
    return cleaned


def infer_model_format(model_name: Any = None, model_info: Optional[Dict[str, Any]] = None, artifact_path: Any = None) -> str:
    if isinstance(model_info, dict):
        for key in ("artifact_format", "model_format", "runtime"):
            value = str(model_info.get(key) or "").strip().lower()
            if value:
                return value

        artifact_file = str(model_info.get("artifact_file") or "").strip()
        if "." in artifact_file:
            return artifact_file.rsplit(".", 1)[-1].lower()

    for candidate in (artifact_path, model_name):
        text = str(candidate or "").strip().lower()
        if "." in text:
            return text.rsplit(".", 1)[-1]
    return "unknown"


def to_iso_timestamp(value: Any, fallback: Optional[str] = None) -> str:
    if value:
        try:
            parsed = pd.to_datetime(value, utc=True)
            if pd.isna(parsed):
                raise ValueError("NaT timestamp")
            return parsed.isoformat()
        except Exception:
            return str(value)
    return fallback or datetime.now(timezone.utc).isoformat()


def build_deterministic_run_id(*parts: Any) -> str:
    digest_source = "|".join(str(part or "") for part in parts)
    digest = hashlib.sha1(digest_source.encode("utf-8")).hexdigest()[:16]
    return f"run_{digest}"


def extract_metrics_snapshot(metrics: Optional[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    metrics = metrics or {}
    memory = metrics.get("memory") or {}
    return {
        "cpu_usage": coerce_float((metrics.get("cpu") or {}).get("percent") or metrics.get("cpu_usage")),
        "ram_usage": coerce_float(memory.get("used_percent") or metrics.get("ram_usage")),
        "temperature": coerce_float(metrics.get("temperature_c") or metrics.get("temperature")),
    }


def extract_model_size_mb(model_info: Optional[Dict[str, Any]], extra_values: Iterable[Any] = ()) -> Optional[float]:
    model_info = model_info or {}
    for key in ("size_mb", "model_size_mb"):
        value = coerce_float(model_info.get(key))
        if value is not None:
            return value

    size_bytes = coerce_float(model_info.get("model_size_bytes"))
    if size_bytes is not None:
        return round(size_bytes / (1024 * 1024), 4)

    for value in extra_values:
        number = coerce_float(value)
        if number is not None:
            return number
    return None


def extract_accuracy(model_info: Optional[Dict[str, Any]], fallback: Any = None) -> Optional[float]:
    model_info = model_info or {}
    for key in ("accuracy", "top1_accuracy", "val_accuracy", "task_accuracy"):
        value = coerce_float(model_info.get(key))
        if value is not None:
            return value
    return coerce_float(fallback)


def compute_prediction_error_pct(energy_mwh: Any, predicted_energy_mwh: Any) -> Optional[float]:
    actual = coerce_float(energy_mwh)
    predicted = coerce_float(predicted_energy_mwh)
    if actual is None or predicted is None or actual == 0:
        return None
    return abs(actual - predicted) / abs(actual) * 100.0


def empty_experiment_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=EXPERIMENT_COLUMNS)


def align_experiment_frame(df: pd.DataFrame) -> pd.DataFrame:
    aligned = df.copy()
    for column in EXPERIMENT_COLUMNS:
        if column not in aligned.columns:
            aligned[column] = pd.NA
    aligned = aligned[EXPERIMENT_COLUMNS]
    for column in NUMERIC_COLUMNS:
        aligned[column] = pd.to_numeric(aligned[column], errors="coerce")
    if "timestamp" in aligned.columns:
        aligned["timestamp"] = pd.to_datetime(aligned["timestamp"], utc=True, errors="coerce")
        aligned["timestamp"] = aligned["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        aligned["timestamp"] = aligned["timestamp"].fillna("")
    return aligned


def _merge_values(existing_value: Any, new_value: Any) -> Any:
    if new_value is None or (isinstance(new_value, float) and pd.isna(new_value)):
        return existing_value
    if isinstance(new_value, str) and not new_value.strip():
        return existing_value
    return new_value


@dataclass
class BootstrapContext:
    predictor: Optional[Callable[[Dict[str, Any]], Optional[float]]] = None


class ExperimentLogger:
    def __init__(self, csv_path: Path | str = DEFAULT_EXPERIMENT_LOG_PATH):
        ensure_results_structure()
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

    def read_dataframe(self) -> pd.DataFrame:
        if not self.csv_path.exists():
            return empty_experiment_frame()
        try:
            df = pd.read_csv(self.csv_path)
        except pd.errors.EmptyDataError:
            return empty_experiment_frame()
        return align_experiment_frame(df)

    def write_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        aligned = align_experiment_frame(df)
        aligned.to_csv(self.csv_path, index=False, encoding="utf-8")
        return aligned

    def normalize_record(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        record = {column: None for column in EXPERIMENT_COLUMNS}
        record.update(payload or {})

        model_info = record.get("model_info") if isinstance(record.get("model_info"), dict) else {}
        metrics = record.get("device_metrics") if isinstance(record.get("device_metrics"), dict) else {}
        extracted_metrics = extract_metrics_snapshot(metrics)

        record["timestamp"] = to_iso_timestamp(record.get("timestamp"))
        record["deployment_mode"] = normalize_deployment_mode(
            record.get("deployment_mode") or model_info.get("deployment_mode")
        )
        record["scenario"] = record.get("scenario") or (
            "adaptive_deployment" if record["deployment_mode"] == ADAPTIVE_MODE else "static_deployment"
        )
        record["event_source"] = record.get("event_source") or "live_logger"
        record["device_name"] = record.get("device_name") or model_info.get("device_name")
        record["device_id"] = record.get("device_id")
        record["device_ip"] = record.get("device_ip")
        record["device_type"] = record.get("device_type") or model_info.get("device_type") or "unknown"
        record["model_name"] = record.get("model_name") or model_info.get("model_name") or "unknown"
        record["model_family"] = record.get("model_family") or infer_model_family(record["model_name"])
        record["model_format"] = record.get("model_format") or infer_model_format(
            record["model_name"], model_info, record.get("artifact_path")
        )
        record["model_size_mb"] = extract_model_size_mb(model_info, [record.get("model_size_mb")])
        record["input_resolution"] = normalize_input_resolution(
            record.get("input_resolution") or model_info.get("input_resolution") or model_info.get("input_size")
        )
        record["latency_ms"] = coerce_float(record.get("latency_ms"))
        record["energy_mwh"] = coerce_float(record.get("energy_mwh"))
        record["predicted_energy_mwh"] = coerce_float(
            record.get("predicted_energy_mwh")
            or model_info.get("predicted_energy_mwh")
            or model_info.get("energy_avg_mwh")
            or model_info.get("predicted_mwh")
        )
        record["cpu_usage"] = coerce_float(record.get("cpu_usage") or extracted_metrics["cpu_usage"])
        record["ram_usage"] = coerce_float(record.get("ram_usage") or extracted_metrics["ram_usage"])
        record["temperature"] = coerce_float(record.get("temperature") or extracted_metrics["temperature"])
        record["accuracy"] = extract_accuracy(model_info, record.get("accuracy"))
        record["benchmark_runs"] = coerce_int(record.get("benchmark_runs"))
        record["throughput_iter_per_s"] = coerce_float(record.get("throughput_iter_per_s"))
        record["sensor_type"] = record.get("sensor_type") or model_info.get("sensor_type") or "unknown"
        record["prediction_error_pct"] = coerce_float(
            record.get("prediction_error_pct")
            or compute_prediction_error_pct(record.get("energy_mwh"), record.get("predicted_energy_mwh"))
        )
        record["notes"] = str(record.get("notes") or "").strip()

        record["experiment_run_id"] = (
            str(record.get("experiment_run_id") or model_info.get("experiment_run_id") or "").strip()
            or build_deterministic_run_id(
                record["timestamp"],
                record["device_type"],
                record["model_name"],
                record["deployment_mode"],
                record["event_source"],
            )
        )
        return {column: record.get(column) for column in EXPERIMENT_COLUMNS}

    def upsert_record(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        record = self.normalize_record(payload)
        df = self.read_dataframe()
        key = record["experiment_run_id"]

        if key and "experiment_run_id" in df.columns and (df["experiment_run_id"] == key).any():
            row_index = df.index[df["experiment_run_id"] == key][-1]
            merged = {column: _merge_values(df.at[row_index, column], record.get(column)) for column in EXPERIMENT_COLUMNS}
            for column, value in merged.items():
                df.at[row_index, column] = value
        else:
            df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)

        if "timestamp" in df.columns:
            sortable_ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df = df.assign(_sortable_timestamp=sortable_ts).sort_values(
                ["_sortable_timestamp", "model_name"], ascending=[True, True]
            )
            df = df.drop(columns=["_sortable_timestamp"])

        self.write_dataframe(df)
        return record

    def replace_records(self, records: List[Dict[str, Any]]) -> pd.DataFrame:
        normalized = [self.normalize_record(record) for record in records]
        df = pd.DataFrame(normalized)
        return self.write_dataframe(df)

    def bootstrap_from_existing_sources(self, predictor: Optional[Callable[[Dict[str, Any]], Optional[float]]] = None) -> pd.DataFrame:
        context = BootstrapContext(predictor=predictor)
        records: List[Dict[str, Any]] = self.read_dataframe().to_dict("records")
        records.extend(self._bootstrap_archive_benchmarks(context))
        records.extend(self._bootstrap_historical_benchmark_reports())
        records = self._merge_historical_energy(records)
        records.extend(self._bootstrap_adaptive_replay(records))
        merged_by_key: Dict[str, Dict[str, Any]] = {}
        ordered_keys: List[str] = []

        for record in records:
            normalized = self.normalize_record(record)
            key = normalized["experiment_run_id"]
            if key not in merged_by_key:
                merged_by_key[key] = normalized
                ordered_keys.append(key)
                continue
            existing = merged_by_key[key]
            merged_by_key[key] = {
                column: _merge_values(existing.get(column), normalized.get(column))
                for column in EXPERIMENT_COLUMNS
            }

        merged_rows = [merged_by_key[key] for key in ordered_keys]
        return self.write_dataframe(pd.DataFrame(merged_rows))

    def _bootstrap_archive_benchmarks(self, context: BootstrapContext) -> List[Dict[str, Any]]:
        mappings = [
            ("jetson_nano", DATA_DIR / "360_models_benchmark_jetson.csv", datetime(2026, 1, 1, tzinfo=timezone.utc)),
            ("cpu", DATA_DIR / "253_models_benchmark_rpi5.csv", datetime(2026, 2, 1, tzinfo=timezone.utc)),
        ]
        records: List[Dict[str, Any]] = []

        for device_type, csv_path, origin in mappings:
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path)
            for idx, row in df.iterrows():
                model_info = {
                    "model_name": row.get("model"),
                    "input_size": row.get("input_size"),
                    "input_resolution": row.get("input_resolution_actual"),
                    "size_mb": row.get("size_mb"),
                    "device_type": device_type,
                }
                predicted = None
                if context.predictor is not None:
                    predicted = context.predictor(
                        {
                            "device_type": device_type,
                            "params_m": coerce_float(row.get("params_m")),
                            "gflops": coerce_float(row.get("gflops")),
                            "gmacs": coerce_float(row.get("gmacs")),
                            "size_mb": coerce_float(row.get("size_mb")),
                            "latency_avg_s": coerce_float(row.get("latency_avg_s")),
                            "throughput_iter_per_s": coerce_float(row.get("throughput_iter_per_s")),
                        }
                    )

                timestamp = origin + timedelta(minutes=int(idx))
                record = {
                    "experiment_run_id": build_deterministic_run_id("archive", device_type, row.get("model"), idx),
                    "timestamp": timestamp.isoformat(),
                    "scenario": "benchmark_archive",
                    "deployment_mode": STATIC_MODE,
                    "event_source": "benchmark_archive",
                    "device_name": "Jetson Nano" if device_type == "jetson_nano" else "CPU",
                    "device_type": device_type,
                    "model_name": row.get("model"),
                    "model_format": infer_model_format(model_name=row.get("model")),
                    "model_size_mb": row.get("size_mb"),
                    "input_resolution": row.get("input_resolution_actual") or row.get("input_size"),
                    "latency_ms": coerce_float(row.get("latency_avg_s")) * 1000.0 if coerce_float(row.get("latency_avg_s")) is not None else None,
                    "energy_mwh": row.get("energy_avg_mwh"),
                    "predicted_energy_mwh": predicted,
                    "throughput_iter_per_s": row.get("throughput_iter_per_s"),
                    "benchmark_runs": row.get("runs"),
                    "notes": "Bootstrapped from benchmark CSV archive with deterministic replay timestamp.",
                    "model_info": model_info,
                }
                records.append(record)
        return records

    def _bootstrap_historical_benchmark_reports(self) -> List[Dict[str, Any]]:
        benchmark_reports = _read_json(DATA_DIR / "benchmark_reports.json", default=[])
        deployment_logs = _read_json(DATA_DIR / "deployment_logs.json", default={}).get("logs", [])
        accuracy_lookup = self._load_fall_accuracy_lookup()
        context_by_model = self._build_deployment_context(deployment_logs)

        records: List[Dict[str, Any]] = []
        for item in benchmark_reports:
            benchmark = item.get("benchmark") or {}
            prediction = item.get("prediction") or {}
            model_name = item.get("model_name") or benchmark.get("model_name") or "unknown"
            model_context = context_by_model.get(model_name, {})
            model_info = dict(model_context.get("model_info") or {})
            model_info.update(item.get("model_info") or {})
            run_id = (
                model_info.get("experiment_run_id")
                or model_context.get("experiment_run_id")
                or build_deterministic_run_id(
                    item.get("timestamp"),
                    item.get("device_type"),
                    model_name,
                    item.get("device_ip"),
                )
            )
            record = {
                "experiment_run_id": run_id,
                "timestamp": item.get("timestamp") or benchmark.get("timestamp"),
                "scenario": "live_device_benchmark",
                "deployment_mode": normalize_deployment_mode(
                    model_info.get("deployment_mode") or model_context.get("deployment_mode")
                ),
                "event_source": "benchmark_report",
                "device_name": item.get("device_name"),
                "device_id": item.get("device_id"),
                "device_ip": item.get("device_ip"),
                "device_type": item.get("device_type"),
                "model_name": model_name,
                "model_format": infer_model_format(model_name, model_info),
                "model_size_mb": extract_model_size_mb(
                    model_info,
                    [
                        ((model_context.get("device_response") or {}).get("model_size_bytes") or 0) / (1024 * 1024)
                        if (model_context.get("device_response") or {}).get("model_size_bytes")
                        else None
                    ],
                ),
                "input_resolution": normalize_input_resolution(
                    benchmark.get("input_size")
                    or model_info.get("input_resolution")
                    or model_info.get("input_size")
                ),
                "latency_ms": coerce_float(benchmark.get("latency_avg_s")) * 1000.0
                if coerce_float(benchmark.get("latency_avg_s")) is not None
                else None,
                "energy_mwh": None,
                "predicted_energy_mwh": prediction.get("predicted_energy_mwh"),
                "accuracy": accuracy_lookup.get(str(model_name).lower()),
                "throughput_iter_per_s": benchmark.get("throughput_iter_per_s"),
                "benchmark_runs": benchmark.get("benchmark_runs"),
                "notes": "Bootstrapped from stored benchmark_reports.json.",
                "model_info": model_info,
            }
            records.append(record)
        return records

    def _merge_historical_energy(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        energy_reports = _read_json(DATA_DIR / "energy_measurements.json", default=[])
        if not energy_reports:
            return records

        index_by_model: Dict[str, List[int]] = {}
        for idx, record in enumerate(records):
            model_name = str(record.get("model_name") or "").lower()
            index_by_model.setdefault(model_name, []).append(idx)

        for item in energy_reports:
            model_name = str(item.get("model_name") or "").lower()
            candidates = index_by_model.get(model_name, [])
            target_idx = candidates[-1] if candidates else None
            if target_idx is None:
                records.append(
                    {
                        "experiment_run_id": build_deterministic_run_id(
                            item.get("timestamp"),
                            item.get("device_type"),
                            item.get("model_name"),
                            "energy_report",
                        ),
                        "timestamp": item.get("timestamp"),
                        "scenario": "live_energy_measurement",
                        "deployment_mode": STATIC_MODE,
                        "event_source": "energy_report",
                        "device_id": item.get("device_id"),
                        "device_type": item.get("device_type"),
                        "model_name": item.get("model_name"),
                        "energy_mwh": item.get("actual_energy_mwh"),
                        "predicted_energy_mwh": item.get("predicted_mwh"),
                        "sensor_type": item.get("sensor_type"),
                        "notes": "Standalone historical energy report.",
                    }
                )
                continue

            record = records[target_idx]
            record["timestamp"] = item.get("timestamp") or record.get("timestamp")
            record["energy_mwh"] = item.get("actual_energy_mwh")
            record["predicted_energy_mwh"] = item.get("predicted_mwh") or record.get("predicted_energy_mwh")
            record["sensor_type"] = item.get("sensor_type")
            record["prediction_error_pct"] = item.get("pct_error")
            record["notes"] = (
                f"{record.get('notes', '').strip()} Merged with historical energy_measurements.json."
            ).strip()
        return records

    def _bootstrap_adaptive_replay(self, existing_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        df = pd.DataFrame(existing_records)
        if df.empty:
            return []

        df = df[df["event_source"] == "benchmark_archive"].copy()
        if df.empty:
            return []

        df["latency_ms"] = pd.to_numeric(df["latency_ms"], errors="coerce")
        df["energy_mwh"] = pd.to_numeric(df["energy_mwh"], errors="coerce")
        replay_records: List[Dict[str, Any]] = []

        for device_type, device_df in df.groupby("device_type"):
            clean_df = device_df.dropna(subset=["latency_ms", "energy_mwh"]).copy()
            if clean_df.empty:
                continue

            static_baseline = clean_df.sort_values("energy_mwh").iloc[len(clean_df) // 2]
            latency_budgets = clean_df["latency_ms"].quantile([0.25, 0.5, 0.75]).tolist()
            thermal_states = [42.0, 50.0, 58.0]

            for request_idx, (latency_budget, temperature) in enumerate(zip(latency_budgets, thermal_states), start=1):
                eligible = clean_df[clean_df["latency_ms"] <= latency_budget].copy()
                if eligible.empty:
                    eligible = clean_df.copy()
                adaptive_choice = eligible.sort_values(["energy_mwh", "latency_ms"]).iloc[0]
                replay_timestamp = datetime(2026, 3, request_idx, 8, 0, tzinfo=timezone.utc).isoformat()

                for mode, selected in (
                    (STATIC_MODE, static_baseline),
                    (ADAPTIVE_MODE, adaptive_choice),
                ):
                    replay_records.append(
                        {
                            "experiment_run_id": build_deterministic_run_id(
                                "adaptive_replay",
                                device_type,
                                request_idx,
                                mode,
                            ),
                            "timestamp": replay_timestamp,
                            "scenario": "deployment_policy_replay",
                            "deployment_mode": mode,
                            "event_source": "adaptive_replay",
                            "device_name": "Jetson Nano" if device_type == "jetson_nano" else "CPU",
                            "device_type": device_type,
                            "model_name": selected["model_name"],
                            "model_family": selected.get("model_family") or infer_model_family(selected["model_name"]),
                            "model_format": selected.get("model_format") or "unknown",
                            "model_size_mb": selected.get("model_size_mb"),
                            "input_resolution": selected.get("input_resolution"),
                            "latency_ms": selected.get("latency_ms"),
                            "energy_mwh": selected.get("energy_mwh"),
                            "predicted_energy_mwh": selected.get("predicted_energy_mwh"),
                            "temperature": temperature,
                            "accuracy": selected.get("accuracy"),
                            "notes": (
                                "Replay scenario derived from measured benchmark archive. "
                                f"Latency budget = {latency_budget:.2f} ms."
                            ),
                        }
                    )
        return replay_records

    def _build_deployment_context(self, deployment_logs: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        context: Dict[str, Dict[str, Any]] = {}
        for item in deployment_logs:
            if item.get("type") != "success":
                continue
            metadata = item.get("metadata") or {}
            device_response = metadata.get("device_response") or {}
            model_info = device_response.get("model_info") or {}
            model_name = str(metadata.get("model_name") or device_response.get("model_name") or "").strip()
            if not model_name:
                continue
            context[model_name] = {
                "timestamp": item.get("timestamp"),
                "deployment_mode": normalize_deployment_mode(
                    metadata.get("deployment_mode") or model_info.get("deployment_mode")
                ),
                "experiment_run_id": model_info.get("experiment_run_id")
                or build_deterministic_run_id(item.get("timestamp"), model_name, metadata.get("device_endpoint")),
                "device_response": device_response,
                "model_info": model_info,
            }
        return context

    def _load_fall_accuracy_lookup(self) -> Dict[str, float]:
        report_dir = ARTIFACTS_DIR / "report_figures"
        reports = sorted(report_dir.glob("ur_fall_eval_*.json"))
        if not reports:
            return {}

        latest_report = reports[-1]
        data = _read_json(latest_report, default={})
        stats = data.get("stats") or {}
        accuracy = coerce_float(stats.get("accuracy"))
        if accuracy is None:
            return {}
        return {"movenet_singlepose_lightning_f16": accuracy * 100.0}
