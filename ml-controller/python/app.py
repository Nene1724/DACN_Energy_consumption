import os
import re
import json
import time
from datetime import datetime, timezone
from urllib.parse import urlparse
from typing import Optional
from flask import Flask, request, render_template, jsonify, send_from_directory, make_response, Response, stream_with_context
import requests
from requests.utils import requote_uri
from model_analyzer import ModelAnalyzer
from energy_predictor_service import EnergyPredictorService
from log_manager import LogManager
from onnx_feature_extractor import extract_features as extract_model_features

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates'))

# Load environment variables from .env (non-destructive: keep existing env vars)
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")


def load_env_file(path: str):
    if not path or not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key and not os.getenv(key):
                    os.environ[key] = val
    except Exception as e:
        print(f"[WARN] Could not load env file {path}: {e}")


load_env_file(ENV_PATH)

DATA_DIR = os.path.join(BASE_DIR, "data")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
MODEL_STORE_DIR = os.path.join(BASE_DIR, "model_store")
NEW_MODELS_DIR = os.path.join(BASE_DIR, "new_models")
CSV_PATH = os.path.join(DATA_DIR, "360_models_benchmark_jetson.csv") 
RPI5_CSV_PATH = os.path.join(DATA_DIR, "253_models_benchmark_rpi5.csv")  
LOG_FILE_PATH = os.path.join(DATA_DIR, "deployment_logs.json")
ENERGY_REPORTS_PATH = os.path.join(DATA_DIR, "energy_measurements.json")
BENCHMARK_REPORTS_PATH = os.path.join(DATA_DIR, "benchmark_reports.json")
FALL_EVENTS_PATH = os.path.join(DATA_DIR, "fall_detection_events.json")
PREFERRED_ARTIFACT_EXTS = [".tflite", ".onnx", ".pth", ".pt", ".bin"]
BALENA_API_BASE = os.getenv("BALENA_API_BASE", "https://api.balena-cloud.com")
BALENA_DEFAULT_TIMEOUT = int(os.getenv("BALENA_API_TIMEOUT", "30"))
BALENA_PUBLIC_URL_CACHE_TTL_S = max(30, int(os.getenv("BALENA_PUBLIC_URL_CACHE_TTL_S", "300") or "300"))
BALENA_PUBLIC_URL_NEGATIVE_TTL_S = max(15, int(os.getenv("BALENA_PUBLIC_URL_NEGATIVE_TTL_S", "60") or "60"))
BALENA_PUBLIC_URL_CACHE = {}
MODEL_ANALYZE_MAX_MB = int(os.getenv("MODEL_ANALYZE_MAX_MB", "128"))


def _get_balena_token() -> str:
    """Return the Balena token from any supported env var name."""
    for key in ("BALENA_API_TOKEN", "BALENA_API_KEY", "BALENA_TOKEN", "BALENA_SESSION_TOKEN"):
        val = (os.getenv(key) or "").strip()
        if val:
            return val
    return ""


def _infer_bind_ip(prefer_connect_to_ip: Optional[str] = None) -> str:
    """Best-effort guess of a LAN-facing IP address for this host."""
    target = (prefer_connect_to_ip or "8.8.8.8").strip()
    try:
        import socket

        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect((target, 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return ""


def _get_controller_base_url(req, prefer_connect_to_ip: Optional[str] = None) -> str:
    """Return a device-reachable base URL for model downloads.

    Priority:
    1) CONTROLLER_PUBLIC_URL (recommended for cross-network / EC2)
    2) Request-derived scheme+host (works when request host is reachable)
    3) Best-effort inferred LAN IP (only when request host is localhost)
    """
    public_base_url = (os.getenv("CONTROLLER_PUBLIC_URL") or "").strip()
    if public_base_url:
        return public_base_url.rstrip("/")

    # Respect reverse-proxy headers if present.
    forwarded_proto = (req.headers.get("X-Forwarded-Proto") or "").split(",")[0].strip()
    forwarded_host = (req.headers.get("X-Forwarded-Host") or "").split(",")[0].strip()
    scheme = forwarded_proto or getattr(req, "scheme", None) or "http"
    host = forwarded_host or getattr(req, "host", "")
    base = f"{scheme}://{host}".rstrip("/")

    # If running locally and accessed via localhost, try to swap in a LAN IP.
    if any(loop in host for loop in ("localhost", "127.0.0.1", "0.0.0.0")):
        inferred_ip = _infer_bind_ip(prefer_connect_to_ip)
        if inferred_ip:
            # Preserve port if present.
            port = ""
            if ":" in host:
                port = host.split(":", 1)[1]
            base = f"{scheme}://{inferred_ip}{(':' + port) if port else ''}"

    return base.rstrip("/")

predictor_service = EnergyPredictorService(ARTIFACTS_DIR)
analyzer = ModelAnalyzer(
    CSV_PATH,
    predictor_service=predictor_service,
    model_store_dir=MODEL_STORE_DIR,
    rpi5_csv_path=RPI5_CSV_PATH,
    extra_model_dirs=[NEW_MODELS_DIR],
)
log_manager = LogManager(LOG_FILE_PATH, max_logs=50)


def _normalize_key(value: str) -> str:
    return "".join(ch.lower() for ch in value if ch.isalnum())


def _iter_local_model_dirs():
    seen = set()
    for directory in (MODEL_STORE_DIR, NEW_MODELS_DIR):
        normalized = os.path.abspath(directory)
        if normalized in seen:
            continue
        seen.add(normalized)
        if os.path.isdir(directory):
            yield directory


def _find_local_artifact_path(filename: str):
    safe_name = os.path.basename(str(filename or ""))
    if not safe_name:
        return None
    for directory in _iter_local_model_dirs():
        candidate = os.path.join(directory, safe_name)
        if os.path.exists(candidate):
            return candidate
    return None


def _list_local_artifacts():
    items = []
    seen = set()
    for directory in _iter_local_model_dirs():
        for filename in os.listdir(directory):
            if filename in seen:
                continue
            seen.add(filename)
            items.append(filename)
    return sorted(items)


def resolve_model_artifact(model_name: str):
    """
    Find a matching model artifact file inside local artifact directories.
    Accepts variations in casing/spacing and tries preferred extensions.
    """
    if not model_name:
        return None

    normalized = _normalize_key(model_name)
    if not normalized:
        return None

    local_dirs = list(_iter_local_model_dirs())
    if not local_dirs:
        return None

    # Exact match ignoring casing / non-alphanumerics
    for directory in local_dirs:
        for filename in os.listdir(directory):
            base, _ = os.path.splitext(filename)
            if _normalize_key(base) == normalized:
                return filename

    # Try preferred extensions with sanitized base names
    provided_base, provided_ext = os.path.splitext(model_name)
    if provided_ext:
        if _find_local_artifact_path(model_name):
            return model_name

    slug = re.sub(r"[^a-zA-Z0-9]+", "_", model_name).strip("_") or model_name
    for ext in PREFERRED_ARTIFACT_EXTS:
        if model_name.lower().endswith(ext.lower()):
            candidate = model_name
        else:
            candidate = f"{model_name}{ext}"

        if _find_local_artifact_path(candidate):
            return candidate

        slug_candidate = f"{slug}{ext}"
        if _find_local_artifact_path(slug_candidate):
            return slug_candidate

    return None


def _safe_float(value):
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _sanitize_model_base_name(value: str) -> str:
    base = os.path.splitext(os.path.basename(str(value or "")))[0]
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._-")
    return sanitized or f"custom_model_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"


def _build_device_urls(bbb_ip: Optional[str], device_uuid: Optional[str], suffix: str):
    suffix = suffix if suffix.startswith("/") else f"/{suffix}"
    urls_to_try = []
    if device_uuid:
        public_url = get_device_public_url(device_uuid)
        if public_url:
            urls_to_try.append(("public", f"{public_url}{suffix}"))
    if bbb_ip:
        urls_to_try.append(("local", f"http://{bbb_ip}:8000{suffix}"))
    return urls_to_try


def _normalize_device_key(device_type: str) -> str:
    device_lower = str(device_type or "").lower()
    if any(k in device_lower for k in ["jetson", "nano"]):
        return "jetson_nano"
    if any(k in device_lower for k in ["raspberry", "rpi", "pi"]):
        return "raspberry_pi5"
    return "jetson_nano"


def _load_energy_thresholds():
    thresholds_path = os.path.join(ARTIFACTS_DIR, "energy_thresholds.json")
    if not os.path.exists(thresholds_path):
        return {}
    try:
        with open(thresholds_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _build_prediction_result(model_name: str, device_type: str, pred: dict, payload: dict):
    thresholds_data = _load_energy_thresholds()
    device_key = _normalize_device_key(device_type)
    thresholds = thresholds_data.get(device_key, {})

    p25 = thresholds.get("p25", 50)
    p50 = thresholds.get("p50", 85)
    p75 = thresholds.get("p75", 150)
    energy = float(pred["prediction_mwh"])

    if energy < p25:
        energy_category = "excellent"
        recommendation = "deploy"
        reason = f"Energy consumption ({energy:.1f} mWh) is within excellent range (< {p25} mWh) for {device_type}"
    elif energy < p50:
        energy_category = "good"
        recommendation = "deploy"
        reason = f"Energy consumption ({energy:.1f} mWh) is good ({p25}-{p50} mWh) for {device_type}"
    elif energy < p75:
        energy_category = "acceptable"
        recommendation = "deploy_with_caution"
        reason = f"Energy consumption ({energy:.1f} mWh) is acceptable ({p50}-{p75} mWh) for {device_type}. Consider optimization."
    else:
        energy_category = "high"
        recommendation = "not_recommend"
        reason = f"Energy consumption ({energy:.1f} mWh) is high (> {p75} mWh) for {device_type}. Not recommended for deployment."

    return {
        "model_name": model_name,
        "device_type": device_type,
        "predicted_energy_mwh": round(energy, 2),
        "ci_lower_mwh": round(pred.get("ci_lower_mwh", 0), 2),
        "ci_upper_mwh": round(pred.get("ci_upper_mwh", 0), 2),
        "energy_category": energy_category,
        "recommendation": recommendation,
        "reason": reason,
        "thresholds": {
            "p25": p25,
            "p50": p50,
            "p75": p75,
        },
        "model_info": {
            "params_m": payload.get("params_m"),
            "gflops": payload.get("gflops"),
            "size_mb": payload.get("size_mb"),
            "latency_avg_s": payload.get("latency_avg_s"),
            "throughput_iter_per_s": payload.get("throughput_iter_per_s"),
        },
        "model_used_for_prediction": pred.get("model_used"),
        "prediction_mape_pct": pred.get("mape_pct"),
    }


def _predict_energy_for_payload(data: dict):
    required_fields = ["device_type", "params_m", "gflops", "gmacs", "size_mb", "latency_avg_s", "throughput_iter_per_s"]
    missing_fields = [field for field in required_fields if data.get(field) is None]
    if missing_fields:
        raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")

    payload = {
        "device_type": data["device_type"],
        "params_m": float(data["params_m"]),
        "gflops": float(data["gflops"]),
        "gmacs": float(data["gmacs"]),
        "size_mb": float(data["size_mb"]),
        "latency_avg_s": float(data["latency_avg_s"]),
        "throughput_iter_per_s": float(data["throughput_iter_per_s"]),
    }
    predictions = predictor_service.predict([payload])
    if not predictions or predictions[0].get("prediction_mwh") is None:
        error_msg = predictions[0].get("error", "Unknown error") if predictions else "No predictions returned"
        raise RuntimeError(f"Energy prediction failed: {error_msg}")

    pred = predictions[0]
    model_name = data.get("model_name", "unknown")
    result = _build_prediction_result(model_name=model_name, device_type=data["device_type"], pred=pred, payload=payload)

    if model_name and model_name != "unknown":
        artifact = resolve_model_artifact(model_name)
        result["model_downloaded"] = artifact is not None
    else:
        result["model_downloaded"] = False

    return result, pred, payload


def _load_energy_reports():
    if not os.path.exists(ENERGY_REPORTS_PATH):
        return []
    try:
        with open(ENERGY_REPORTS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save_energy_reports(items):
    os.makedirs(os.path.dirname(ENERGY_REPORTS_PATH), exist_ok=True)
    with open(ENERGY_REPORTS_PATH, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


def _get_latest_energy_report(device_id=None, model_name=None):
    items = _load_energy_reports()
    normalized_device_id = str(device_id or "").strip()
    normalized_model_name = str(model_name or "").strip().lower()

    def matches(item, require_model=True):
        if normalized_device_id and str(item.get("device_id") or "").strip() != normalized_device_id:
            return False
        if require_model and normalized_model_name:
            return str(item.get("model_name") or "").strip().lower() == normalized_model_name
        return True

    for item in reversed(items):
        if matches(item, require_model=True):
            return item

    if normalized_device_id:
        for item in reversed(items):
            if matches(item, require_model=False):
                return item

    return None


def _load_json_list(path: str):
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save_json_list(path: str, items):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


def _load_benchmark_reports():
    return _load_json_list(BENCHMARK_REPORTS_PATH)


def _save_benchmark_reports(items):
    _save_json_list(BENCHMARK_REPORTS_PATH, items)


def _get_latest_benchmark_report(device_id=None, device_ip=None, model_name=None):
    items = _load_benchmark_reports()
    normalized_device_id = str(device_id or "").strip()
    normalized_device_ip = str(device_ip or "").strip()
    normalized_model_name = str(model_name or "").strip().lower()

    for item in reversed(items):
        if normalized_device_id and str(item.get("device_id") or "").strip() != normalized_device_id:
            continue
        if normalized_device_ip and str(item.get("device_ip") or "").strip() != normalized_device_ip:
            continue
        if normalized_model_name and str(item.get("model_name") or "").strip().lower() != normalized_model_name:
            continue
        return item
    return None


def _load_fall_events():
    return _load_json_list(FALL_EVENTS_PATH)


def _save_fall_events(items):
    _save_json_list(FALL_EVENTS_PATH, items)


def _get_latest_fall_event(device_id=None, device_ip=None):
    items = _load_fall_events()
    normalized_device_id = str(device_id or "").strip()
    normalized_device_ip = str(device_ip or "").strip()
    for item in reversed(items):
        if normalized_device_id and str(item.get("device_id") or "").strip() != normalized_device_id:
            continue
        if normalized_device_ip and str(item.get("device_ip") or "").strip() != normalized_device_ip:
            continue
        return item
    return None


def _append_benchmark_report(item: dict):
    items = _load_benchmark_reports()
    items.append(item)
    items = items[-500:]
    _save_benchmark_reports(items)
    return item


def _append_fall_event(item: dict):
    items = _load_fall_events()
    items.append(item)
    items = items[-500:]
    _save_fall_events(items)
    return item


def _wait_for_device_ready_for_benchmark(bbb_ip=None, device_uuid=None, timeout_s: float = 75.0):
    started = datetime.now(timezone.utc).timestamp()
    last_status = None
    while (datetime.now(timezone.utc).timestamp() - started) < timeout_s:
        for _, url in _build_device_urls(bbb_ip, device_uuid, "/status"):
            try:
                resp = requests.get(url, timeout=12)
                resp.raise_for_status()
                payload = resp.json()
                if isinstance(payload, dict):
                    last_status = payload
                    status_value = str(payload.get("status") or "").lower()
                    if status_value == "running":
                        return payload
                    if status_value == "error":
                        return payload
            except Exception:
                continue
        import time
        time.sleep(2.0)
    return last_status


def _benchmark_and_repredict_device(
    *,
    bbb_ip=None,
    device_uuid=None,
    device_name=None,
    device_type=None,
    model_name=None,
    model_info=None,
    warmup_runs: int = 5,
    benchmark_runs: int = 30,
):
    urls_to_try = _build_device_urls(bbb_ip, device_uuid, "/benchmark")
    if not urls_to_try:
        raise ValueError("No valid endpoint available")

    benchmark_payload = {
        "warmup_runs": int(warmup_runs),
        "benchmark_runs": int(benchmark_runs),
    }

    last_error = None
    benchmark_result = None
    used_url_type = None
    for url_type, url in urls_to_try:
        try:
            resp = requests.post(url, json=benchmark_payload, timeout=120)
            resp.raise_for_status()
            benchmark_result = resp.json()
            used_url_type = url_type
            break
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
            last_error = str(exc)
            continue
        except Exception as exc:
            last_error = str(exc)
            continue

    if not benchmark_result:
        raise RuntimeError(last_error or "Unable to benchmark model on device")

    merged_model_info = {}
    if isinstance(model_info, dict):
        merged_model_info.update(model_info)

    status_payload = _wait_for_device_ready_for_benchmark(bbb_ip=bbb_ip, device_uuid=device_uuid, timeout_s=20.0)
    if isinstance(status_payload, dict) and isinstance(status_payload.get("model_info"), dict):
        merged_model_info = {**merged_model_info, **status_payload["model_info"]}

    inferred_device_type = (
        device_type
        or merged_model_info.get("device_type")
        or merged_model_info.get("device")
        or "jetson_nano"
    )
    effective_model_name = model_name or merged_model_info.get("model_name") or benchmark_result.get("model_name") or "unknown"

    predictor_input = {
        "model_name": effective_model_name,
        "device_type": inferred_device_type,
        "params_m": _safe_float(merged_model_info.get("params_m")),
        "gflops": _safe_float(merged_model_info.get("gflops")),
        "gmacs": _safe_float(merged_model_info.get("gmacs")),
        "size_mb": _safe_float(merged_model_info.get("size_mb")),
        "latency_avg_s": _safe_float(benchmark_result.get("latency_avg_s")),
        "throughput_iter_per_s": _safe_float(benchmark_result.get("throughput_iter_per_s")),
    }

    adjusted_prediction = None
    prediction_error = None
    if all(predictor_input.get(key) is not None for key in ("params_m", "gflops", "gmacs", "size_mb", "latency_avg_s", "throughput_iter_per_s")):
        try:
            adjusted_prediction, _, _ = _predict_energy_for_payload(predictor_input)
        except Exception as exc:
            prediction_error = str(exc)
    else:
        prediction_error = "Missing model features required for calibrated prediction"

    report_item = {
        "timestamp": benchmark_result.get("timestamp") or datetime.now(timezone.utc).isoformat(),
        "device_id": device_uuid,
        "device_ip": bbb_ip,
        "device_name": device_name,
        "model_name": effective_model_name,
        "device_type": inferred_device_type,
        "benchmark": benchmark_result,
        "prediction": adjusted_prediction,
        "prediction_error": prediction_error,
        "used_url_type": used_url_type,
    }
    _append_benchmark_report(report_item)
    return report_item


def _sanitize_filter_value(value: str) -> str:
    """Escape single quotes for OData filters."""
    return value.replace("'", "''")


def _extract_app_info(app_field):
    if isinstance(app_field, list) and app_field:
        candidate = app_field[0]
    elif isinstance(app_field, dict):
        candidate = app_field
    else:
        candidate = {}
    if isinstance(candidate, dict):
        return {
            "name": candidate.get("app_name") or candidate.get("device_type"),
            "slug": candidate.get("slug")
        }
    return {"name": None, "slug": None}


def _transform_balena_device(raw):
    app_info = _extract_app_info(raw.get("belongs_to__application"))
    
    # Balena API returns multiple IPs separated by spaces (IPv4 + IPv6s)
    # Parse and take first IP only
    endpoint = raw.get("vpn_address")
    if not endpoint:
        ip_address = raw.get("ip_address") or ""
        ips = [ip.strip() for ip in ip_address.split() if ip.strip()]
        endpoint = ips[0] if ips else None
    
    return {
        "id": raw.get("id"),
        "name": raw.get("device_name"),
        "uuid": raw.get("uuid"),
        "status": raw.get("status"),
        "is_online": raw.get("is_online"),
        "os_version": raw.get("os_version"),
        "supervisor_version": raw.get("supervisor_version"),
        "last_connectivity_event": raw.get("last_connectivity_event"),
        "app": app_info,
        "ip_address": raw.get("ip_address"),
        "vpn_address": raw.get("vpn_address"),
        "endpoint": endpoint,
        "webconsole_url": raw.get("webconsole_url"),
        "is_web_accessible": raw.get("is_web_accessible")
    }


def _get_balena_base_url() -> str:
    """
    Return a validated Balena API base URL (scheme + host only).
    Falls back to default if env is missing/blank/malformed.
    """
    raw = (os.getenv("BALENA_API_BASE") or "https://api.balena-cloud.com").strip()
    if not raw:
        raw = "https://api.balena-cloud.com"
    if "://" not in raw:
        raw = f"https://{raw.lstrip('/')}"
    parsed = urlparse(raw)
    if not parsed.scheme or not parsed.netloc:
        return "https://api.balena-cloud.com"
    return f"{parsed.scheme}://{parsed.netloc}"


def fetch_balena_devices(app_slug=None, online_only=False, limit=50, token=None):
    token = (token or _get_balena_token()).strip()
    if not token:
        raise ValueError("BALENA_API_TOKEN chưa được cấu hình")

    limit = max(1, min(int(limit or 50), 200))

    # Build OData query parameters
    select_fields = [
        "id", "device_name", "uuid", "status", "is_online",
        "ip_address", "vpn_address", "os_version",
        "supervisor_version", "last_connectivity_event", "webconsole_url",
        "is_web_accessible"
    ]
    
    query_parts = [
        f"$select={','.join(select_fields)}",
        "$expand=belongs_to__application($select=app_name,slug)",
        "$orderby=device_name asc",
        f"$top={limit}"
    ]

    filters = []
    if app_slug:
        if app_slug.isdigit():
            filters.append(f"belongs_to__application eq {app_slug}")
        else:
            # Use proper OData encoding for slug with special characters
            sanitized = _sanitize_filter_value(app_slug)
            filters.append(f"belongs_to__application/any(app:app/slug eq '{sanitized}')")
    if online_only:
        filters.append("is_online eq true")
    if filters:
        query_parts.append(f"$filter={' and '.join(filters)}")

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json"
    }

    # Build URL manually to avoid double encoding
    url = f"{BALENA_API_BASE.rstrip('/')}/v6/device?{'&'.join(query_parts)}"
    
    resp = requests.get(url, headers=headers, timeout=BALENA_DEFAULT_TIMEOUT)
    resp.raise_for_status()
    payload = resp.json()
    devices = payload.get("d") or payload.get("value") or []
    return [_transform_balena_device(item) for item in devices]


def get_device_public_url(device_uuid: str, token=None) -> str:
    """
    Fetch the Public Device URL for a device by its UUID.
    Returns: https://<uuid>.balena-devices.com or empty string if not enabled.
    """
    token = (token or _get_balena_token()).strip()
    if not token or not device_uuid:
        return ""

    cache_key = str(device_uuid).strip()
    now_ts = time.time()
    cached = BALENA_PUBLIC_URL_CACHE.get(cache_key)
    if cached and cached.get("expires_at", 0) > now_ts:
        return cached.get("url") or ""

    try:
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json"
        }
        url = f"{BALENA_API_BASE.rstrip('/')}/v6/device?$filter=uuid eq '{device_uuid}'&$select=uuid,is_web_accessible"
        resp = requests.get(url, headers=headers, timeout=BALENA_DEFAULT_TIMEOUT)
        resp.raise_for_status()
        payload = resp.json()
        devices = payload.get("d") or payload.get("value") or []
        
        if devices and devices[0].get("is_web_accessible"):
            # Public URL format: https://<uuid>.balena-devices.com
            resolved = f"https://{device_uuid}.balena-devices.com"
            BALENA_PUBLIC_URL_CACHE[cache_key] = {
                "url": resolved,
                "expires_at": now_ts + BALENA_PUBLIC_URL_CACHE_TTL_S,
            }
            return resolved
        BALENA_PUBLIC_URL_CACHE[cache_key] = {
            "url": "",
            "expires_at": now_ts + BALENA_PUBLIC_URL_NEGATIVE_TTL_S,
        }
    except Exception as e:
        print(f"[WARN] Could not fetch public URL for device {device_uuid}: {e}")
        BALENA_PUBLIC_URL_CACHE[cache_key] = {
            "url": "",
            "expires_at": now_ts + BALENA_PUBLIC_URL_NEGATIVE_TTL_S,
        }
    
    return ""


def _render_with_cache(template_name, **context):
    resp = make_response(render_template(template_name, **context))
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


@app.route("/")
@app.route("/deployment")
def index():
    return _render_with_cache("index.html", initial_view="deployment")


@app.route("/monitoring")
def monitoring():
    return _render_with_cache("index.html", initial_view="monitoring")


@app.route("/medical")
def medical():
    return _render_with_cache("index.html", initial_view="medical")


@app.route("/analytics")
def analytics():
    """Legacy analytics route: keep URL alive but render deployment view."""
    return _render_with_cache("index.html", initial_view="deployment")


@app.route("/favicon.ico")
def favicon():
    """Return empty response for favicon to avoid 404 errors"""
    return "", 204


@app.route("/api/models/all", methods=["GET"])
def get_all_models():
    """API: Lấy danh sách tất cả models từ benchmark CSV (KHÔNG bao gồm popular models)"""
    try:
        models = analyzer.get_all_models()
        
        # Add download status for each model
        for model in models:
            model_name = model.get("name", "")
            artifact = None
            for ext in ['.tflite', '.onnx']:
                artifact = resolve_model_artifact(model_name + ext)
                if artifact:
                    break
            
            model["model_downloaded"] = artifact is not None
            if artifact:
                model["artifact_file"] = artifact
        
        stats = analyzer.get_energy_stats()
        return jsonify({
            "success": True, 
            "models": models,
            "stats": stats
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/models/<model_name>", methods=["GET"])
def get_model_details(model_name):
    """API: Lấy chi tiết 1 model"""
    try:
        details = analyzer.get_model_details(model_name)
        if details is None:
            return jsonify({"success": False, "error": "Model not found"}), 404
        
        return jsonify({"success": True, "model": details})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/models/check", methods=["POST"])
def check_model_availability():
    """API: Check if model exists in local artifact directories."""
    try:
        data = request.get_json()
        model_name = data.get("model_name")
        
        if not model_name:
            return jsonify({"success": False, "error": "Model name is required"}), 400
        
        # Check if model artifact exists
        artifact = resolve_model_artifact(model_name)
        
        if artifact:
            artifact_path = _find_local_artifact_path(artifact)
            if not artifact_path:
                return jsonify({
                    "success": False,
                    "available": False,
                    "message": f"Artifact {artifact} could not be resolved on disk"
                }), 404
            file_size = os.path.getsize(artifact_path)
            file_size_mb = round(file_size / (1024 * 1024), 2)
            
            return jsonify({
                "success": True,
                "available": True,
                "artifact": artifact,
                "size_mb": file_size_mb,
                "format": os.path.splitext(artifact)[1].lstrip('.')
            })
        else:
            return jsonify({
                "success": True,
                "available": False,
                "message": f"Model {model_name} not found in local artifact directories"
            })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/models/download", methods=["POST"])
def download_model_from_hub():
    """API: Download model về model_store từ Hugging Face"""
    try:
        data = request.get_json()
        model_name = data.get("model_name")
        
        if not model_name:
            return jsonify({"success": False, "error": "Model name is required"}), 400
        
        # Check if model already exists
        existing_artifact = resolve_model_artifact(model_name)
        if existing_artifact:
            return jsonify({
                "success": True, 
                "message": f"Model {model_name} already downloaded",
                "artifact": existing_artifact
            })
        
        # Try to download using timm and convert to ONNX
        try:
            import timm
            import torch
            import onnx
            
            # Create model_store directory if not exists
            os.makedirs(MODEL_STORE_DIR, exist_ok=True)
            
            # Download model from timm
            print(f"Downloading model: {model_name}")
            model = timm.create_model(model_name, pretrained=True)
            model.eval()  # Set to evaluation mode
            
            # Get input size from model config
            try:
                data_config = timm.data.resolve_data_config(model.pretrained_cfg)
                input_size = data_config.get('input_size', (3, 224, 224))
            except:
                input_size = (3, 224, 224)  # Default input size
            
            # Create dummy input for ONNX export
            batch_size = 1
            dummy_input = torch.randn(batch_size, *input_size)
            
            # Export to ONNX format (best compatibility for embedded devices)
            temp_output = os.path.join(MODEL_STORE_DIR, f"{model_name}_temp.onnx")
            output_filename = f"{model_name}.onnx"
            output_path = os.path.join(MODEL_STORE_DIR, output_filename)
            
            print(f"Converting {model_name} to ONNX format...")
            torch.onnx.export(
                model,
                dummy_input,
                temp_output,
                export_params=True,
                opset_version=17,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            # Check if external data file was created
            temp_data_file = temp_output + ".data"
            if os.path.exists(temp_data_file):
                print(f"Merging external data into single file...")
                # Load with external data and save as single file
                onnx_model = onnx.load(temp_output, load_external_data=True)
                onnx.save(onnx_model, output_path, save_as_external_data=False)
                
                # Remove temp files
                os.remove(temp_output)
                os.remove(temp_data_file)
            else:
                # No external data, just rename
                os.rename(temp_output, output_path)
            
            # Verify file was created and has reasonable size
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                if file_size < 100000:  # Less than 100KB - likely error
                    os.remove(output_path)
                    return jsonify({
                        "success": False,
                        "error": f"Model conversion produced invalid file (only {file_size} bytes)"
                    }), 500
                
                file_size_mb = file_size / (1024 * 1024)
                print(f"✓ Successfully converted to ONNX: {output_filename} ({file_size_mb:.2f} MB)")
                return jsonify({
                    "success": True,
                    "message": f"Successfully downloaded and converted {model_name} to ONNX ({file_size_mb:.2f} MB)",
                    "artifact": output_filename,
                    "format": "onnx",
                    "size_mb": round(file_size_mb, 2)
                })
            else:
                return jsonify({
                    "success": False,
                    "error": "Model conversion failed - ONNX file not created"
                }), 500
                
        except ImportError as ie:
            return jsonify({
                "success": False,
                "error": (
                    f"Optional libraries not installed for model download/convert: {str(ie)}. "
                    "This endpoint requires extra deps (timm/torch/onnx). "
                    "Install with: pip install onnx onnx-tool timm torch (may require a supported Python version)."
                )
            }), 500
        except Exception as download_error:
            import traceback
            traceback.print_exc()
            return jsonify({
                "success": False,
                "error": f"Failed to download/convert model: {str(download_error)}"
            }), 500
            
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ============================================================================

# ============================================================================
# DEPLOYMENT API
# ============================================================================

@app.route("/api/deploy", methods=["POST"])
def deploy():
    """API: Deploy model xuống BBB"""
    try:
        data = request.get_json()
        bbb_ip_raw = data.get("bbb_ip")
        model_name = data.get("model_name")
        max_energy = data.get("max_energy")
        force = bool(data.get("force", False))

        # Parse bbb_ip: handle cases where multiple IPs are provided (space-separated)
        # Extract first IPv4 address
        if not bbb_ip_raw:
            bbb_ip = None
        else:
            bbb_ip_parts = str(bbb_ip_raw).strip().split()
            # Find first IPv4 address (contains dots but not colons)
            bbb_ip = None
            for part in bbb_ip_parts:
                if '.' in part and ':' not in part:
                    bbb_ip = part
                    break
            # If no IPv4 found, use first part
            if not bbb_ip and bbb_ip_parts:
                bbb_ip = bbb_ip_parts[0]

        if not bbb_ip or not model_name:
            return jsonify({
                "success": False,
                "error": "bbb_ip và model_name là bắt buộc"
            }), 400

        try:
            energy_budget = float(max_energy) if max_energy not in (None, "") else None
        except (TypeError, ValueError):
            return jsonify({
                "success": False,
                "error": "max_energy phải là số"
            }), 400

        # Lấy thông tin model từ CSV
        model_info = analyzer.get_model_details(model_name)
        if model_info is None:
            return jsonify({
                "success": False,
                "error": f"Model '{model_name}' không tồn tại"
            }), 404

        model_energy = model_info.get("energy_avg_mwh")
        if (
            energy_budget is not None
            and model_energy is not None
            and model_energy > energy_budget
            and not force
        ):
            return jsonify({
                "success": False,
                "error": (
                    f"Model '{model_name}' cần {model_energy} mWh, "
                    f"vượt quá ngưỡng {energy_budget} mWh. "
                    "Điều chỉnh ngưỡng hoặc chấp nhận override để tiếp tục."
                ),
                "model_energy": model_energy,
                "energy_budget_mwh": energy_budget
            }), 409

        # Tìm file artifact tương ứng trong thư mục models
        # Ưu tiên tìm .tflite hoặc .onnx cho embedded devices
        artifact = None
        for preferred_ext in ['.tflite', '.onnx']:
            candidate = resolve_model_artifact(model_name + preferred_ext)
            if candidate:
                artifact = candidate
                print(f"Found artifact with {preferred_ext}: {artifact}")
                break
        
        # Fallback: tìm bất kỳ artifact nào
        if not artifact:
            artifact = resolve_model_artifact(model_name)
            if artifact:
                print(f"Found fallback artifact: {artifact}")
        
        if not artifact:
            # List available models to help debug
            available_models = []
            available_models = _list_local_artifacts()
            
            return jsonify({
                "success": False,
                "error": (
                    f"Không tìm thấy file artifact cho model '{model_name}'. "
                    f"Hãy import model local dưới dạng .tflite/.onnx hoặc upload qua giao diện deployment. "
                    f"Available models: {', '.join(available_models[:10])}"
                )
            }), 404
        
        # Kiểm tra xem artifact có phù hợp với embedded device không
        artifact_ext = artifact.lower().split('.')[-1]
        if artifact_ext not in ['tflite', 'onnx']:
            return jsonify({
                "success": False,
                "error": (
                    f"Model artifact '{artifact}' có format .{artifact_ext} "
                    "không được hỗ trợ bởi embedded devices. "
                    "Vui lòng convert sang .tflite hoặc .onnx trước khi deploy."
                )
            }), 400

        controller_base_url = _get_controller_base_url(request, prefer_connect_to_ip=bbb_ip)
        model_url = f"{controller_base_url}/models/{artifact}"
        log_manager.add_log(
            log_type="info",
            message=f"Model URL: {model_url}",
            metadata={"controller_base_url": controller_base_url, "bbb_ip": bbb_ip, "artifact": artifact}
        )

        # Gửi lệnh deploy xuống BBB
        deploy_url = f"http://{bbb_ip}:8000/deploy"
        payload = {
            "model_name": model_name,
            "model_url": model_url,
            "model_info": model_info
        }
        if energy_budget is not None:
            payload["energy_budget_mwh"] = energy_budget

        # Check if BBB is reachable first
        try:
            test_resp = requests.get(f"http://{bbb_ip}:8000/status", timeout=5)
            test_resp.raise_for_status()
        except requests.exceptions.RequestException as conn_err:
            error_msg = (
                f"Không thể kết nối tới BBB tại {bbb_ip}:8000. "
                f"Vui lòng kiểm tra: (1) Thiết bị có online không? "
                f"(2) Service ml-agent đã deploy chưa? "
                f"(3) IP có đúng không? Chi tiết: {str(conn_err)}"
            )
            log_manager.add_log(
                log_type="error",
                message=error_msg,
                metadata={
                    "model_name": model_name,
                    "device_ip": bbb_ip,
                    "error_type": "device_unreachable"
                }
            )
            return jsonify({
                "success": False,
                "error": error_msg
            }), 503

        resp = requests.post(deploy_url, json=payload, timeout=60)
        resp.raise_for_status()
        result = resp.json()
        
        # Log deployment success
        log_manager.add_log(
            log_type="success",
            message=f"Deploy thành công model '{model_name}' lên thiết bị {bbb_ip}",
            metadata={
                "model_name": model_name,
                "device_ip": bbb_ip,
                "energy_mwh": model_energy,
                "energy_budget_mwh": energy_budget,
                "artifact": artifact
            }
        )

        return jsonify({
            "success": True,
            "result": result,
            "model_info": model_info,
            "artifact": artifact,
            "energy_budget_mwh": energy_budget
        })

    except requests.exceptions.RequestException as e:
        error_msg = f"Lỗi kết nối BBB: {str(e)}"
        log_manager.add_log(
            log_type="error",
            message=error_msg,
            metadata={
                "model_name": data.get("model_name"),
                "device_ip": data.get("bbb_ip"),
                "error_type": "connection_error"
            }
        )
        return jsonify({
            "success": False,
            "error": error_msg
        }), 500
    except Exception as e:
        error_msg = str(e)
        log_manager.add_log(
            log_type="error",
            message=error_msg,
            metadata={
                "model_name": data.get("model_name"),
                "device_ip": data.get("bbb_ip"),
                "error_type": "general_error"
            }
        )
        return jsonify({
            "success": False,
            "error": error_msg
        }), 500


@app.route("/api/status", methods=["GET"])
def get_status():
    """
    API: Get device status
    Accepts: bbb_ip (IP address) or device_uuid (UUID for public URL)
    """
    bbb_ip = request.args.get("bbb_ip")
    device_uuid = request.args.get("device_uuid")
    
    if not bbb_ip and not device_uuid:
        return jsonify({"success": False, "error": "bbb_ip or device_uuid is required"}), 400

    # Try public URL first if UUID provided
    urls_to_try = []
    
    if device_uuid:
        public_url = get_device_public_url(device_uuid)
        if public_url:
            urls_to_try.append(f"{public_url}/status")
    
    if bbb_ip:
        urls_to_try.append(f"http://{bbb_ip}:8000/status")
    
    if not urls_to_try:
        return jsonify({
            "success": False,
            "error": "No valid endpoint available"
        }), 400
    
    # Try each URL
    for url in urls_to_try:
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            payload = resp.json()
            if isinstance(payload, dict):
                model_name = payload.get("model_name") or ((payload.get("model_info") or {}).get("model_name"))
                latest_energy_report = _get_latest_energy_report(device_id=device_uuid, model_name=model_name)
                if latest_energy_report:
                    payload["latest_energy_report"] = latest_energy_report
                latest_benchmark_report = _get_latest_benchmark_report(device_id=device_uuid, device_ip=bbb_ip, model_name=model_name)
                if latest_benchmark_report:
                    payload["latest_benchmark_report"] = latest_benchmark_report
                latest_fall_event = _get_latest_fall_event(device_id=device_uuid, device_ip=bbb_ip)
                if latest_fall_event:
                    payload["latest_fall_event"] = latest_fall_event
            return jsonify({
                "success": True,
                "status": payload
            })
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            continue
        except Exception:
            continue
    
    # All URLs failed
    return jsonify({
        "success": False,
        "error": "Unable to connect to device",
        "device_offline": True
    }), 200

@app.route("/api/device/metrics", methods=["GET"])
def get_device_metrics():
    """
    API: Get system metrics from device (CPU, Memory, Storage, Temperature)
    Accepts: bbb_ip (IP address) or device_uuid (UUID for public URL)
    """
    bbb_ip = request.args.get("bbb_ip")
    device_uuid = request.args.get("device_uuid")
    
    if not bbb_ip and not device_uuid:
        return jsonify({"success": False, "error": "bbb_ip or device_uuid is required"}), 400

    # Try public URL first if UUID provided
    urls_to_try = []
    
    if device_uuid:
        public_url = get_device_public_url(device_uuid)
        if public_url:
            urls_to_try.append(f"{public_url}/metrics")
    
    if bbb_ip:
        urls_to_try.append(f"http://{bbb_ip}:8000/metrics")
    
    if not urls_to_try:
        return jsonify({
            "success": False,
            "error": "No valid endpoint available"
        }), 400
    
    # Try each URL
    for url in urls_to_try:
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            return jsonify(resp.json())
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            continue
        except Exception:
            continue
    
    # All URLs failed
    return jsonify({
        "success": False,
        "error": f"Unable to connect to device",
        "device_offline": True
    }), 200


@app.route("/api/device/camera-frame", methods=["GET"])
def get_device_camera_frame():
    """
    Proxy a JPEG camera snapshot from the selected edge device.
    Accepts: bbb_ip (IP address) or device_uuid (UUID for public URL)
    """
    bbb_ip = request.args.get("bbb_ip")
    device_uuid = request.args.get("device_uuid")
    annotate = request.args.get("annotate", "1")

    if not bbb_ip and not device_uuid:
        return jsonify({"success": False, "error": "bbb_ip or device_uuid is required"}), 400

    urls_to_try = []
    if device_uuid:
        public_url = get_device_public_url(device_uuid)
        if public_url:
            urls_to_try.append(f"{public_url}/camera/snapshot?annotate={annotate}")
    if bbb_ip:
        urls_to_try.append(f"http://{bbb_ip}:8000/camera/snapshot?annotate={annotate}")

    if not urls_to_try:
        return jsonify({"success": False, "error": "No valid endpoint available"}), 400

    for url in urls_to_try:
        try:
            resp = requests.get(url, timeout=20)
            resp.raise_for_status()

            response = make_response(resp.content)
            response.headers["Content-Type"] = resp.headers.get("Content-Type", "image/jpeg")
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
            camera_source = resp.headers.get("X-Camera-Source")
            if camera_source:
                response.headers["X-Camera-Source"] = camera_source
            return response
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            continue
        except Exception:
            continue

    return jsonify({
        "success": False,
        "error": "Unable to capture camera frame from device",
        "device_offline": True
    }), 502


@app.route("/api/device/camera-stream", methods=["GET"])
def get_device_camera_stream():
    """Proxy a multipart MJPEG camera stream from the selected edge device."""
    bbb_ip = request.args.get("bbb_ip")
    device_uuid = request.args.get("device_uuid")
    annotate = request.args.get("annotate", "1")
    fps = request.args.get("fps", "5")
    camera_source = request.args.get("camera_source")

    if not bbb_ip and not device_uuid:
        return jsonify({"success": False, "error": "bbb_ip or device_uuid is required"}), 400

    base_urls = []
    if device_uuid:
        public_url = get_device_public_url(device_uuid)
        if public_url:
            base_urls.append(public_url)
    if bbb_ip:
        base_urls.append(f"http://{bbb_ip}:8000")

    if not base_urls:
        return jsonify({"success": False, "error": "No valid endpoint available"}), 400

    last_error = None
    params = {"annotate": annotate, "fps": fps}
    if camera_source:
        params["camera_source"] = camera_source

    for base in base_urls:
        try:
            upstream = requests.get(f"{base}/camera/stream", params=params, stream=True, timeout=10)
            upstream.raise_for_status()
            content_type = upstream.headers.get("Content-Type") or "multipart/x-mixed-replace"

            def generate():
                try:
                    for chunk in upstream.iter_content(chunk_size=8192):
                        if chunk:
                            yield chunk
                finally:
                    try:
                        upstream.close()
                    except Exception:
                        pass

            response = Response(stream_with_context(generate()), content_type=content_type)
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
            return response
        except Exception as exc:
            last_error = str(exc)
            continue

    return jsonify({
        "success": False,
        "error": f"Unable to stream camera from device: {last_error or 'unknown error'}",
        "device_offline": True
    }), 502


@app.route("/api/device/camera/fall-detect", methods=["POST"])
def run_device_camera_fall_detect():
    """Proxy a live fall-detection request to the selected edge device."""
    data = request.get_json(force=True, silent=True) or {}
    bbb_ip = data.get("bbb_ip")
    device_uuid = data.get("device_uuid")

    if not bbb_ip and not device_uuid:
        return jsonify({"success": False, "error": "bbb_ip or device_uuid is required"}), 400

    payload = {
        "duration_s": data.get("duration_s"),
        "max_frames": data.get("max_frames"),
        "camera_device": data.get("camera_device"),
        "camera_source": data.get("camera_source"),
        "fast_mode": data.get("fast_mode"),
    }

    urls_to_try = []
    # Prefer local endpoint first to keep camera control on the LAN path.
    if bbb_ip:
        urls_to_try.append(f"http://{bbb_ip}:8000/camera/fall-detect")
    if device_uuid:
        public_url = get_device_public_url(device_uuid)
        if public_url:
            urls_to_try.append(f"{public_url}/camera/fall-detect")

    if not urls_to_try:
        return jsonify({"success": False, "error": "No valid endpoint available"}), 400

    def _should_retry_no_frames(response_payload):
        if not isinstance(response_payload, dict):
            return False
        frames_analyzed = int(response_payload.get("frames_analyzed") or 0)
        if frames_analyzed > 0:
            return False
        details = response_payload.get("details") if isinstance(response_payload.get("details"), dict) else {}
        label = str(response_payload.get("label") or "").strip().lower()
        err_text = str(details.get("last_frame_error") or response_payload.get("error") or "").strip().lower()
        if label == "no_frames":
            return True
        return any(token in err_text for token in ("no frame", "contention", "unable to open camera", "camera frame read failed"))

    last_error = None
    last_failure_payload = None
    for url in urls_to_try:
        for attempt_idx in range(2):
            try:
                try:
                    request_payload = dict(payload)
                    if attempt_idx > 0:
                        # On retry, lengthen the detection window slightly so the device can recover camera ownership.
                        duration_s = _safe_float(request_payload.get("duration_s"))
                        if duration_s is not None:
                            request_payload["duration_s"] = round(max(duration_s, 2.5) + 1.0, 2)
                        # Retry in full window mode for better recovery when fast mode misses frames.
                        request_payload["fast_mode"] = False
                        max_frames = request_payload.get("max_frames")
                        try:
                            if max_frames is not None:
                                request_payload["max_frames"] = max(int(max_frames), 12)
                        except Exception:
                            pass

                    resp = requests.post(url, json=request_payload, timeout=60)
                except Exception:
                    raise

                try:
                    response_payload = resp.json()
                except Exception:
                    body_text = ""
                    try:
                        body_text = (resp.text or "").strip()
                    except Exception:
                        body_text = ""
                    if len(body_text) > 220:
                        body_text = body_text[:220] + "..."
                    err_msg = f"Invalid JSON from device ({resp.status_code})"
                    if body_text:
                        err_msg = f"{err_msg}: {body_text}"
                    response_payload = {"success": False, "error": err_msg}

                if resp.status_code < 400 and isinstance(response_payload, dict) and response_payload.get("success"):
                    if attempt_idx == 0 and _should_retry_no_frames(response_payload):
                        time.sleep(0.35)
                        continue

                    details = response_payload.get("details") or {}
                    if not isinstance(details, dict):
                        details = {}
                    frames_analyzed = int(response_payload.get("frames_analyzed") or 0)
                    response_payload["camera_ready"] = bool(frames_analyzed > 0 and response_payload.get("camera_ready") is not False)
                    if frames_analyzed <= 0 and not details.get("last_frame_error"):
                        details["last_frame_error"] = "No frames analyzed by device; likely camera contention between stream and detection"
                    response_payload["details"] = details

                    event_item = {
                        "event_id": f"fall-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}",
                        "timestamp": response_payload.get("timestamp") or datetime.now(timezone.utc).isoformat(),
                        "device_id": device_uuid,
                        "device_ip": bbb_ip,
                        "device_name": data.get("device_name"),
                        "model_name": response_payload.get("model"),
                        "camera_source": response_payload.get("camera_source"),
                        "camera_ready": response_payload.get("camera_ready"),
                        "fall_detected": bool(response_payload.get("fall_detected")),
                        "fall_score": _safe_float(response_payload.get("fall_score")),
                        "label": response_payload.get("label"),
                        "frames_analyzed": response_payload.get("frames_analyzed"),
                        "severity": "critical" if response_payload.get("fall_detected") else "normal",
                        "acknowledged": False,
                        "source": "controller_proxy",
                        "details": details,
                    }
                    _append_fall_event(event_item)
                    response_payload["event"] = event_item

                    return jsonify(response_payload), resp.status_code

                # Save device-side failure and optionally retry once on likely camera contention.
                failure_error = (response_payload.get("error") if isinstance(response_payload, dict) else None) or f"Device fall-detect failed ({resp.status_code})"
                if attempt_idx == 0 and _should_retry_no_frames(response_payload):
                    time.sleep(0.35)
                    continue

                last_error = failure_error
                last_failure_payload = {
                    "model": response_payload.get("model") if isinstance(response_payload, dict) else None,
                    "runtime": response_payload.get("runtime") if isinstance(response_payload, dict) else None,
                    "camera_source": (response_payload.get("camera_source") if isinstance(response_payload, dict) else None) or data.get("camera_source") or data.get("camera_device") or "/dev/video0",
                    "duration_s": response_payload.get("duration_s") if isinstance(response_payload, dict) else None,
                    "frames_requested": response_payload.get("frames_requested") if isinstance(response_payload, dict) else None,
                    "error": failure_error,
                }
                break
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
                last_error = str(exc)
                break
            except Exception as exc:
                last_error = str(exc)
                break

    # Convert the final failure into a structured event so the UI keeps updating.
    failed_payload = {
        "success": True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": (last_failure_payload or {}).get("model"),
        "runtime": (last_failure_payload or {}).get("runtime"),
        "camera_source": (last_failure_payload or {}).get("camera_source") or data.get("camera_source") or data.get("camera_device") or "/dev/video0",
        "camera_ready": False,
        "duration_s": (last_failure_payload or {}).get("duration_s"),
        "frames_requested": (last_failure_payload or {}).get("frames_requested"),
        "frames_analyzed": 0,
        "fall_detected": False,
        "fall_score": 0.0,
        "label": "no_frames",
        "error": (last_failure_payload or {}).get("error") or last_error or "Unable to run fall detection on device",
        "details": {
            "avg_fall_score": None,
            "best_frame": None,
            "fall_frame_ratio": None,
            "fall_frames": 0,
            "last_frame_error": (last_failure_payload or {}).get("error") or last_error or "Unable to run fall detection on device",
            "max_consecutive_fall_frames": 0,
        },
    }
    event_item = {
        "event_id": f"fall-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}",
        "timestamp": failed_payload["timestamp"],
        "device_id": device_uuid,
        "device_ip": bbb_ip,
        "device_name": data.get("device_name"),
        "model_name": failed_payload.get("model"),
        "camera_source": failed_payload.get("camera_source"),
        "camera_ready": failed_payload.get("camera_ready"),
        "fall_detected": False,
        "fall_score": 0.0,
        "label": "no_frames",
        "frames_analyzed": 0,
        "severity": "normal",
        "acknowledged": False,
        "source": "controller_proxy",
        "details": failed_payload.get("details") or {},
    }
    _append_fall_event(event_item)
    failed_payload["event"] = event_item
    return jsonify(failed_payload), 200


@app.route("/api/device/benchmark", methods=["POST"])
def run_device_benchmark():
    """Run inference benchmarking on the selected edge device and recalibrate prediction."""
    data = request.get_json(force=True, silent=True) or {}
    bbb_ip = data.get("bbb_ip")
    device_uuid = data.get("device_uuid")

    if not bbb_ip and not device_uuid:
        return jsonify({"success": False, "error": "bbb_ip or device_uuid is required"}), 400

    try:
        report_item = _benchmark_and_repredict_device(
            bbb_ip=bbb_ip,
            device_uuid=device_uuid,
            device_name=data.get("device_name"),
            device_type=data.get("device_type"),
            model_name=data.get("model_name"),
            model_info=data.get("model_info"),
            warmup_runs=int(data.get("warmup_runs", 5)),
            benchmark_runs=int(data.get("benchmark_runs", 30)),
        )
        return jsonify({
            "success": True,
            "report": report_item,
        })
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 502


@app.route("/api/device/start", methods=["POST"])
def start_device_model():
    """API: Kích hoạt lại model hiện tại trên thiết bị"""
    data = request.get_json(force=True, silent=True) or {}
    bbb_ip = data.get("bbb_ip")
    if not bbb_ip:
        return jsonify({"success": False, "error": "bbb_ip is required"}), 400

    try:
        resp = requests.post(f"http://{bbb_ip}:8000/start", timeout=20)
        resp.raise_for_status()
        payload = resp.json()

        log_manager.add_log(
            log_type="info",
            message=f"Re-run request sent to device {bbb_ip}",
            metadata={"device_ip": bbb_ip, "action": "start"}
        )

        # Try to fetch latest status after triggering start
        status_payload = None
        try:
            status_resp = requests.get(f"http://{bbb_ip}:8000/status", timeout=10)
            status_resp.raise_for_status()
            status_payload = status_resp.json()
        except Exception:
            status_payload = None

        return jsonify({"success": True, "result": payload, "status": status_payload})
    except requests.exceptions.RequestException as exc:
        error_msg = f"Unable to start model on device {bbb_ip}: {exc}"
        log_manager.add_log(
            log_type="error",
            message=error_msg,
            metadata={"device_ip": bbb_ip, "error_type": "start_error"}
        )
        return jsonify({"success": False, "error": error_msg}), 502

@app.route("/api/balena/fleets", methods=["GET"])
def get_balena_fleets():
    """API: Liệt kê các fleet/application trên Balena Cloud - chỉ fleet có device của user"""
    token = _get_balena_token()
    if not token:
        return jsonify({
            "success": False,
            "error": "BALENA_API_TOKEN not configured on controller.",
            "needs_token": True
        }), 400

    try:
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json"
        }
        
        # Get user's devices first - this gives us the fleets user actually owns
        device_url = f"{BALENA_API_BASE.rstrip('/')}/v6/device?$select=id&$expand=belongs_to__application($select=id,app_name,slug)&$top=500"
        device_resp = requests.get(device_url, headers=headers, timeout=BALENA_DEFAULT_TIMEOUT)
        device_resp.raise_for_status()
        device_payload = device_resp.json()
        devices = device_payload.get("d") or device_payload.get("value") or []
        
        # Extract unique fleets from devices
        fleet_map = {}
        for device in devices:
            app = device.get("belongs_to__application")
            if not app:
                continue
            
            # Handle both list and dict formats
            app_data = app[0] if isinstance(app, list) and app else app if isinstance(app, dict) else None
            if not app_data:
                continue
            
            fleet_id = app_data.get("id")
            fleet_name = app_data.get("app_name")
            fleet_slug = app_data.get("slug")
            
            if fleet_id and fleet_name and fleet_slug:
                fleet_map[fleet_id] = {
                    "id": fleet_id,
                    "name": fleet_name,
                    "slug": fleet_slug
                }
        
        # Convert to list and sort by name
        user_fleets = sorted(fleet_map.values(), key=lambda x: x["name"])
        
        return jsonify({"success": True, "fleets": user_fleets})
    except requests.HTTPError as e:
        status = e.response.status_code if e.response else 502
        try:
            msg = e.response.json()
        except Exception:
            msg = e.response.text if e.response else str(e)

        msg_str = str(msg).lower()
        print(f"[ERROR] /api/balena/fleets HTTPError: status={status}, msg={msg}")

        # Check if it's an auth error (either status 401/403 OR message contains "unauthorized"/"401")
        if status in (401, 403) or "unauthorized" in msg_str or "401" in msg_str or "403" in msg_str:
            print("[INFO] Detected auth error, returning 401 with needs_token=True")
            return jsonify({
                "success": False,
                "error": "Balena token expired or unauthorized. Please refresh your login.",
                "needs_token": True,
                "details": str(msg)
            }), 401

        return jsonify({"success": False, "error": f"Balena API error ({status}): {msg}"}), 502
    except requests.RequestException as e:
        return jsonify({"success": False, "error": f"Connection error: {str(e)}"}), 500

@app.route("/api/balena/devices", methods=["GET"])
def get_balena_devices():
    """API: Liệt kê các thiết bị Balena Cloud sẵn sàng deploy"""
    token = _get_balena_token()
    if not token:
        return jsonify({
            "success": False,
            "error": "BALENA_API_TOKEN chưa được cấu hình trên server controller.",
            "needs_token": True
        }), 400

    app_filter = request.args.get("app") or None
    online_only_flag = request.args.get("online_only", "false").lower() == "true"
    limit_param = request.args.get("limit")
    limit_value = None
    if limit_param:
        try:
            limit_value = int(limit_param)
        except ValueError:
            return jsonify({
                "success": False,
                "error": "limit phải là số"
            }), 400

    try:
        devices = fetch_balena_devices(
            app_slug=app_filter,
            online_only=online_only_flag,
            limit=limit_value or 50,
            token=token
        )
        return jsonify({
            "success": True,
            "devices": devices
        })
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except requests.HTTPError as e:
        status = e.response.status_code if e.response else 502
        try:
            msg = e.response.json()
        except Exception:
            msg = e.response.text if e.response else str(e)
        
        # Log chi tiết lỗi để debug
        msg_str = str(msg).lower()
        print(f"[ERROR] /api/balena/devices HTTPError: status={status}, msg={msg}")
        
        # Check if it's an auth error (either status 401/403 OR message contains "unauthorized"/"401")
        if status in (401, 403) or "unauthorized" in msg_str or "401" in msg_str or "403" in msg_str:
            print("[INFO] Detected auth error, returning 401 with needs_token=True")
            return jsonify({
                "success": False,
                "error": "Balena token expired or unauthorized. Please refresh your login.",
                "needs_token": True,
                "details": str(msg)
            }), 401

        return jsonify({
            "success": False,
            "error": f"Lỗi Balena API ({status}): {msg}",
            "details": str(msg)
        }), 502
    except requests.RequestException as e:
        print(f"[ERROR] Balena Connection Error: {str(e)}")
        return jsonify({
            "success": False,
            "error": f"Lỗi kết nối Balena: {str(e)}"
        }), 502
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/balena/devices/<device_uuid>/logs", methods=["GET"])
def get_device_logs(device_uuid):
    """API: Fetch device logs from Balena Cloud
    
    Note: Device logs require Supervisor API access (running on device) or
    Balena Cloud Enterprise features. For now, return mock data.
    """
    # Token validation
    token = os.getenv("BALENA_SESSION_TOKEN") or os.getenv("BALENA_API_TOKEN")
    if not token:
        return jsonify({
            "success": False,
            "error": "BALENA_SESSION_TOKEN or BALENA_API_TOKEN not configured",
            "needs_token": True
        }), 400
    
    # For now, return helpful message instead of failing
    # Real device logs require:
    # 1. Supervisor API access (device-local)
    # 2. Balena Cloud Enterprise plan
    # 3. Or using Balena CLI to fetch logs
    
    return jsonify({
        "success": True,
        "logs": [
            {
                "timestamp": "2026-01-04T02:00:00Z",
                "message": "⚠️ Device logs require Supervisor API access or Balena Enterprise plan",
                "isError": False,
                "isSystem": True
            },
            {
                "timestamp": "2026-01-04T02:00:01Z",
                "message": "To view logs, use: balena logs " + device_uuid,
                "isError": False,
                "isSystem": True
            },
            {
                "timestamp": "2026-01-04T02:00:02Z",
                "message": "Or enable Supervisor API on your device and configure BALENA_SUPERVISOR_ADDRESS",
                "isError": False,
                "isSystem": True
            }
        ],
        "device_uuid": device_uuid,
        "note": "Device logs API currently unavailable via Balena Cloud public API"
    })


@app.route("/api/balena/deploy", methods=["POST"])
def deploy_model():
    """API: Deploy model to Balena device"""
    token = os.getenv("BALENA_SESSION_TOKEN") or os.getenv("BALENA_API_TOKEN")
    if not token:
        return jsonify({
            "success": False,
            "error": "BALENA_API_TOKEN not configured"
        }), 400
    
    try:
        data = request.get_json()
        model_name = data.get("model_name")
        requested_artifact = data.get("artifact_file") or data.get("artifact")
        device_uuid = data.get("device_uuid")
        device_endpoint = data.get("device_endpoint")
        fleet = data.get("fleet")
        provided_model_info = data.get("model_info") if isinstance(data.get("model_info"), dict) else {}
        energy_budget = data.get("energy_budget_mwh")
        auto_benchmark_after_deploy = data.get("auto_benchmark_after_deploy", True)
        warmup_runs = data.get("warmup_runs", 5)
        benchmark_runs = data.get("benchmark_runs", 30)
        
        if not model_name:
            return jsonify({"success": False, "error": "model_name required"}), 400
        
        if not device_uuid and not device_endpoint:
            return jsonify({"success": False, "error": "device_uuid or device_endpoint required"}), 400
        
        device_display = device_endpoint or device_uuid[:8] if device_uuid else "unknown"
        
        # Log deployment attempt
        log_manager.add_log(
            "info",
            f"Starting deployment of '{model_name}' to device {device_display}",
            {
                "model_name": model_name,
                "device_uuid": device_uuid,
                "device_endpoint": device_endpoint,
                "fleet": fleet
            }
        )
        
        # Check if model file exists and get model info
        model_path = None
        model_ext = None

        if requested_artifact:
            safe_artifact = os.path.basename(str(requested_artifact))
            explicit_path = _find_local_artifact_path(safe_artifact)
            if explicit_path and os.path.exists(explicit_path):
                model_path = explicit_path
                model_ext = os.path.splitext(explicit_path)[1].lower()

        if not model_path:
            resolved = resolve_model_artifact(model_name)
            if resolved:
                resolved_path = _find_local_artifact_path(resolved)
                if resolved_path:
                    model_path = resolved_path
                    model_ext = os.path.splitext(model_path)[1].lower()

        if not model_path:
            base_name = _sanitize_model_base_name(model_name)
            for ext in [".tflite", ".onnx", ".pth", ".pt"]:
                potential_path = _find_local_artifact_path(f"{base_name}{ext}")
                if potential_path and os.path.exists(potential_path):
                    model_path = potential_path
                    model_ext = ext
                    break
        
        if not model_path:
            log_manager.add_log(
                "error",
                f"Không tìm thấy file model '{model_name}' cho thiết bị {device_display}",
                {
                    "model_name": model_name,
                    "device_uuid": device_uuid,
                    "device_endpoint": device_endpoint,
                    "error_type": "model_not_found"
                }
            )
            return jsonify({
                "success": False,
                "error": f"Model file not found for {model_name}"
            }), 404
        
        # Get model info from analyzer
        model_info = {}
        try:
            df = analyzer.df
            model_row = df[df['model'] == model_name]
            if not model_row.empty:
                row = model_row.iloc[0]
                model_info = {
                    "input_size": row.get('input_size', '3x224x224'),
                    "energy_avg_mwh": float(row.get('Energy (mWh)', 0)),
                    "latency_avg_s": float(row.get('Latency (s)', 0)),
                    "params": int(row.get('Params', 0)),
                        "flops": float(row.get('FLOPs (G)', 0))
                }
        except Exception as e:
            print(f"[WARNING] Could not get model info: {e}")

        if provided_model_info:
            model_info.update(provided_model_info)

        model_info.setdefault("model_name", model_name)
        if model_ext:
            model_info.setdefault("artifact_format", model_ext.lstrip("."))
        model_info.setdefault("artifact_file", os.path.basename(model_path) if model_path else requested_artifact)

        predicted_energy = None
        for key in ("predicted_energy_mwh", "energy_avg_mwh", "predicted_mwh"):
            predicted_energy = _safe_float(model_info.get(key))
            if predicted_energy is not None:
                break
        if predicted_energy is not None:
            model_info["predicted_energy_mwh"] = round(predicted_energy, 4)
            model_info.setdefault("energy_avg_mwh", round(predicted_energy, 4))
        
        # Deploy to device via HTTP
        # Build list of URLs to try (Public URL first, then local IP)
        urls_to_try = []
        public_url = get_device_public_url(device_uuid) if device_uuid else ""
        
        if public_url:
            urls_to_try.append(("public", f"{public_url}/deploy"))
        if device_endpoint:
            urls_to_try.append(("local", f"http://{device_endpoint}:8000/deploy"))
        
        if not urls_to_try:
            log_manager.add_log(
                "error",
                f"No device endpoint available (UUID: {device_uuid})",
                {"device_uuid": device_uuid}
            )
            return jsonify({
                "success": False,
                "error": "No device endpoint available (missing UUID and IP)"
            }), 400
        
        controller_base_url = _get_controller_base_url(request, prefer_connect_to_ip=(device_endpoint or None))
        model_url = f"{controller_base_url}/models/{os.path.basename(model_path)}"
        print(f"[INFO] Using controller base URL for model download: {controller_base_url}")
        
        # Prepare deployment payload for device
        deploy_payload = {
            "model_name": model_name,
            "model_url": model_url,
            "model_info": model_info
        }
        
        if energy_budget is not None:
            deploy_payload["energy_budget_mwh"] = energy_budget
        
        # Try each URL in order (Public URL first, then local IP)
        last_error = None
        used_url_type = None
        device_url = None
        
        for url_type, url in urls_to_try:
            device_url = url
            print(f"[INFO] Trying {url_type} URL for deployment: {device_url}")
            
            try:
                response = requests.post(
                    device_url,
                    json=deploy_payload,
                    timeout=60
                )
                response.raise_for_status()
                device_response = response.json()
                used_url_type = url_type
                
                log_manager.add_log(
                    "success",
                    f"Deploy thành công model '{model_name}' lên thiết bị {device_display} (via {url_type} URL)",
                    {
                        "model_name": model_name,
                        "device_uuid": device_uuid,
                        "device_endpoint": device_endpoint,
                        "model_path": model_path,
                        "fleet": fleet,
                        "device_response": device_response,
                        "used_url_type": url_type
                    }
                )
                
                benchmark_report = None
                benchmark_warning = None
                if auto_benchmark_after_deploy:
                    try:
                        status_payload = _wait_for_device_ready_for_benchmark(
                            bbb_ip=device_endpoint,
                            device_uuid=device_uuid,
                            timeout_s=90.0,
                        )
                        if isinstance(status_payload, dict) and str(status_payload.get("status") or "").lower() == "running":
                            benchmark_report = _benchmark_and_repredict_device(
                                bbb_ip=device_endpoint,
                                device_uuid=device_uuid,
                                device_name=device_display,
                                device_type=model_info.get("device_type"),
                                model_name=model_name,
                                model_info=model_info,
                                warmup_runs=int(warmup_runs),
                                benchmark_runs=int(benchmark_runs),
                            )
                        else:
                            benchmark_warning = "Device did not reach running state in time for automatic benchmark"
                    except Exception as benchmark_exc:
                        benchmark_warning = str(benchmark_exc)

                return jsonify({
                    "success": True,
                    "message": f"Model {model_name} deployed successfully",
                    "device_uuid": device_uuid,
                    "device_endpoint": device_endpoint,
                    "model_path": model_path,
                    "model_url": model_url,
                    "device_response": device_response,
                    "used_url_type": url_type,
                    "benchmark": benchmark_report,
                    "benchmark_warning": benchmark_warning
                })
                
            except requests.exceptions.RequestException as e:
                last_error = str(e)
                print(f"[WARN] Failed to deploy via {url_type} URL ({device_url}): {e}")
                # Try next URL
                continue
        
        # All URLs failed
        error_msg = f"Failed to deploy to device (tried {len(urls_to_try)} URLs). Last status: {last_error}"
        log_manager.add_log(
            "error",
            error_msg,
            {
                "model_name": model_name,
                "device_uuid": device_uuid,
                "device_endpoint": device_endpoint,
                "urls_tried": [url for _, url in urls_to_try],
                "last_error": last_error
            }
        )
        return jsonify({
            "success": False,
            "error": error_msg,
            "urls_tried": [url for _, url in urls_to_try]
        }), 500
        
    except Exception as e:
        error_msg = f"Deployment exception: {str(e)}"
        log_manager.add_log(
            "error",
            error_msg,
            {
                "model_name": data.get("model_name") if data else None,
                "error_type": "deployment_exception",
                "error_details": str(e)
            }
        )
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/predict-energy", methods=["POST"])
def predict_energy():
    """API: Dự đoán năng lượng tiêu thụ từ metadata model"""
    try:
        data = request.get_json(force=True, silent=False)
    except Exception:
        return jsonify({"success": False, "error": "Payload JSON không hợp lệ"}), 400

    if not data:
        return jsonify({"success": False, "error": "Payload trống"}), 400

    # Chấp nhận truyền 1 model (dict) hoặc danh sách models
    items = data if isinstance(data, list) else data.get("models") or [data]
    if not isinstance(items, list) or not items:
        return jsonify({
            "success": False,
            "error": "Payload phải là list hoặc chứa field 'models'"
        }), 400

    try:
        predictions = predictor_service.predict(items)
        
        # Load thresholds for categorization
        thresholds_path = os.path.join(ARTIFACTS_DIR, "energy_thresholds.json")
        thresholds_data = {}
        if os.path.exists(thresholds_path):
            with open(thresholds_path, 'r', encoding='utf-8') as f:
                thresholds_data = json.load(f)
        
        # Add energy category to each prediction
        for pred in predictions:
            if pred.get("prediction_mwh") is not None:
                device_type = pred.get("device_type", "unknown")
                device_lower = device_type.lower()
                if any(k in device_lower for k in ["jetson", "nano"]):
                    device_key = "jetson_nano"
                elif any(k in device_lower for k in ["raspberry", "rpi", "pi"]):
                    device_key = "raspberry_pi5"
                else:
                    device_key = None

                if device_key is None:
                    pred["energy_category"] = "unknown"
                    continue

                thresholds = thresholds_data.get(device_key, {})
                
                p25 = thresholds.get("p25", 50)
                p50 = thresholds.get("p50", 85)
                p75 = thresholds.get("p75", 150)
                
                energy = pred["prediction_mwh"]
                if energy < p25:
                    pred["energy_category"] = "excellent"
                elif energy < p50:
                    pred["energy_category"] = "good"
                elif energy < p75:
                    pred["energy_category"] = "acceptable"
                else:
                    pred["energy_category"] = "high"
        
        # Clean NaN/Inf values before JSON serialization
        import math
        def clean_value(v):
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                return None
            elif isinstance(v, dict):
                return {k: clean_value(val) for k, val in v.items()}
            elif isinstance(v, list):
                return [clean_value(val) for val in v]
            return v
        
        cleaned_predictions = [clean_value(p) for p in predictions]
        
        return jsonify({
            "success": True,
            "count": len(cleaned_predictions),
            "predictions": cleaned_predictions,
            "metadata": predictor_service.metadata,
        })
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        return jsonify({"success": False, "error": f"Lỗi khi dự đoán: {e}"}), 500


@app.route("/models/<path:filename>")
def download_model(filename):
    """Endpoint để BBB/Jetson/RPi tải model files"""
    try:
        safe_name = os.path.basename(filename)
        file_path = _find_local_artifact_path(safe_name)
        
        if not file_path or not os.path.exists(file_path):
            return jsonify({"error": f"Model file not found: {safe_name}"}), 404
        
        # Send with proper binary headers to avoid proxy/tunnel interference
        response = send_from_directory(
            os.path.dirname(file_path),
            os.path.basename(file_path),
            as_attachment=True,  # Force download
            mimetype='application/octet-stream'  # Binary file
        )
        
        # Add headers to prevent caching and ensure binary transfer
        response.headers['Content-Type'] = 'application/octet-stream'
        response.headers['Content-Disposition'] = f'attachment; filename={os.path.basename(file_path)}'
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        
        return response
    except Exception as e:
        return jsonify({"error": f"Failed to serve model: {str(e)}"}), 500


@app.route("/api/energy/thresholds", methods=["GET"])
def get_energy_thresholds():
    """API: Lấy energy thresholds từ percentile analysis"""
    try:
        thresholds_path = os.path.join(ARTIFACTS_DIR, "energy_thresholds.json")
        
        if not os.path.exists(thresholds_path):
            # Fallback to hardcoded values if file doesn't exist
            return jsonify({
                "success": True,
                "thresholds": {
                    "jetson_nano": {
                        "recommended_threshold": 50.0,
                        "p25": 50.0,
                        "p50": 85.0,
                        "unit": "mWh",
                        "source": "fallback_default"
                    },
                    "raspberry_pi5": {
                        "recommended_threshold": 30.0,
                        "p25": 30.0,
                        "p50": 50.0,
                        "unit": "mWh",
                        "source": "fallback_default"
                    }
                }
            })
        
        with open(thresholds_path, 'r', encoding='utf-8') as f:
            thresholds = json.load(f)
        
        return jsonify({
            "success": True,
            "thresholds": thresholds
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/energy/metadata", methods=["GET"])
def get_energy_metadata():
    """API: Lấy model metadata (MAPE, R², training info)"""
    try:
        metadata_path = os.path.join(ARTIFACTS_DIR, "device_specific_metadata.json")
        
        if not os.path.exists(metadata_path):
            # Fallback defaults
            return jsonify({
                "success": True,
                "metadata": {
                    "jetson_model": {
                        "metrics": {"cv_mape_eps": 21.5, "cv_r2": 0.93},
                        "model_name": "Extra Trees",
                        "log_transform_target": True,
                        "inference_note": "predict() returns log-scale; apply np.expm1() to get mWh"
                    },
                    "rpi5_model": {
                        "metrics": {"cv_mape_eps": 12.8, "cv_r2": 0.956},
                        "model_name": "Extra Trees",
                        "log_transform_target": True,
                        "inference_note": "predict() returns log-scale; apply np.expm1() to get mWh"
                    }
                },
                "source": "fallback"
            })
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        return jsonify({
            "success": True,
            "metadata": metadata,
            "source": "device_specific_metadata.json"
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/energy/report", methods=["POST"])
def report_measured_energy():
    """Persist measured energy samples from edge agents for later analysis/reporting."""
    data = request.get_json(force=True, silent=True) or {}
    actual_energy_mwh = _safe_float(data.get("actual_energy_mwh"))
    if actual_energy_mwh is None:
        return jsonify({"success": False, "error": "actual_energy_mwh is required"}), 400

    model_name = data.get("model_name") or "unknown"
    device_type = data.get("device_type") or "unknown"

    predicted_mwh = None
    model_info = data.get("model_info")
    if isinstance(model_info, dict):
        for key in ("predicted_energy_mwh", "energy_avg_mwh", "predicted_mwh"):
            predicted_mwh = _safe_float(model_info.get(key))
            if predicted_mwh is not None:
                break

    if predicted_mwh is None:
        predicted_mwh = _safe_float(data.get("predicted_mwh"))

    abs_error_mwh = None
    pct_error = None
    if predicted_mwh is not None and predicted_mwh > 0:
        abs_error_mwh = abs(actual_energy_mwh - predicted_mwh)
        pct_error = (abs_error_mwh / predicted_mwh) * 100.0

    item = {
        "timestamp": data.get("timestamp") or datetime.now(timezone.utc).isoformat(),
        "device_id": data.get("device_id"),
        "device_type": device_type,
        "model_name": model_name,
        "actual_energy_mwh": round(actual_energy_mwh, 4),
        "predicted_mwh": round(predicted_mwh, 4) if predicted_mwh is not None else None,
        "abs_error_mwh": round(abs_error_mwh, 4) if abs_error_mwh is not None else None,
        "pct_error": round(pct_error, 4) if pct_error is not None else None,
        "avg_power_mw": _safe_float(data.get("avg_power_mw")),
        "duration_s": _safe_float(data.get("duration_s")),
        "sensor_type": data.get("sensor_type") or data.get("meter_source") or "unknown",
        "source": data.get("source") or "edge_agent",
    }

    items = _load_energy_reports()
    items.append(item)
    items = items[-500:]
    _save_energy_reports(items)

    return jsonify({
        "success": True,
        "item": item,
        "stored": len(items),
    })


@app.route("/api/energy/recent", methods=["GET"])
def get_recent_energy_reports():
    limit = max(1, min(request.args.get("n", default=10, type=int), 100))
    items = _load_energy_reports()
    recent = list(reversed(items[-limit:]))
    return jsonify({
        "success": True,
        "count": len(recent),
        "items": recent,
    })


@app.route("/api/medical/fall-events", methods=["GET"])
def get_medical_fall_events():
    limit = max(1, min(request.args.get("limit", default=20, type=int), 100))
    device_uuid = request.args.get("device_uuid")
    device_ip = request.args.get("bbb_ip")
    items = _load_fall_events()

    normalized_uuid = str(device_uuid or "").strip()
    normalized_ip = str(device_ip or "").strip()

    filtered = []
    for item in reversed(items):
        # Prefer UUID matching when available; IPs can change or be stale.
        if normalized_uuid:
            if str(item.get("device_id") or "").strip() != normalized_uuid:
                continue
        elif normalized_ip:
            if str(item.get("device_ip") or "").strip() != normalized_ip:
                continue
        filtered.append(item)
        if len(filtered) >= limit:
            break

    return jsonify({
        "success": True,
        "count": len(filtered),
        "items": filtered,
    })


@app.route("/api/medical/fall-events/<event_id>/ack", methods=["POST"])
def acknowledge_medical_fall_event(event_id):
    items = _load_fall_events()
    updated_item = None
    for item in items:
        if str(item.get("event_id")) != str(event_id):
            continue
        item["acknowledged"] = True
        item["acknowledged_at"] = datetime.now(timezone.utc).isoformat()
        updated_item = item
        break

    if not updated_item:
        return jsonify({"success": False, "error": "Event not found"}), 404

    _save_fall_events(items)
    return jsonify({
        "success": True,
        "item": updated_item,
    })


@app.route("/api/models/popular", methods=["GET"])
def get_popular_models():
    """API: Lấy danh sách popular models phù hợp với edge devices"""
    try:
        popular_models_path = os.path.join(ARTIFACTS_DIR, "popular_models_metadata.json")
        
        if not os.path.exists(popular_models_path):
            return jsonify({
                "success": False,
                "error": "Popular models metadata not found. Please create popular_models_metadata.json"
            }), 404
        
        with open(popular_models_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        models = data.get("models", [])
        
        # Add model_downloaded field by checking if artifact exists
        for model in models:
            model_name = model.get("name", "")
            # Check ONLY for .tflite or .onnx artifacts (required for embedded devices)
            # Do NOT fallback to .pth files as they cannot be deployed
            artifact = None
            for ext in ['.tflite', '.onnx']:
                artifact = resolve_model_artifact(model_name + ext)
                if artifact:
                    break
            
            model["model_downloaded"] = artifact is not None
            if artifact:
                model["artifact_file"] = artifact
        
        # Filter by device if specified
        device_type = request.args.get("device")
        if device_type:
            device_type_lower = device_type.lower()
            filtered_models = []
            for model in models:
                recommended_devices = [d.lower() for d in model.get("recommended_devices", [])]
                if any(device_type_lower in d for d in recommended_devices):
                    filtered_models.append(model)
            models = filtered_models
        
        # Sort by params (lightweight first)
        models = sorted(models, key=lambda x: x.get("params_m", 0))
        
        return jsonify({
            "success": True,
            "models": models,
            "count": len(models),
            "description": data.get("description", "")
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/predict-energy-for-model", methods=["POST"])
def predict_energy_for_custom_model():
    """
    API: Dự đoán năng lượng cho model bất kỳ (popular model không có trong dataset)
    
    Request body:
    {
        "device_type": "jetson_nano" hoặc "raspberry_pi5",
        "model_name": "mobilenetv3_small_050",
        "params_m": 1.53,
        "gflops": 0.024,
        "gmacs": 0.012,
        "size_mb": 6.1,
        "latency_avg_s": 0.008,
        "throughput_iter_per_s": 125.0
    }
    
    Returns:
    {
        "success": true,
        "prediction": {
            "model_name": "mobilenetv3_small_050",
            "device_type": "jetson_nano",
            "predicted_energy_mwh": 45.2,
            "ci_lower_mwh": 38.1,
            "ci_upper_mwh": 52.3,
            "energy_category": "excellent",
            "recommendation": "deploy" hoặc "not_recommend",
            "reason": "Energy consumption is within excellent range for jetson_nano"
        }
    }
    """
    try:
        data = request.get_json(force=True, silent=False)
    except Exception:
        return jsonify({"success": False, "error": "Invalid JSON payload"}), 400

    if not data:
        return jsonify({"success": False, "error": "Empty payload"}), 400
    
    try:
        result, _, _ = _predict_energy_for_payload(data)
        return jsonify({
            "success": True,
            "prediction": result
        })
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": f"Prediction error: {e}"}), 500


@app.route("/api/analyze-model-file", methods=["POST"])
def analyze_model_file():
    """
    Upload a model file (.onnx or .tflite) and extract features for energy prediction.

    Multipart form fields:
        file        — model file (required)
        device_type — "jetson_nano" or "raspberry_pi5" (default: jetson_nano)
        input_h     — input height (default: 224)
        input_w     — input width  (default: 224)
        input_c     — input channels (default: 3)

    Returns extracted features ready to feed into /api/predict-energy-for-model.
    """
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded. Use multipart/form-data with field 'file'."}), 400

    f = request.files["file"]
    filename = f.filename or "model.onnx"

    # Allow practical ONNX/TFLite uploads while keeping a clear safety cap.
    MAX_BYTES = MODEL_ANALYZE_MAX_MB * 1024 * 1024
    file_bytes = f.read(MAX_BYTES + 1)
    if len(file_bytes) > MAX_BYTES:
        return jsonify({
            "success": False,
            "error": f"File too large (max {MODEL_ANALYZE_MAX_MB} MB)."
        }), 413

    device_type = request.form.get("device_type", "jetson_nano")
    try:
        ih = int(request.form.get("input_h", 224))
        iw = int(request.form.get("input_w", 224))
        ic = int(request.form.get("input_c", 3))
        input_shape = (1, ic, ih, iw)
    except (ValueError, TypeError):
        input_shape = None

    try:
        result = extract_model_features(
            file_bytes=file_bytes,
            filename=filename,
            device_type=device_type,
            input_shape=input_shape,
            jetson_csv=CSV_PATH,
            rpi5_csv=RPI5_CSV_PATH,
        )
        return jsonify({"success": True, "filename": filename, "features": result})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/models/upload", methods=["POST"])
def upload_custom_model():
    """Save a user-supplied ONNX/TFLite model into local artifacts for deployment."""
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"}), 400

    upload = request.files["file"]
    original_name = upload.filename or ""
    ext = os.path.splitext(original_name)[1].lower()
    if ext not in (".onnx", ".tflite"):
        return jsonify({
            "success": False,
            "error": "Only .onnx and .tflite files are supported"
        }), 400

    requested_name = request.form.get("model_name") or original_name
    model_base = _sanitize_model_base_name(requested_name)
    final_filename = f"{model_base}{ext}"
    target_path = os.path.join(MODEL_STORE_DIR, final_filename)
    os.makedirs(MODEL_STORE_DIR, exist_ok=True)
    upload.save(target_path)

    file_size_mb = round(os.path.getsize(target_path) / (1024 * 1024), 4)
    return jsonify({
        "success": True,
        "model_name": model_base,
        "artifact_file": final_filename,
        "path": target_path,
        "size_mb": file_size_mb,
        "format": ext.lstrip("."),
    })


@app.route("/api/models/recommended", methods=["GET"])
def get_recommended_models():
    """API: Lấy recommended models dựa trên device type và energy threshold"""
    try:
        device_type = request.args.get("device", "jetson_nano")
        limit = request.args.get("limit", type=int, default=30)
        
        # Get models filtered by device
        device_key = "jetson_nano" if "jetson" in device_type.lower() else "raspberry_pi5"
        
        # Lấy data từ CSV (đã có actual energy)
        if device_key == "jetson_nano":
            df = analyzer.df_jetson
        else:
            df = analyzer.df_rpi5
        
        # Convert to list và sort by actual energy
        models_list = []
        for _, row in df.iterrows():
            models_list.append({
                "name": row.get("model", "Unknown"),
                "params_m": round(row.get("params_m", 0), 2),
                "gflops": round(row.get("gflops", 0), 2),
                "gmacs": round(row.get("gmacs", 0), 2),
                "size_mb": round(row.get("size_mb", 0), 2),
                "latency_avg_s": round(row.get("latency_avg_s", 0), 4),
                "throughput_iter_per_s": round(row.get("throughput_iter_per_s", 0), 2),
                "predicted_energy": round(row.get("energy_avg_mwh", 0), 2),  # Use actual as prediction
                "actual_energy": round(row.get("energy_avg_mwh", 0), 2)
            })
        
        # Sort by energy (ascending)
        models_list.sort(key=lambda x: x["predicted_energy"])
        
        # Get thresholds
        thresholds_path = os.path.join(ARTIFACTS_DIR, "energy_thresholds.json")
        thresholds = {}
        if os.path.exists(thresholds_path):
            with open(thresholds_path, 'r', encoding='utf-8') as f:
                thresholds_data = json.load(f)
                thresholds = thresholds_data.get(device_key, {})
        
        p25 = thresholds.get("p25", 50)
        p50 = thresholds.get("p50", 85)
        p75 = thresholds.get("p75", 150)
        
        # Group by energy level
        result = {
            "excellent": [],  # < p25
            "good": [],       # p25 - p50
            "acceptable": [], # p50 - p75
            "high": []        # > p75
        }
        
        for model in models_list[:limit]:
            energy = model["predicted_energy"]
            if energy < p25:
                result["excellent"].append(model)
            elif energy < p50:
                result["good"].append(model)
            elif energy < p75:
                result["acceptable"].append(model)
            else:
                result["high"].append(model)
        
        return jsonify({
            "success": True,
            "device_type": device_type,
            "thresholds": {"p25": p25, "p50": p50, "p75": p75},
            "models": result,
            "total_analyzed": len(models_list)
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/logs", methods=["GET"])
def get_logs():
    """API: Lấy deployment logs"""
    try:
        limit = request.args.get("limit", type=int, default=50)
        log_type = request.args.get("type")  # filter by type: success, error, info, warning
        
        logs = log_manager.get_logs(limit=limit, log_type=log_type)
        stats = log_manager.get_deployment_stats()
        
        return jsonify({
            "success": True,
            "logs": logs,
            "stats": stats
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/logs/clear", methods=["POST"])
def clear_logs():
    """API: Xóa tất cả logs"""
    try:
        log_manager.clear_logs()
        return jsonify({
            "success": True,
            "message": "Đã xóa tất cả logs"
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/logs/export", methods=["GET"])
def export_logs():
    """API: Export logs ra JSON hoặc CSV"""
    try:
        format_type = request.args.get("format", "json").lower()
        
        if format_type == "json":
            data = log_manager.export_logs(format_type="json")
            return jsonify(data)
        elif format_type == "csv":
            csv_data = log_manager.export_logs(format_type="csv")
            from flask import Response
            return Response(
                csv_data,
                mimetype="text/csv",
                headers={"Content-Disposition": "attachment;filename=deployment_logs.csv"}
            )
        else:
            return jsonify({
                "success": False,
                "error": "Format không hợp lệ. Chỉ hỗ trợ 'json' hoặc 'csv'"
            }), 400
            
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/device/last-deployment", methods=["GET"])
def get_last_deployment():
    """API: Lấy log deploy thành công mới nhất (optional filter theo bbb_ip)"""
    device_ip = request.args.get("bbb_ip")
    log_entry = log_manager.get_latest_success(device_ip=device_ip) if log_manager else None
    if not log_entry:
        return jsonify({"success": False, "error": "No deployment found"}), 404
    return jsonify({"success": True, "log": log_entry})


@app.route("/api/device/status/<device_endpoint>", methods=["GET"])
def get_device_status(device_endpoint):
    """
    API: Proxy to get device status (avoids CORS)
    device_endpoint can be:
      - Device UUID (will fetch public URL)
      - IP address (will use http://<ip>:8000/status)
    """
    device_url = None
    try:
        fallback_ip = (request.args.get("bbb_ip") or "").strip()
        urls_to_try = []

        # Check if device_endpoint looks like a UUID (32+ hex chars)
        looks_like_uuid = (
            len(device_endpoint) >= 32
            and all(c in "0123456789abcdef" for c in device_endpoint.lower())
        )

        if looks_like_uuid:
            # It's a UUID, try to get public URL
            token = _get_balena_token()
            if not token:
                return jsonify({
                    "success": False,
                    "error": "BALENA_API_TOKEN is not configured; cannot resolve device Public URL",
                }), 503

            public_url = get_device_public_url(device_endpoint, token=token)
            if not public_url:
                return jsonify({
                    "success": False,
                    "error": "Public URL not available for this device (disabled or token lacks access)",
                }), 503

            urls_to_try.append(f"{public_url}/status")
        else:
            # Treat as IP address
            urls_to_try.append(f"http://{device_endpoint}:8000/status")

        if fallback_ip:
            fallback_url = f"http://{fallback_ip}:8000/status"
            if fallback_url not in urls_to_try:
                urls_to_try.append(fallback_url)

        response = None
        last_error = None
        for candidate_url in urls_to_try:
            device_url = candidate_url
            try:
                response = requests.get(device_url, timeout=5, headers={"Accept": "application/json"})
                break
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
                last_error = exc
                response = None
                continue

        if response is None:
            raise last_error or requests.exceptions.ConnectionError("No reachable device URL")

        if response.status_code == 200:
            try:
                payload = response.json()
                if isinstance(payload, dict):
                    model_name = payload.get("model_name") or ((payload.get("model_info") or {}).get("model_name"))
                    latest_energy_report = _get_latest_energy_report(device_id=device_endpoint if looks_like_uuid else None, model_name=model_name)
                    if latest_energy_report:
                        payload["latest_energy_report"] = latest_energy_report
            except ValueError:
                body_preview = (response.text or "")[:500]
                return jsonify({
                    "success": False,
                    "error": "Device returned non-JSON response",
                    "device_url": device_url,
                    "content_type": response.headers.get("Content-Type"),
                    "body_preview": body_preview,
                }), 502

            return jsonify({
                "success": True,
                "data": payload,
                "device_url": device_url,
            })

        body_preview = (response.text or "")[:500]
        return jsonify({
            "success": False,
            "error": f"HTTP {response.status_code}",
            "device_url": device_url,
            "body_preview": body_preview,
        }), response.status_code

    except requests.exceptions.Timeout:
        return jsonify({
            "success": False,
            "error": "Request timeout",
            "device_url": device_url,
        }), 504
    except requests.exceptions.SSLError as e:
        return jsonify({
            "success": False,
            "error": "SSL error",
            "details": str(e),
            "device_url": device_url,
        }), 502
    except requests.exceptions.ConnectionError:
        return jsonify({
            "success": False,
            "error": "Connection failed",
            "device_url": device_url,
        }), 503
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "device_url": device_url,
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
