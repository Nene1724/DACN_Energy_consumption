import os
import threading
from datetime import datetime
from flask import Flask, request, jsonify
import requests

import shutil
import time
import glob
import math
import traceback
try:
    import numpy as np
except ImportError:
    np = None

try:
    import tflite_runtime.interpreter as tflite
    TFLITE_AVAILABLE = True
except ImportError:
    tflite = None
    TFLITE_AVAILABLE = False

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ort = None
    ONNX_AVAILABLE = False

app = Flask(__name__)

# Persistent storage on BBB (allow override for local testing on Windows)
MODEL_DIR = os.getenv("MODEL_DIR_OVERRIDE", "/data/models")
os.makedirs(MODEL_DIR, exist_ok=True)
CURRENT_MODEL_PATH = os.path.join(MODEL_DIR, "current_model.tflite")  # Use .tflite extension
STATE_FILE_PATH = os.path.join(MODEL_DIR, "agent_state.json")
ENERGY_HISTORY_LIMIT = 40
CONTROLLER_URL = os.getenv("CONTROLLER_URL", "http://localhost:5000") # ss kq dd

# Used only for simulated fallback telemetry when the OS doesn't expose temp sensors.
_SIM_TEMP_START_TS = time.time()


def _create_energy_metrics(budget=None):
    return {
        "budget_mwh": budget,
        "latest_mwh": None,
        "avg_mwh": None,
        "status": "no_data",
        "history": []
    }


STATE = {
    "model_name": None,
    "model_info": None,
    "status": "idle",  # idle | downloading | ready | running | error
    "last_update": None,
    "error": None,
    "inference_active": False,
    "energy_metrics": _create_energy_metrics()
}

# In-memory runtime cache (TFLite)
LOADED_INTERPRETER = None
LOADED_MODEL_NAME = None
LOADED_INPUT_SIZE = None
LOADED_AT = None
MODEL_LOCK = threading.RLock()


def _unload_loaded_model(reason: str = ""):
    """Drop the in-memory model cache to avoid mixing weights across deploys."""
    global LOADED_INTERPRETER, LOADED_MODEL_NAME, LOADED_INPUT_SIZE, LOADED_AT
    with MODEL_LOCK:
        if LOADED_INTERPRETER is not None:
            log(f"Unloading cached model{': ' + reason if reason else ''}")
        LOADED_INTERPRETER = None
        LOADED_MODEL_NAME = None
        LOADED_INPUT_SIZE = None
        LOADED_AT = None


def reset_energy_metrics(budget=None):
    """Reset live energy monitoring container"""
    STATE["energy_metrics"] = _create_energy_metrics(budget)


def record_energy_sample(energy_mwh, power_mw=None, latency_s=None, note=None, timestamp=None):
    """Track live energy measurements and halt inference if budget is exceeded"""
    if timestamp is None:
        timestamp = datetime.utcnow().isoformat() + "Z"

    metrics = STATE.get("energy_metrics") or _create_energy_metrics()
    history = metrics.get("history", [])

    sample = {
        "timestamp": timestamp,
        "energy_mwh": round(float(energy_mwh), 4)
    }
    if power_mw is not None:
        sample["power_mw"] = round(float(power_mw), 4)
    if latency_s is not None:
        sample["latency_s"] = round(float(latency_s), 5)
    if note:
        sample["note"] = str(note)

    history.append(sample)
    if len(history) > ENERGY_HISTORY_LIMIT:
        history = history[-ENERGY_HISTORY_LIMIT:]
    metrics["history"] = history

    metrics["latest_mwh"] = sample["energy_mwh"]
    metrics["avg_mwh"] = round(
        sum(entry["energy_mwh"] for entry in history) / len(history),
        4
    )

    budget = metrics.get("budget_mwh")
    over_budget = budget is not None and sample["energy_mwh"] > budget
    metrics["status"] = "over_budget" if over_budget else "ok"

    state_update = {
        "energy_metrics": metrics
    }

    if over_budget and STATE.get("status") == "running":
        message = (
            f"Energy budget exceeded: "
            f"{sample['energy_mwh']} mWh > budget {budget} mWh"
        )
        log(message)
        state_update.update(
            status="error",
            inference_active=False,
            error=message
        )

    set_state(**state_update)
    return not over_budget


def save_state_to_disk():
    """Persist current state to disk"""
    import json
    try:
        state_data = {
            "model_name": STATE.get("model_name"),
            "model_info": STATE.get("model_info"),
            "status": STATE.get("status"),
            "last_update": STATE.get("last_update")
        }
        with open(STATE_FILE_PATH, "w") as f:
            json.dump(state_data, f)
    except Exception as e:
        log(f"Failed to save state: {e}")


def load_state_from_disk():
    """Restore state from disk on startup"""
    import json
    try:
        if os.path.exists(STATE_FILE_PATH):
            with open(STATE_FILE_PATH, "r") as f:
                state_data = json.load(f)
            STATE["model_name"] = state_data.get("model_name")
            STATE["model_info"] = state_data.get("model_info")
            # Don't restore status, start as ready if model exists
            if STATE["model_name"] and os.path.exists(CURRENT_MODEL_PATH):
                STATE["status"] = "ready"
                log(f"Restored model state: {STATE['model_name']}")
            else:
                STATE["status"] = "idle"
            return True
    except Exception as e:
        log(f"Failed to load state: {e}")
    return False


def set_state(**kwargs):
    """Update agent state"""
    for k, v in kwargs.items():
        STATE[k] = v
    STATE["last_update"] = datetime.utcnow().isoformat() + "Z"
    # Persist important state changes to disk
    if "model_name" in kwargs or "model_info" in kwargs or "status" in kwargs:
        save_state_to_disk()


def log(message):
    """Simple logging"""
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def _read_text(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return None


def _cpu_percent_from_proc(sample_interval_s=0.15):
    """Compute CPU utilization (%) from /proc/stat without psutil."""
    def read_cpu_ticks():
        raw = _read_text("/proc/stat")
        if not raw:
            return None
        first = raw.splitlines()[0].split()
        if len(first) < 5 or first[0] != "cpu":
            return None
        ticks = [int(x) for x in first[1:]]
        total = sum(ticks)
        idle = ticks[3] + (ticks[4] if len(ticks) > 4 else 0)
        return total, idle

    a = read_cpu_ticks()
    if a is None:
        return None
    time.sleep(max(0.0, float(sample_interval_s)))
    b = read_cpu_ticks()
    if b is None:
        return None

    total_delta = b[0] - a[0]
    idle_delta = b[1] - a[1]
    if total_delta <= 0:
        return None
    usage = 100.0 * (1.0 - (idle_delta / total_delta))
    return max(0.0, min(100.0, usage))


def _memory_from_proc():
    raw = _read_text("/proc/meminfo")
    if not raw:
        return None
    mem = {}
    for line in raw.splitlines():
        parts = line.split(":", 1)
        if len(parts) != 2:
            continue
        key = parts[0].strip()
        value = parts[1].strip().split()[0]
        try:
            mem[key] = int(value) * 1024  # kB -> bytes
        except Exception:
            continue
    total = mem.get("MemTotal")
    available = mem.get("MemAvailable")
    if total is None or available is None:
        return None
    used = max(0, total - available)
    return {
        "total_bytes": int(total),
        "available_bytes": int(available),
        "used_bytes": int(used),
        "used_percent": round((used / total) * 100.0, 2) if total else None,
    }


def _storage_for_path(path="/"):
    try:
        usage = shutil.disk_usage(path)
        total = int(usage.total)
        used = int(usage.used)
        free = int(usage.free)
        return {
            "path": path,
            "total_bytes": total,
            "used_bytes": used,
            "free_bytes": free,
            "used_percent": round((used / total) * 100.0, 2) if total else None,
        }
    except Exception:
        return None


def _temperature_c():
    """Best-effort temperature (°C) from sysfs.

    Works across many embedded Linux targets without external dependencies.
    Returns None if no readable sensor is present in the container.
    """

    def to_celsius(raw_value: str):
        if raw_value is None:
            return None
        try:
            v = float(str(raw_value).strip())
        except Exception:
            return None

        # Most sysfs temps are in milli-degrees C.
        if v > 1000:
            v = v / 1000.0
        elif v > 200:
            # defensive: some boards report 42000 without crossing 1000 check due to parsing quirks
            v = v / 1000.0

        # Sanity range for CPU/SOC temp.
        if v < -20 or v > 150:
            return None
        return round(v, 1)

    # 1) Prefer thermal zones that look like CPU/SOC.
    zone_dirs = []
    for pattern in (
        "/sys/class/thermal/thermal_zone*",
        "/sys/devices/virtual/thermal/thermal_zone*",
    ):
        zone_dirs.extend(glob.glob(pattern))
    zone_dirs = sorted(set(zone_dirs))
    preferred_candidates = []
    fallback_candidates = []
    for zone in zone_dirs:
        zone_type = _read_text(os.path.join(zone, "type")) or ""
        temp_path = os.path.join(zone, "temp")
        if not os.path.exists(temp_path):
            continue
        if any(k in zone_type.lower() for k in ("cpu", "soc", "x86_pkg_temp", "processor", "core")):
            preferred_candidates.append(temp_path)
        else:
            fallback_candidates.append(temp_path)

    for path in preferred_candidates + fallback_candidates:
        t = to_celsius(_read_text(path))
        if t is not None:
            return t

    # 2) Fallback: hwmon temp inputs (common on many boards).
    hwmon_paths = []
    for pattern in (
        "/sys/class/hwmon/hwmon*/temp*_input",
        "/sys/devices/virtual/hwmon/hwmon*/temp*_input",
        "/sys/devices/platform/*/hwmon/hwmon*/temp*_input",
        # BBB and many embedded kernels expose hwmon deeper under platform devices (e.g. ocp/*/bandgap/*/hwmon/hwmon0/temp1_input)
        "/sys/devices/platform/**/hwmon/hwmon*/temp*_input",
        "/sys/bus/i2c/devices/*/hwmon/hwmon*/temp*_input",
    ):
        if "**" in pattern:
            hwmon_paths.extend(glob.glob(pattern, recursive=True))
        else:
            hwmon_paths.extend(glob.glob(pattern))
    hwmon_paths = sorted(set(hwmon_paths))
    for path in hwmon_paths:
        t = to_celsius(_read_text(path))
        if t is not None:
            return t

    # 3) Fallback: IIO temperature channels (some kernels expose die temp here).
    iio_paths = []
    for pattern in (
        "/sys/bus/iio/devices/iio:device*/in_temp*_input",
        "/sys/bus/iio/devices/iio:device*/in_temp_input",
        "/sys/devices/platform/**/iio:device*/in_temp*_input",
        "/sys/devices/platform/**/iio:device*/in_temp_input",
    ):
        if "**" in pattern:
            iio_paths.extend(glob.glob(pattern, recursive=True))
        else:
            iio_paths.extend(glob.glob(pattern))
    iio_paths = sorted(set(iio_paths))
    for path in iio_paths:
        t = to_celsius(_read_text(path))
        if t is not None:
            return t

    return None


def _simulated_temperature_c(now_ts=None):
    """Generate a smooth, continuously varying temperature (°C).

    This is a fallback used only when no real temperature sensor is visible.
    """
    t = float(now_ts if now_ts is not None else time.time()) - float(_SIM_TEMP_START_TS)

    # Smooth periodic variation + a secondary wobble.
    base = 52.0
    primary_period_s = 120.0
    secondary_period_s = 23.0
    v = (
        base
        + 6.5 * math.sin((2.0 * math.pi * t) / primary_period_s)
        + 1.2 * math.sin((2.0 * math.pi * t) / secondary_period_s)
    )

    # Clamp to a reasonable range.
    v = max(35.0, min(80.0, v))
    return round(v, 1)


def _temperature_debug_info(limit=30):
    """Return discoverability info for temperature sensors (for troubleshooting)."""

    def safe_listdir(path, max_items=80):
        try:
            items = os.listdir(path)
            items = sorted(items)
            if len(items) > max_items:
                items = items[:max_items] + [f"... (+{len(items) - max_items} more)"]
            return {"exists": True, "items": items}
        except FileNotFoundError:
            return {"exists": False, "items": []}
        except Exception as e:
            return {"exists": True, "items": [], "error": str(e)}

    def filtered_mounts():
        mounts_raw = _read_text("/proc/mounts")
        if not mounts_raw:
            return []
        results = []
        for line in mounts_raw.splitlines():
            parts = line.split()
            if len(parts) < 3:
                continue
            src, mnt, fstype = parts[0], parts[1], parts[2]
            if mnt in ("/sys", "/proc") or mnt.startswith("/sys/"):
                results.append({"source": src, "mountpoint": mnt, "type": fstype})
        return results

    info = {
        "fs": {
            "sys_exists": os.path.exists("/sys"),
            "proc_exists": os.path.exists("/proc"),
            "mounts": filtered_mounts(),
            "dir_sys": safe_listdir("/sys"),
            "dir_sys_class": safe_listdir("/sys/class"),
            "dir_sys_class_thermal": safe_listdir("/sys/class/thermal"),
            "dir_sys_class_hwmon": safe_listdir("/sys/class/hwmon"),
        },
        "thermal_zone_paths": [],
        "hwmon_paths": [],
        "iio_temp_paths": [],
        "readable_samples": [],
        "errors": [],
    }

    # Thermal zones
    zone_dirs = []
    for pattern in (
        "/sys/class/thermal/thermal_zone*",
        "/sys/devices/virtual/thermal/thermal_zone*",
    ):
        zone_dirs.extend(glob.glob(pattern))
    zone_dirs = sorted(set(zone_dirs))
    for zone in zone_dirs[:limit]:
        zone_type = _read_text(os.path.join(zone, "type"))
        temp_path = os.path.join(zone, "temp")
        info["thermal_zone_paths"].append({
            "zone": zone,
            "type": zone_type,
            "temp_path": temp_path,
            "exists": os.path.exists(temp_path),
            "readable": os.access(temp_path, os.R_OK) if os.path.exists(temp_path) else False,
        })

    # HWMON
    hwmon_paths = []
    for pattern in (
        "/sys/class/hwmon/hwmon*/temp*_input",
        "/sys/devices/virtual/hwmon/hwmon*/temp*_input",
        "/sys/devices/platform/*/hwmon/hwmon*/temp*_input",
        "/sys/devices/platform/**/hwmon/hwmon*/temp*_input",
        "/sys/bus/i2c/devices/*/hwmon/hwmon*/temp*_input",
    ):
        if "**" in pattern:
            hwmon_paths.extend(glob.glob(pattern, recursive=True))
        else:
            hwmon_paths.extend(glob.glob(pattern))
    hwmon_paths = sorted(set(hwmon_paths))
    for p in hwmon_paths[:limit]:
        info["hwmon_paths"].append({
            "path": p,
            "exists": os.path.exists(p),
            "readable": os.access(p, os.R_OK) if os.path.exists(p) else False,
        })

    # Try to read a few samples
    candidates = []
    for z in info["thermal_zone_paths"]:
        if z.get("exists"):
            candidates.append(z.get("temp_path"))
    for h in info["hwmon_paths"]:
        if h.get("exists"):
            candidates.append(h.get("path"))

    # IIO temperature inputs
    iio_paths = []
    for pattern in (
        "/sys/bus/iio/devices/iio:device*/in_temp*_input",
        "/sys/bus/iio/devices/iio:device*/in_temp_input",
        "/sys/devices/platform/**/iio:device*/in_temp*_input",
        "/sys/devices/platform/**/iio:device*/in_temp_input",
    ):
        if "**" in pattern:
            iio_paths.extend(glob.glob(pattern, recursive=True))
        else:
            iio_paths.extend(glob.glob(pattern))
    iio_paths = sorted(set(iio_paths))
    for p in iio_paths[:limit]:
        info["iio_temp_paths"].append({
            "path": p,
            "exists": os.path.exists(p),
            "readable": os.access(p, os.R_OK) if os.path.exists(p) else False,
        })
    for p in iio_paths:
        if os.path.exists(p):
            candidates.append(p)

    for p in candidates[:10]:
        try:
            raw = _read_text(p)
            info["readable_samples"].append({"path": p, "raw": raw})
        except Exception as e:
            info["errors"].append({"path": p, "error": str(e)})

    return info


@app.route("/metrics/debug", methods=["GET"])
def metrics_debug():
    """Debug endpoint: show what temperature sensors are visible to the container."""
    real_temp = _temperature_c()
    temp = real_temp if real_temp is not None else _simulated_temperature_c()
    source = "sysfs" if real_temp is not None else "simulated"
    return jsonify({
        "success": True,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "temperature_c": temp,
        "temperature_source": source,
        "temperature_debug": _temperature_debug_info(),
    })


@app.route("/status", methods=["GET"])
def status():
    """Return current agent status with runtime capabilities"""
    response = dict(STATE)
    response["runtime_capabilities"] = {
        "tflite_available": TFLITE_AVAILABLE,
        "numpy_available": np is not None
    }
    return jsonify(response)


@app.route("/metrics", methods=["GET"])
def metrics():
    """Lightweight device metrics without native deps (works on armv7)."""
    cpu = _cpu_percent_from_proc()
    mem = _memory_from_proc()
    storage = _storage_for_path("/")
    real_temp = _temperature_c()
    temp = real_temp if real_temp is not None else _simulated_temperature_c()
    temp_source = "sysfs" if real_temp is not None else "simulated"

    return jsonify({
        "success": True,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "cpu": {"percent": round(cpu, 2) if cpu is not None else None},
        "memory": mem,
        "storage": storage,
        "temperature_c": temp,
        "temperature_source": temp_source,
        "agent": {
            "status": STATE.get("status"),
            "model_name": STATE.get("model_name"),
            "inference_active": STATE.get("inference_active"),
        },
    })


@app.route("/deploy", methods=["POST"])
def deploy():
    """
    Deploy new model from web controller
    
    JSON payload:
    {
        "model_name": "mobilenetv3_small_100",
        "model_url": "http://192.168.137.1:5000/models/mobilenetv3_small_100.pth",
        "model_info": {...}
    }
    """
    data = request.get_json(force=True)
    model_name = data.get("model_name")
    model_url = data.get("model_url")
    model_info = data.get("model_info", {})
    energy_budget = data.get("energy_budget_mwh")

    try:
        energy_budget_value = float(energy_budget) if energy_budget is not None else None
    except (TypeError, ValueError):
        return jsonify({"error": "energy_budget_mwh must be numeric"}), 400

    if not model_name or not model_url:
        return jsonify({"error": "model_name and model_url are required"}), 400

    # Check supported model formats
    model_ext = model_url.lower().split('.')[-1]
    if model_ext not in ['tflite', 'onnx']:
        return jsonify({"error": "This device only supports .tflite and .onnx artifacts"}), 400
    
    if model_ext == 'onnx' and not ONNX_AVAILABLE:
        return jsonify({"error": "ONNX Runtime not installed on this device"}), 500

    try:
        # Always drop any cached model before swapping artifacts on disk.
        _unload_loaded_model("deploy")

        # Stop old model if running
        if STATE.get("status") == "running" or STATE.get("inference_active"):
            log(f"Stopping current model before deployment")
            set_state(status="ready", inference_active=False)
        
        log(f"Starting deployment: {model_name}")
        reset_energy_metrics(energy_budget_value)
        set_state(
            status="downloading",
            model_name=model_name,
            model_info=model_info,
            error=None,
            inference_active=False,
            energy_metrics=STATE["energy_metrics"]
        )

        # Download model from controller
        log(f"Downloading from: {model_url}")
        resp = requests.get(model_url, stream=True, timeout=60)
        resp.raise_for_status()

        tmp_path = CURRENT_MODEL_PATH + ".tmp"
        with open(tmp_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        # Replace current model (automatically removes old model)
        if os.path.exists(CURRENT_MODEL_PATH):
            log(f"Removing old model: {CURRENT_MODEL_PATH}")
        os.replace(tmp_path, CURRENT_MODEL_PATH)
        file_size = os.path.getsize(CURRENT_MODEL_PATH)
        
        log(f"Model downloaded: {file_size} bytes")
        log(f"Old model removed, new model deployed")
        set_state(
            status="ready",
            model_name=model_name,
            model_info=model_info,
            error=None,
            energy_metrics=STATE["energy_metrics"]
        )

        # Auto-start model inference in background
        log(f"Auto-starting new model: {model_name}")
        threading.Thread(target=start_model_inference, daemon=True).start()

        return jsonify({
            "message": "Deployment successful",
            "model_name": model_name,
            "model_path": CURRENT_MODEL_PATH,
            "model_size_bytes": file_size,
            "model_info": model_info,
            "energy_budget_mwh": energy_budget_value
        })

    except Exception as e:
        log(f"Deployment error: {str(e)}")
        set_state(status="error", error=str(e))
        return jsonify({"error": str(e)}), 500


@app.route("/start", methods=["POST"])
def start_inference():
    """Manually start model inference"""
    # Allow retrying start after a previous error as long as a model exists.
    if STATE.get("status") not in ("ready", "error"):
        return jsonify({"error": "Model not ready"}), 400

    if not STATE.get("model_name") or not os.path.exists(CURRENT_MODEL_PATH):
        return jsonify({"error": "No deployed model artifact found"}), 400
    
    threading.Thread(target=start_model_inference, daemon=True).start()
    return jsonify({"message": "Starting inference..."})


@app.route("/stop", methods=["POST"])
def stop_inference():
    """Stop model inference"""
    if STATE["status"] == "running":
        set_state(status="ready", inference_active=False)
        log("Inference stopped")
        return jsonify({"message": "Inference stopped"})
    
    return jsonify({"error": "Inference not running"}), 400


@app.route("/predict", methods=["POST"])
def predict():
    """
    Inference endpoint: runs a forward pass using TFLite (cached after /deploy or /start).
    """
    if STATE.get("status") not in ("running", "ready"):
        return jsonify({"error": "Model not running"}), 400

    _ensure_numpy()

    try:
        with MODEL_LOCK:
            if LOADED_INTERPRETER is None or LOADED_INPUT_SIZE is None:
                return jsonify({"error": "Model not loaded in memory"}), 400
            interpreter = LOADED_INTERPRETER
            input_size = LOADED_INPUT_SIZE

        c, h, w = input_size
        input_details = interpreter.get_input_details()[0]
        dtype = input_details["dtype"]
        shape = input_details["shape"]

        try:
            if len(shape) == 4 and shape[1] == h and shape[2] == w:
                dummy_input = np.random.rand(*shape).astype(dtype)
            else:
                dummy_input = np.random.rand(1, h, w, c).astype(dtype)
        except Exception:
            dummy_input = np.random.rand(1, h, w, c).astype(np.float32)

        interpreter.set_tensor(input_details["index"], dummy_input)
        interpreter.invoke()
        output_details = interpreter.get_output_details()[0]
        output = interpreter.get_tensor(output_details["index"])

        flat = output.flatten()
        sample = flat[:5].tolist() if flat.size else []
        summary = {
            "dtype": str(output.dtype),
            "shape": list(output.shape),
            "sample": [float(x) for x in sample],
        }

        return jsonify({
            "success": True,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "model": STATE.get("model_name"),
            "runtime": "tflite",
            "input_size": [c, h, w],
            "summary": summary,
        })
    except Exception as exc:
        err_msg = f"Inference error: {exc}"
        log(err_msg)
        traceback.print_exc()
        _unload_loaded_model("predict failed")
        set_state(status="error", error=str(exc), inference_active=False)
        return jsonify({"error": f"Inference error: {exc}"}), 500


def _parse_input_size(raw):
    """
    Parse input_size from metadata (e.g., "3x224x224" or [3,224,224])
    Returns (channels, height, width)
    """
    if not raw:
        return (3, 224, 224)
    if isinstance(raw, (list, tuple)):
        vals = [int(x) for x in raw if isinstance(x, (int, float, str)) and str(x).strip().lstrip("-").isdigit()]
        if len(vals) == 4:
            # NCHW -> CHW
            return int(vals[1]), int(vals[2]), int(vals[3])
        if len(vals) >= 3:
            return int(vals[0]), int(vals[1]), int(vals[2])
    if isinstance(raw, dict):
        # common forms
        c = raw.get("channels") or raw.get("c") or 3
        h = raw.get("height") or raw.get("h") or raw.get("size") or 224
        w = raw.get("width") or raw.get("w") or raw.get("size") or 224
        try:
            return int(c), int(h), int(w)
        except Exception:
            return (3, 224, 224)
    if isinstance(raw, str):
        parts = raw.lower().replace("x", " ").replace(",", " ").split()
        nums = [int(p) for p in parts if p.isdigit()]
        if len(nums) >= 3:
            return nums[0], nums[1], nums[2]
    return (3, 224, 224)


def _ensure_numpy():
    if np is None:
        raise RuntimeError("numpy is required but not available on this device")


def _load_tflite_model(model_path, input_size):
    """Load a .tflite model and run a warmup inference."""
    if not TFLITE_AVAILABLE:
        raise RuntimeError("tflite_runtime is not available on this device")
    _ensure_numpy()

    interpreter = tflite.Interpreter(model_path=model_path, num_threads=1)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    c, h, w = input_size

    # Build dummy input with correct dtype/shape
    dtype = input_details["dtype"]
    shape = input_details["shape"]
    try:
        # If model expects NHWC
        if len(shape) == 4 and shape[1] == h and shape[2] == w:
            dummy = np.random.rand(*shape).astype(dtype)
        else:
            dummy = np.random.rand(1, h, w, c).astype(dtype)
    except Exception:
        dummy = np.random.rand(1, h, w, c).astype(np.float32)

    interpreter.set_tensor(input_details["index"], dummy)
    interpreter.invoke()  # warmup
    return interpreter, input_details

# ss kq
# -------- Energy measurement utilities (powercap-based) --------
def _find_powercap_energy_file():
    base = "/sys/class/powercap"
    if not os.path.isdir(base):
        return None
    for root, dirs, files in os.walk(base):
        if "energy_uj" in files:
            return os.path.join(root, "energy_uj")
    return None


def _read_uint(path: str):
    try:
        with open(path, "r") as f:
            return int(f.read().strip())
    except Exception:
        return None


def _run_single_inference_locked(interpreter, input_size):
    c, h, w = input_size
    input_details = interpreter.get_input_details()[0]
    dtype = input_details["dtype"]
    shape = input_details["shape"]
    try:
        if len(shape) == 4 and shape[1] == h and shape[2] == w:
            dummy = np.random.rand(*shape).astype(dtype)
        else:
            dummy = np.random.rand(1, h, w, c).astype(dtype)
    except Exception:
        dummy = np.random.rand(1, h, w, c).astype(np.float32)
    interpreter.set_tensor(input_details["index"], dummy)
    interpreter.invoke()


def measure_energy_during_inference(duration_s: float = 10.0):
    energy_file = _find_powercap_energy_file()
    if energy_file is None:
        return {
            "success": False,
            "error": "powercap energy_uj not available",
            "sensor_type": "unknown"
        }

    if STATE.get("status") not in ("running", "ready"):
        return {"success": False, "error": "Model not ready/running"}

    with MODEL_LOCK:
        interpreter = LOADED_INTERPRETER
        input_size = LOADED_INPUT_SIZE
    if interpreter is None or input_size is None:
        return {"success": False, "error": "Interpreter not loaded"}

    start_uj = _read_uint(energy_file)
    start_ts = time.time()
    iters = 0
    while (time.time() - start_ts) < duration_s:
        try:
            with MODEL_LOCK:
                _run_single_inference_locked(interpreter, input_size)
            iters += 1
        except Exception:
            break
    end_uj = _read_uint(energy_file)
    end_ts = time.time()

    if start_uj is None or end_uj is None or end_uj < start_uj:
        return {"success": False, "error": "Invalid energy_uj readings"}

    delta_uj = end_uj - start_uj
    elapsed_s = max(end_ts - start_ts, 1e-6)
    energy_mwh = float(delta_uj) / 3_600_000.0
    avg_power_mw = (energy_mwh * 3600.0) / elapsed_s

    return {
        "success": True,
        "sensor_type": "powercap",
        "duration_s": elapsed_s,
        "iterations": iters,
        "actual_energy_mwh": energy_mwh,
        "avg_power_mw": avg_power_mw
    }


@app.route("/measure_energy", methods=["POST"])
def measure_energy_endpoint():
    data = request.get_json(silent=True) or {}
    duration_s = float(data.get("duration_s", 10.0))
    report = measure_energy_during_inference(duration_s=duration_s)
    if not report.get("success"):
        return jsonify(report), 400

    try:
        controller = data.get("controller_url") or CONTROLLER_URL
        if controller:
            payload = {
                "device_type": "bbb",
                "device_id": os.getenv("BALENA_DEVICE_UUID"),
                "model_name": STATE.get("model_name"),
                "actual_energy_mwh": report.get("actual_energy_mwh"),
                "avg_power_mw": report.get("avg_power_mw"),
                "duration_s": report.get("duration_s"),
                "sensor_type": report.get("sensor_type"),
            }
            try:
                requests.post(f"{controller.rstrip('/')}/api/energy/report", json=payload, timeout=10)
            except Exception as post_err:
                report["post_warning"] = f"Failed to post to controller: {post_err}"
    except Exception as e:
        report["post_warning"] = str(e)

    return jsonify(report)


@app.route("/measure_energy_fnb58", methods=["POST"])
def measure_energy_fnb58():
    """Đo năng lượng bằng FNB58 USB tester qua cổng serial.
    
    Request body:
    {
      "fnb58_port": "/dev/ttyUSB0",  # hoặc "COM3" trên Windows
      "duration_s": 30,
      "auto_detect": true  # nếu true, tự tìm FNB58
    }
    
    Trả về năng lượng đo được và tự post về server.
    """
    try:
        import sys
        sys.path.insert(0, os.path.dirname(__file__))
        from fnb58_reader import FNB58Reader, detect_fnb58_port
    except ImportError as e:
        return jsonify({
            "success": False,
            "error": f"FNB58 reader không khả dụng: {e}. Cài: pip install pyserial"
        }), 500
    
    data = request.get_json(silent=True) or {}
    duration_s = float(data.get("duration_s", 30.0))
    port = data.get("fnb58_port")
    auto_detect = data.get("auto_detect", True)
    
    if not port and auto_detect:
        log(f"[FNB58] Cố tìm tự động...")
        port = detect_fnb58_port()
        if port:
            log(f"[FNB58] Tìm thấy trên {port}")
    
    if not port:
        return jsonify({
            "success": False,
            "error": "Không tìm thấy cổng FNB58. Chỉ định fnb58_port hoặc bật auto_detect"
        }), 400
    
    try:
        log(f"[FNB58] Đo {duration_s}s trên {port}...")
        reader = FNB58Reader(port)
        
        if not reader.start():
            return jsonify({
                "success": False,
                "error": f"Không kết nối được {port}: {reader.connection_error}"
            }), 400
        
        time.sleep(duration_s)
        result = reader.stop()
        
        if not result.get('success'):
            return jsonify({
                "success": False,
                "error": f"Đo FNB58 lỗi: {result.get('error')}"
            }), 400
        
        response = {
            "success": True,
            "sensor_type": "fnb58",
            "port": port,
            "duration_s": duration_s,
            "samples_count": result.get('samples_count'),
            "actual_energy_mwh": result.get('total_energy_mwh'),
            "avg_power_mw": result.get('avg_power_mw'),
            "last_values": result.get('last_values')
        }
        
        try:
            controller = data.get("controller_url") or CONTROLLER_URL
            if controller and response.get('actual_energy_mwh'):
                payload = {
                    "device_type": "bbb",
                    "device_id": os.getenv("BALENA_DEVICE_UUID"),
                    "model_name": STATE.get("model_name"),
                    "actual_energy_mwh": response['actual_energy_mwh'],
                    "avg_power_mw": response.get('avg_power_mw'),
                    "duration_s": duration_s,
                    "sensor_type": "fnb58",
                }
                try:
                    requests.post(f"{controller.rstrip('/')}/api/energy/report", json=payload, timeout=10)
                    response["posted_to_controller"] = True
                except Exception as post_err:
                    response["post_warning"] = f"Lỗi post về server: {post_err}"
        except Exception as e:
            response["post_warning"] = str(e)
        
        return jsonify(response)
    
    except Exception as e:
        log(f"[FNB58] Lỗi: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route("/telemetry", methods=["POST"])
def telemetry():
    """
    Receive live telemetry (energy, power, latency) from BBB sensors
    Example payload:
    {
        "energy_mwh": 45.3,
        "power_mw": 180,
        "latency_s": 0.123,
        "note": "iio-sensor-sample",
        "timestamp": "2024-05-20T12:00:00Z"
    }
    """
    data = request.get_json(force=True)
    if data is None:
        return jsonify({"error": "Missing telemetry payload"}), 400

    energy = data.get("energy_mwh")
    if energy is None:
        return jsonify({"error": "energy_mwh is required"}), 400

    power = data.get("power_mw")
    latency = data.get("latency_s")
    note = data.get("note")
    timestamp = data.get("timestamp")

    within_budget = record_energy_sample(
        energy,
        power_mw=power,
        latency_s=latency,
        note=note,
        timestamp=timestamp
    )

    message = "Telemetry ingested"
    if not within_budget:
        message = "Telemetry ingested - energy budget exceeded, inference halted"

    return jsonify({
        "message": message,
        "energy_metrics": STATE["energy_metrics"],
        "status": STATE["status"]
    })


def start_model_inference():
    """
    Start model inference (runs in background thread)
    
    Loads TFLite model from CURRENT_MODEL_PATH and runs warmup forward.
    Keeps interpreter in memory for /predict.
    """
    try:
        global LOADED_INTERPRETER, LOADED_MODEL_NAME, LOADED_INPUT_SIZE, LOADED_AT

        log("Starting model inference...")
        model_name = STATE.get("model_name")
        model_info = STATE.get("model_info") or {}
        if not model_name:
            raise RuntimeError("No model_name present in state")
        if not os.path.exists(CURRENT_MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {CURRENT_MODEL_PATH}")

        if not CURRENT_MODEL_PATH.endswith(".tflite"):
            raise RuntimeError("Only .tflite models are supported on this device")

        input_size = _parse_input_size(model_info.get("input_size"))
        with MODEL_LOCK:
            # Ensure any previous cache is dropped before loading.
            _unload_loaded_model("start")
            interpreter, _ = _load_tflite_model(CURRENT_MODEL_PATH, input_size)
            LOADED_INTERPRETER = interpreter
            LOADED_MODEL_NAME = model_name
            LOADED_INPUT_SIZE = input_size
            LOADED_AT = datetime.utcnow().isoformat() + "Z"

        set_state(status="running", inference_active=True, error=None)

        log(f"Model '{STATE['model_name']}' is now running (runtime: tflite)")
        log(f"Estimated energy draw: {STATE['model_info'].get('energy_avg_mwh', 'N/A')} mWh")
        budget = STATE.get("energy_metrics", {}).get("budget_mwh")
        if budget is not None:
            log(f"Energy budget assigned: {budget} mWh")

    except Exception as e:
        err_msg = f"Error starting inference: {e}"
        log(err_msg)
        traceback.print_exc()
        _unload_loaded_model("start failed")
        set_state(status="error", error=str(e), inference_active=False)


if __name__ == "__main__":
    log("BBB ML Agent starting...")
    log(f"Model directory: {MODEL_DIR}")
    
    # Restore state from previous session
    load_state_from_disk()
    
    # Check if there's an existing model
    if os.path.exists(CURRENT_MODEL_PATH):
        size = os.path.getsize(CURRENT_MODEL_PATH)
        log(f"Found existing model: {size} bytes")
        if STATE.get("model_name"):
            log(f"Current model: {STATE['model_name']}")
    
    # Start Flask server
    app.run(host="0.0.0.0", port=8000)
