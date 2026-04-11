import os
import threading
from datetime import datetime, timezone
from urllib.parse import urlparse
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

# Try TensorFlow Lite first (full TensorFlow package), fallback to tflite_runtime
try:
    import tensorflow as tf
    tflite = tf.lite
    TFLITE_AVAILABLE = True
    print("[INFO] Using TensorFlow Lite from tensorflow package")
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite
        TFLITE_AVAILABLE = True
        print("[INFO] Using tflite_runtime package")
    except ImportError:
        tflite = None
        TFLITE_AVAILABLE = False
        print("[WARNING] Neither TensorFlow nor tflite_runtime available")

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ort = None
    ONNX_AVAILABLE = False

try:
    from fnb58_reader import FNB58Reader as FNB58SerialReader, detect_fnb58_port
except ImportError:
    FNB58SerialReader = None

    def detect_fnb58_port():
        return None

try:
    from fnb58_usb_reader import FNB58USBReader, detect_fnb58_usb
except ImportError:
    FNB58USBReader = None

    def detect_fnb58_usb():
        return None

try:
    from fnb58_hidraw_reader import FNB58HIDRawReader, detect_fnb58_hidraw
except ImportError:
    FNB58HIDRawReader = None

    def detect_fnb58_hidraw():
        return None

try:
    from fnb58_exporter_reader import FNB58ExporterReader
except ImportError:
    FNB58ExporterReader = None

app = Flask(__name__)

# Persistent storage on Raspberry Pi (allow override for local testing on Windows)
MODEL_DIR = os.getenv("MODEL_DIR_OVERRIDE", "/data/models")
os.makedirs(MODEL_DIR, exist_ok=True)
CURRENT_MODEL_BASENAME = os.path.join(MODEL_DIR, "current_model")
STATE_FILE_PATH = os.path.join(MODEL_DIR, "agent_state.json")
ENERGY_HISTORY_LIMIT = 40
CONTROLLER_URL = os.getenv("CONTROLLER_URL", "")
FNB58_AUTO_START = (os.getenv("FNB58_AUTO_START", "true").strip().lower() in ("1", "true", "yes", "on"))
FNB58_PORT = (os.getenv("FNB58_PORT") or "").strip()
FNB58_RETRY_INTERVAL_S = max(2.0, float(os.getenv("FNB58_RETRY_INTERVAL_S", "5")))

# Used only for simulated fallback telemetry when the OS doesn't expose temp sensors.
_SIM_TEMP_START_TS = time.time()


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _safe_float(value):
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _current_model_path_for_ext(ext: str):
    return CURRENT_MODEL_BASENAME + ext


def _current_model_candidates():
    return [
        _current_model_path_for_ext(".tflite"),
        _current_model_path_for_ext(".onnx"),
    ]


def _get_active_model_path():
    artifact_path = STATE.get("artifact_path")
    if artifact_path and os.path.exists(artifact_path):
        return artifact_path
    for path in _current_model_candidates():
        if os.path.exists(path):
            return path
    return None


def _remove_other_model_files(keep_path=None):
    for path in _current_model_candidates():
        if keep_path and os.path.abspath(path) == os.path.abspath(keep_path):
            continue
        if os.path.exists(path):
            try:
                os.remove(path)
            except FileNotFoundError:
                pass


def _create_energy_metrics(budget=None, predicted_mwh=None):
    return {
        "budget_mwh": budget,
        "predicted_mwh": predicted_mwh,
        "latest_mwh": None,
        "avg_mwh": None,
        "latest_abs_error_mwh": None,
        "latest_error_pct": None,
        "mape_pct": None,
        "status": "no_data",
        "history": []
    }


def _create_meter_metrics():
    return {
        "status": "idle",
        "connected": False,
        "port": None,
        "transport": None,
        "updated_at": None,
        "voltage_v": None,
        "current_a": None,
        "power_w": None,
        "power_mw": None,
        "total_energy_wh": None,
        "baseline_energy_wh": None,
        "measured_energy_mwh": None,
        "last_values": {},
        "error": None,
    }


STATE = {
    "model_name": None,
    "artifact_path": None,
    "runtime": None,
    "model_info": None,
    "controller_url": CONTROLLER_URL,
    "status": "idle",  # idle | downloading | ready | running | error
    "last_update": None,
    "error": None,
    "inference_active": False,
    "energy_metrics": _create_energy_metrics(),
    "meter_metrics": _create_meter_metrics(),
}

# In-memory runtime cache
LOADED_INTERPRETER = None
LOADED_ONNX_SESSION = None
LOADED_RUNTIME = None
LOADED_MODEL_NAME = None
LOADED_MODEL_PATH = None
LOADED_INPUT_SIZE = None
LOADED_INPUT_LAYOUT = None
LOADED_INPUT_META = None
LOADED_AT = None
MODEL_LOCK = threading.RLock()
FNB58_MONITOR = {"reader": None, "thread": None}
FNB58_RETRY_LOOP = {"thread": None}


def _unload_loaded_model(reason: str = ""):
    """Drop the in-memory model cache to avoid mixing weights across deploys."""
    global LOADED_INTERPRETER, LOADED_ONNX_SESSION, LOADED_RUNTIME
    global LOADED_MODEL_NAME, LOADED_MODEL_PATH, LOADED_INPUT_SIZE, LOADED_INPUT_LAYOUT, LOADED_INPUT_META, LOADED_AT
    with MODEL_LOCK:
        if LOADED_INTERPRETER is not None or LOADED_ONNX_SESSION is not None:
            log(f"Unloading cached model{': ' + reason if reason else ''}")
        LOADED_INTERPRETER = None
        LOADED_ONNX_SESSION = None
        LOADED_RUNTIME = None
        LOADED_MODEL_NAME = None
        LOADED_MODEL_PATH = None
        LOADED_INPUT_SIZE = None
        LOADED_INPUT_LAYOUT = None
        LOADED_INPUT_META = None
        LOADED_AT = None


def reset_energy_metrics(budget=None, predicted_mwh=None):
    """Reset live energy monitoring container"""
    if predicted_mwh is None:
        model_info = STATE.get("model_info") or {}
        predicted_mwh = _safe_float(
            model_info.get("predicted_energy_mwh")
            or model_info.get("energy_avg_mwh")
            or model_info.get("predicted_mwh")
        )
    STATE["energy_metrics"] = _create_energy_metrics(budget, predicted_mwh)


def record_energy_sample(energy_mwh, power_mw=None, latency_s=None, note=None, timestamp=None, source=None):
    """Track live energy measurements and halt inference if budget is exceeded"""
    if timestamp is None:
        timestamp = _now_iso()

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
    if source:
        sample["source"] = str(source)

    history.append(sample)
    if len(history) > ENERGY_HISTORY_LIMIT:
        history = history[-ENERGY_HISTORY_LIMIT:]
    metrics["history"] = history

    metrics["latest_mwh"] = sample["energy_mwh"]
    metrics["avg_mwh"] = round(
        sum(entry["energy_mwh"] for entry in history) / len(history),
        4
    )

    predicted = _safe_float(metrics.get("predicted_mwh"))
    if predicted is not None and predicted > 0:
        abs_error = abs(sample["energy_mwh"] - predicted)
        error_pct = (abs_error / predicted) * 100.0
        sample["predicted_mwh"] = round(predicted, 4)
        sample["abs_error_mwh"] = round(abs_error, 4)
        sample["error_pct"] = round(error_pct, 4)
        metrics["latest_abs_error_mwh"] = round(abs_error, 4)
        metrics["latest_error_pct"] = round(error_pct, 4)
        error_samples = [entry.get("error_pct") for entry in history if entry.get("error_pct") is not None]
        if error_samples:
            metrics["mape_pct"] = round(sum(error_samples) / len(error_samples), 4)

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


def _derive_energy_payload(payload):
    energy_mwh = _safe_float(payload.get("energy_mwh"))
    power_mw = _safe_float(payload.get("power_mw"))
    power_w = _safe_float(payload.get("power_w"))
    duration_s = _safe_float(payload.get("duration_s"))
    duration_ms = _safe_float(payload.get("duration_ms"))
    energy_wh = _safe_float(payload.get("energy_wh"))
    energy_mah = _safe_float(payload.get("energy_mah"))
    voltage_v = _safe_float(payload.get("voltage_v"))

    parsed_from = "energy_mwh"
    if power_mw is None and power_w is not None:
        power_mw = power_w * 1000.0
    if energy_mwh is None and energy_wh is not None:
        energy_mwh = energy_wh * 1000.0
        parsed_from = "energy_wh"
    if energy_mwh is None and energy_mah is not None and voltage_v is not None:
        energy_mwh = energy_mah * voltage_v
        parsed_from = "energy_mah_voltage_v"
    if energy_mwh is None:
        if duration_s is None and duration_ms is not None:
            duration_s = duration_ms / 1000.0
        if power_mw is not None and duration_s is not None:
            energy_mwh = power_mw * (duration_s / 3600.0)
            parsed_from = "power_duration"
    return energy_mwh, power_mw, parsed_from


def save_state_to_disk():
    """Persist current state to disk"""
    import json
    try:
        state_data = {
            "model_name": STATE.get("model_name"),
            "artifact_path": STATE.get("artifact_path"),
            "runtime": STATE.get("runtime"),
            "model_info": STATE.get("model_info"),
            "controller_url": STATE.get("controller_url"),
            "status": STATE.get("status"),
            "inference_active": STATE.get("inference_active", False),
            "last_update": STATE.get("last_update"),
        }
        with open(STATE_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump(state_data, f)
    except Exception as e:
        log(f"Failed to save state: {e}")


def load_state_from_disk():
    """Restore state from disk on startup"""
    import json
    try:
        if os.path.exists(STATE_FILE_PATH):
            with open(STATE_FILE_PATH, "r", encoding="utf-8") as f:
                state_data = json.load(f)
            STATE["model_name"] = state_data.get("model_name")
            STATE["artifact_path"] = state_data.get("artifact_path")
            STATE["runtime"] = state_data.get("runtime")
            STATE["model_info"] = state_data.get("model_info")
            STATE["controller_url"] = state_data.get("controller_url") or CONTROLLER_URL
            was_running = state_data.get("inference_active", False)
            
            # Don't restore status, start as ready if model exists
            artifact_path = _get_active_model_path()
            if STATE["model_name"] and artifact_path and os.path.exists(artifact_path):
                STATE["artifact_path"] = artifact_path
                STATE["status"] = "ready"
                log(f"Restored model state: {STATE['model_name']}")
                
                # Auto-start if inference was active before shutdown
                if was_running:
                    log(f"Auto-starting model after restart: {STATE['model_name']}")
                    threading.Thread(target=_auto_start_model_on_boot, daemon=True).start()
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
    STATE["last_update"] = _now_iso()
    # Persist important state changes to disk
    if (
        "model_name" in kwargs or
        "artifact_path" in kwargs or
        "runtime" in kwargs or
        "model_info" in kwargs or
        "controller_url" in kwargs or
        "status" in kwargs or
        "inference_active" in kwargs
    ):
        save_state_to_disk()


def _auto_start_model_on_boot():
    """Auto-start model after device reboot if it was running before"""
    import time
    time.sleep(3)  # Wait for system to fully boot
    
    try:
        model_path = _get_active_model_path()
        if not model_path or not os.path.exists(model_path):
            log("Auto-start skipped: No model file found")
            return
        
        model_name = STATE.get("model_name")
        if not model_name:
            log("Auto-start skipped: No model name in state")
            return
        
        log(f"Auto-starting model: {model_name}")
        start_model_inference()
        log("Auto-start completed successfully")
    except Exception as e:
        log(f"Auto-start failed: {e}")
        set_state(status="error", error=str(e), inference_active=False)


def _update_meter_snapshot(result=None, *, status=None, connected=None, port=None, error=None):
    meter = STATE.get("meter_metrics") or _create_meter_metrics()
    if status is not None:
        meter["status"] = status
    if connected is not None:
        meter["connected"] = connected
    if port is not None:
        meter["port"] = port
    if error is not None:
        meter["error"] = error
    meter["updated_at"] = _now_iso()

    if result:
        last_values = result.get("last_values") or {}
        meter["last_values"] = dict(last_values)
        if result.get("transport"):
            meter["transport"] = result.get("transport")
        meter["voltage_v"] = _safe_float(last_values.get("voltage_v"))
        meter["current_a"] = _safe_float(last_values.get("current_a"))
        meter["power_w"] = _safe_float(last_values.get("power_w"))
        meter["power_mw"] = (meter["power_w"] * 1000.0) if meter["power_w"] is not None else None
        meter["total_energy_wh"] = _safe_float(last_values.get("energy_wh"))

        if meter.get("baseline_energy_wh") is None and meter["total_energy_wh"] is not None:
            meter["baseline_energy_wh"] = meter["total_energy_wh"]

        baseline_wh = _safe_float(meter.get("baseline_energy_wh"))
        if baseline_wh is not None and meter["total_energy_wh"] is not None:
            measured_energy_mwh = max(meter["total_energy_wh"] - baseline_wh, 0.0) * 1000.0
            meter["measured_energy_mwh"] = round(measured_energy_mwh, 4)
            record_energy_sample(
                measured_energy_mwh,
                power_mw=meter["power_mw"],
                note=last_values,
                timestamp=meter["updated_at"],
                source=result.get("meter_source") or "fnb58_meter",
            )

    STATE["meter_metrics"] = meter


def _reset_meter_baseline():
    meter = STATE.get("meter_metrics") or _create_meter_metrics()
    total_wh = _safe_float(meter.get("total_energy_wh"))
    meter["baseline_energy_wh"] = total_wh
    meter["measured_energy_mwh"] = 0.0 if total_wh is not None else None
    meter["updated_at"] = _now_iso()
    STATE["meter_metrics"] = meter


def _refresh_meter_from_reader(reader):
    try:
        while getattr(reader, "running", False):
            result = {
                "last_values": getattr(reader, "last_values", {}) or {},
                "samples_count": len(getattr(reader, "samples", []) or []),
                "meter_source": getattr(reader, "transport", None) or "fnb58_meter",
                "transport": getattr(reader, "transport", None),
            }
            _update_meter_snapshot(result, status="connected", connected=True, port=getattr(reader, "port", None), error=None)
            time.sleep(1.0)
    except Exception as exc:
        _update_meter_snapshot(status="error", connected=False, port=getattr(reader, "port", None), error=str(exc))
    finally:
        _update_meter_snapshot(status="idle", connected=False, port=getattr(reader, "port", None), error=getattr(reader, "connection_error", None))


def _detect_fnb58_target(preferred=None):
    if preferred:
        preferred_text = str(preferred).strip()
        if preferred_text:
            if preferred_text.startswith("/dev/hidraw"):
                return ("hidraw", preferred_text)
            if preferred_text.startswith("usb:"):
                return ("usb", preferred_text)
            return ("serial", preferred_text)

    usb_target = detect_fnb58_usb() if FNB58USBReader is not None else None
    if usb_target:
        return ("usb", usb_target)

    hidraw_target = detect_fnb58_hidraw() if FNB58HIDRawReader is not None else None
    if hidraw_target:
        return ("hidraw", hidraw_target)

    serial_target = detect_fnb58_port() if FNB58SerialReader is not None else None
    if serial_target:
        return ("serial", serial_target)

    return (None, None)


def start_fnb58_monitor(port=None):
    if FNB58HIDRawReader is None and FNB58USBReader is None and FNB58SerialReader is None:
        _update_meter_snapshot(status="error", connected=False, port=port, error="FNB58 readers are not available")
        return False

    existing_reader = FNB58_MONITOR.get("reader")
    if existing_reader and getattr(existing_reader, "running", False):
        log(f"[FNB58] Monitor already running on {getattr(existing_reader, 'port', None)}")
        return True

    transport, selected_port = _detect_fnb58_target(port or FNB58_PORT or None)
    if not selected_port:
        log("[FNB58] Device not found during start attempt")
        _update_meter_snapshot(status="searching", connected=False, error="FNB58 device not found")
        return False

    if transport == "hidraw" and FNB58HIDRawReader is None:
        _update_meter_snapshot(
            {
                "transport": "hidraw",
                "meter_source": "fnb58_hidraw",
            },
            status="error",
            connected=False,
            port=selected_port,
            error="FNB58 hidraw reader is not available",
        )
        return False

    if transport == "usb" and FNB58USBReader is None and FNB58ExporterReader is None:
        _update_meter_snapshot(
            {
                "transport": "usb_hid",
                "meter_source": "fnb58_usb_hid",
            },
            status="error",
            connected=False,
            port=selected_port,
            error="FNB58 USB readers are not available",
        )
        return False

    if transport == "serial" and FNB58SerialReader is None:
        _update_meter_snapshot(
            {
                "transport": "serial",
                "meter_source": "fnb58_serial",
            },
            status="error",
            connected=False,
            port=selected_port,
            error="FNB58 serial reader is not available",
        )
        return False

    reader = None
    last_error = None
    attempts = 2 if transport == "hidraw" else (3 if transport == "usb" else 1)
    for attempt in range(attempts):
        log(f"[FNB58] Start attempt {attempt + 1}/{attempts} via {transport} on {selected_port}")
        if transport == "hidraw":
            refreshed_transport, refreshed_port = _detect_fnb58_target(port or FNB58_PORT or selected_port)
            if refreshed_transport == "hidraw" and refreshed_port:
                selected_port = refreshed_port
            reader = FNB58HIDRawReader(selected_port)
        elif transport == "usb":
            refreshed_transport, refreshed_port = _detect_fnb58_target(port or FNB58_PORT or selected_port)
            if refreshed_transport == "usb" and refreshed_port:
                selected_port = refreshed_port
            reader = FNB58USBReader(selected_port) if FNB58USBReader is not None else None
            if reader is not None and reader.start():
                break
            last_error = getattr(reader, "connection_error", None) if reader is not None else None
            if FNB58ExporterReader is not None:
                reader = FNB58ExporterReader(selected_port)
            else:
                reader = None
        else:
            reader = FNB58SerialReader(selected_port)

        if reader is not None and reader.start():
            log(f"[FNB58] Reader connected via {getattr(reader, 'transport', transport)} on {getattr(reader, 'port', selected_port)}")
            break

        last_error = getattr(reader, "connection_error", None) if reader is not None else last_error
        if last_error:
            log(f"[FNB58] Start attempt failed: {last_error}")
        if attempt < attempts - 1:
            time.sleep(0.5)
    else:
        _update_meter_snapshot(
            {
                "transport": getattr(reader, "transport", transport),
                "meter_source": getattr(reader, "transport", transport),
            },
            status="error",
            connected=False,
            port=selected_port,
            error=last_error or getattr(reader, "connection_error", None),
        )
        log(f"[FNB58] Unable to start monitor: {last_error or getattr(reader, 'connection_error', None)}")
        return False

    monitor_thread = threading.Thread(target=_refresh_meter_from_reader, args=(reader,), daemon=True)
    FNB58_MONITOR["reader"] = reader
    FNB58_MONITOR["thread"] = monitor_thread
    monitor_thread.start()
    _update_meter_snapshot(
        {
            "transport": getattr(reader, "transport", transport),
            "meter_source": getattr(reader, "transport", transport),
        },
        status="connected",
        connected=True,
        port=selected_port,
        error=None,
    )
    return True


def stop_fnb58_monitor():
    reader = FNB58_MONITOR.get("reader")
    if reader and getattr(reader, "running", False):
        result = reader.stop()
        _update_meter_snapshot(result, status="stopped", connected=False, port=getattr(reader, "port", None), error=result.get("error"))
    FNB58_MONITOR["reader"] = None
    FNB58_MONITOR["thread"] = None


def _fnb58_retry_loop():
    while True:
        try:
            existing_reader = FNB58_MONITOR.get("reader")
            if existing_reader and getattr(existing_reader, "running", False):
                time.sleep(FNB58_RETRY_INTERVAL_S)
                continue

            meter = STATE.get("meter_metrics") or {}
            transport, selected_port = _detect_fnb58_target(FNB58_PORT or None)
            if selected_port:
                known_port = meter.get("port")
                if meter.get("connected") is not True or known_port != selected_port:
                    log(f"[FNB58] Auto-reconnect detected {transport} target {selected_port}")
                    start_fnb58_monitor(selected_port)
            else:
                _update_meter_snapshot(status="searching", connected=False, port=None, error="FNB58 device not found")
        except Exception as exc:
            log(f"[FNB58] Retry loop error: {exc}")

        time.sleep(FNB58_RETRY_INTERVAL_S)


def ensure_fnb58_retry_loop():
    thread = FNB58_RETRY_LOOP.get("thread")
    if thread and thread.is_alive():
        return
    thread = threading.Thread(target=_fnb58_retry_loop, daemon=True)
    FNB58_RETRY_LOOP["thread"] = thread
    thread.start()


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
    mib = 1024 * 1024
    return {
        "total_bytes": int(total),
        "available_bytes": int(available),
        "used_bytes": int(used),
        "total_mb": round(total / mib, 2),
        "available_mb": round(available / mib, 2),
        "used_mb": round(used / mib, 2),
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
            "total_gb": round(total / (1024 * 1024 * 1024), 3),
            "used_gb": round(used / (1024 * 1024 * 1024), 3),
            "free_gb": round(free / (1024 * 1024 * 1024), 3),
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
        # RPi and many embedded kernels expose hwmon deeper under platform devices
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
    base = 48.0
    primary_period_s = 120.0
    secondary_period_s = 23.0
    v = (
        base
        + 5.5 * math.sin((2.0 * math.pi * t) / primary_period_s)
        + 1.0 * math.sin((2.0 * math.pi * t) / secondary_period_s)
    )

    # Clamp to a reasonable range.
    v = max(30.0, min(75.0, v))
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
        "onnx_available": ONNX_AVAILABLE,
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
        "timestamp": _now_iso(),
        "cpu": {"percent": round(cpu, 2) if cpu is not None else None},
        "memory": mem,
        "storage": storage,
        "temperature_c": temp,
        "temperature_source": temp_source,
        "agent": {
            "status": STATE.get("status"),
            "model_name": STATE.get("model_name"),
            "inference_active": STATE.get("inference_active"),
            "runtime": STATE.get("runtime"),
        },
        "meter": {
            "status": STATE.get("meter_metrics", {}).get("status"),
            "connected": STATE.get("meter_metrics", {}).get("connected"),
            "port": STATE.get("meter_metrics", {}).get("port"),
            "transport": STATE.get("meter_metrics", {}).get("transport"),
            "power_mw": STATE.get("meter_metrics", {}).get("power_mw"),
            "measured_energy_mwh": STATE.get("meter_metrics", {}).get("measured_energy_mwh"),
            "updated_at": STATE.get("meter_metrics", {}).get("updated_at"),
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
    parsed_url = urlparse(model_url)
    model_ext = os.path.splitext(parsed_url.path)[1].lower().lstrip(".")
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
        predicted_energy = _safe_float(
            model_info.get("predicted_energy_mwh")
            or model_info.get("energy_avg_mwh")
            or model_info.get("predicted_mwh")
        )
        reset_energy_metrics(energy_budget_value, predicted_energy)
        set_state(
            status="downloading",
            model_name=model_name,
            model_info=model_info,
            controller_url=f"{parsed_url.scheme}://{parsed_url.netloc}" if parsed_url.scheme and parsed_url.netloc else CONTROLLER_URL,
            runtime=model_ext,
            error=None,
            inference_active=False,
            energy_metrics=STATE["energy_metrics"]
        )

        # Download model from controller
        log(f"Downloading from: {model_url}")
        resp = requests.get(model_url, stream=True, timeout=60)
        resp.raise_for_status()

        target_path = _current_model_path_for_ext(f".{model_ext}")
        tmp_path = target_path + ".tmp"
        with open(tmp_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        # Replace current model (automatically removes old model)
        _remove_other_model_files(keep_path=target_path)
        os.replace(tmp_path, target_path)
        file_size = os.path.getsize(target_path)
        
        log(f"Model downloaded: {file_size} bytes")
        log(f"Old model removed, new model deployed")
        set_state(
            status="ready",
            model_name=model_name,
            artifact_path=target_path,
            runtime=model_ext,
            model_info=model_info,
            error=None,
            energy_metrics=STATE["energy_metrics"]
        )

        if FNB58_AUTO_START:
            start_fnb58_monitor()
        _reset_meter_baseline()

        # Auto-start model inference in background
        log(f"Auto-starting new model: {model_name}")
        threading.Thread(target=start_model_inference, daemon=True).start()

        return jsonify({
            "message": "Deployment successful",
            "model_name": model_name,
            "model_path": target_path,
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

    active_model_path = _get_active_model_path()
    if not STATE.get("model_name") or not active_model_path or not os.path.exists(active_model_path):
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
            if LOADED_INPUT_SIZE is None or (LOADED_INTERPRETER is None and LOADED_ONNX_SESSION is None):
                return jsonify({"error": "Model not loaded in memory"}), 400
            interpreter = LOADED_INTERPRETER
            session = LOADED_ONNX_SESSION
            input_size = LOADED_INPUT_SIZE
            input_layout = LOADED_INPUT_LAYOUT
            input_meta = LOADED_INPUT_META
            runtime = LOADED_RUNTIME

        c, h, w = input_size
        if runtime == "onnx":
            dummy_input = _build_dummy_input(input_layout or "nchw", input_size, getattr(input_meta, "shape", None), np.float32)
            outputs = session.run(None, {input_meta.name: dummy_input})
            output = outputs[0]
        else:
            input_details = interpreter.get_input_details()[0]
            dtype = input_details["dtype"]
            shape = input_details["shape"]
            dummy_input = _build_dummy_input(input_layout or "nhwc", input_size, shape, dtype)
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
            "timestamp": _now_iso(),
            "model": STATE.get("model_name"),
            "runtime": runtime or "tflite",
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


def _infer_layout_and_size(shape, fallback):
    c_fallback, h_fallback, w_fallback = fallback
    if shape is None or len(shape) < 4:
        return "nchw", fallback

    dims = []
    for dim in shape[:4]:
        try:
            dims.append(int(dim) if dim is not None and int(dim) > 0 else None)
        except Exception:
            dims.append(None)

    if dims[1] in (1, 3, 4):
        return "nchw", (dims[1] or c_fallback, dims[2] or h_fallback, dims[3] or w_fallback)
    return "nhwc", (dims[3] or c_fallback, dims[1] or h_fallback, dims[2] or w_fallback)


def _build_dummy_input(layout, input_size, shape=None, dtype=np.float32):
    _ensure_numpy()
    c, h, w = input_size
    if layout == "nhwc":
        target_shape = tuple(shape) if shape is not None and len(shape) == 4 else (1, h, w, c)
    else:
        target_shape = tuple(shape) if shape is not None and len(shape) == 4 else (1, c, h, w)

    normalized_shape = []
    for dim in target_shape:
        try:
            dim_value = int(dim)
        except Exception:
            dim_value = 1
        normalized_shape.append(dim_value if dim_value > 0 else 1)
    return np.random.rand(*normalized_shape).astype(dtype)


def _load_tflite_model(model_path, input_size):
    """Load a .tflite model and run a warmup inference."""
    if not TFLITE_AVAILABLE:
        raise RuntimeError("tflite_runtime is not available on this device")
    _ensure_numpy()

    interpreter = tflite.Interpreter(model_path=model_path, num_threads=1)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    layout, normalized_input_size = _infer_layout_and_size(input_details.get("shape"), input_size)
    dtype = input_details["dtype"]
    shape = input_details["shape"]
    dummy = _build_dummy_input(layout, normalized_input_size, shape, dtype)
    interpreter.set_tensor(input_details["index"], dummy)
    interpreter.invoke()  # warmup
    return interpreter, input_details, normalized_input_size, layout


def _load_onnx_model(model_path, input_size):
    """Load an ONNX model and run a warmup inference."""
    if not ONNX_AVAILABLE:
        raise RuntimeError("ONNX Runtime is not available on this device")
    _ensure_numpy()

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]
    layout, normalized_input_size = _infer_layout_and_size(input_meta.shape, input_size)
    dummy = _build_dummy_input(layout, normalized_input_size, input_meta.shape, np.float32)
    session.run(None, {input_meta.name: dummy})
    return session, input_meta, normalized_input_size, layout

#ss kq dd
# -------- Energy measurement utilities (powercap-based) --------
def _find_powercap_energy_file():
    base = "/sys/class/powercap"
    if not os.path.isdir(base):
        return None
    # try package-level energy_uj (total)
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
    """Measure energy using powercap energy_uj while running inferences."""
    energy_file = _find_powercap_energy_file()
    if energy_file is None:
        return {
            "success": False,
            "error": "powercap energy_uj not available",
            "sensor_type": "unknown"
        }

    if STATE.get("status") not in ("running", "ready"):
        return {"success": False, "error": "Model not ready/running"}

    # Prepare model handles
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

    delta_uj = end_uj - start_uj  # microjoules
    elapsed_s = max(end_ts - start_ts, 1e-6)
    # mWh = μJ / 3.6e6
    energy_mwh = float(delta_uj) / 3_600_000.0
    avg_power_mw = (energy_mwh * 3600.0) / elapsed_s  # mWh -> mW via P = E/t

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

    # Optionally post to controller
    try:
        controller = data.get("controller_url") or STATE.get("controller_url") or CONTROLLER_URL
        if controller:
            payload = {
                "device_type": "jetson_nano",
                "device_id": os.getenv("BALENA_DEVICE_UUID"),
                "model_name": STATE.get("model_name"),
                "actual_energy_mwh": report.get("actual_energy_mwh"),
                "predicted_mwh": (STATE.get("energy_metrics") or {}).get("predicted_mwh"),
                "avg_power_mw": report.get("avg_power_mw"),
                "duration_s": report.get("duration_s"),
                "sensor_type": report.get("sensor_type"),
                "model_info": STATE.get("model_info"),
            }
            try:
                requests.post(f"{controller.rstrip('/')}/api/energy/report", json=payload, timeout=10)
            except Exception as post_err:
                report["post_warning"] = f"Failed to post to controller: {post_err}"
    except Exception as e:
        report["post_warning"] = str(e)

    return jsonify(report)


@app.route("/fnb58/status", methods=["GET"])
def fnb58_status():
    return jsonify({
        "success": True,
        "meter_metrics": STATE.get("meter_metrics"),
    })


@app.route("/fnb58/start", methods=["POST"])
def fnb58_start():
    data = request.get_json(silent=True) or {}
    port = data.get("port") or FNB58_PORT or None
    connected = start_fnb58_monitor(port)
    return jsonify({
        "success": connected,
        "meter_metrics": STATE.get("meter_metrics"),
    })


@app.route("/fnb58/stop", methods=["POST"])
def fnb58_stop():
    stop_fnb58_monitor()
    return jsonify({
        "success": True,
        "meter_metrics": STATE.get("meter_metrics"),
    })


@app.route("/measure_energy_fnb58", methods=["POST"])
def measure_energy_fnb58():
    """Measure energy using FNB58 USB tester via USB HID or serial fallback.
    
    Request body:
    {
      "fnb58_port": "usb:001:010",  # or /dev/ttyUSB0 when the device exposes a serial port
      "duration_s": 30,
      "auto_detect": true  # if true, try to find FNB58 automatically
    }
    
    Returns energy measurement and optionally posts to controller.
    """
    data = request.get_json(silent=True) or {}
    duration_s = float(data.get("duration_s", 30.0))
    port = data.get("fnb58_port")
    auto_detect = data.get("auto_detect", True)

    meter = STATE.get("meter_metrics") or {}
    if meter.get("connected") and _safe_float(meter.get("total_energy_wh")) is not None:
        start_total_wh = _safe_float(meter.get("total_energy_wh"))
        time.sleep(duration_s)
        meter_after = STATE.get("meter_metrics") or {}
        end_total_wh = _safe_float(meter_after.get("total_energy_wh"))
        if start_total_wh is not None and end_total_wh is not None:
            actual_energy_mwh = max(end_total_wh - start_total_wh, 0.0) * 1000.0
            avg_power_mw = (actual_energy_mwh * 3600.0 / duration_s) if duration_s > 0 else None
            response = {
                "success": True,
                "sensor_type": "fnb58",
                "port": meter_after.get("port"),
                "duration_s": duration_s,
                "actual_energy_mwh": round(actual_energy_mwh, 4),
                "avg_power_mw": round(avg_power_mw, 4) if avg_power_mw is not None else None,
                "last_values": meter_after.get("last_values"),
            }
            try:
                controller = data.get("controller_url") or STATE.get("controller_url") or CONTROLLER_URL
                if controller and response.get("actual_energy_mwh") is not None:
                    payload = {
                        "device_type": "jetson_nano",
                        "device_id": os.getenv("BALENA_DEVICE_UUID"),
                        "model_name": STATE.get("model_name"),
                        "actual_energy_mwh": response["actual_energy_mwh"],
                        "predicted_mwh": (STATE.get("energy_metrics") or {}).get("predicted_mwh"),
                        "avg_power_mw": response.get("avg_power_mw"),
                        "duration_s": duration_s,
                        "sensor_type": "fnb58",
                        "model_info": STATE.get("model_info"),
                    }
                    requests.post(f"{controller.rstrip('/')}/api/energy/report", json=payload, timeout=10)
                    response["posted_to_controller"] = True
            except Exception as post_err:
                response["post_warning"] = str(post_err)
            return jsonify(response)
    
    # Try to detect FNB58 if not specified
    if not port and auto_detect:
        log(f"[FNB58] Attempting auto-detection...")
        _, port = _detect_fnb58_target()
        if port:
            log(f"[FNB58] Detected on {port}")
    
    if not port:
        return jsonify({
            "success": False,
            "error": "FNB58 device not found. Specify fnb58_port or enable auto_detect"
        }), 400

    try:
        log(f"[FNB58] Measuring for {duration_s}s on {port}...")
        transport, resolved_port = _detect_fnb58_target(port)
        if transport == "hidraw":
            if FNB58HIDRawReader is None:
                return jsonify({
                    "success": False,
                    "error": "FNB58 hidraw reader is not available in the agent image."
                }), 500
            reader = FNB58HIDRawReader(resolved_port)
        elif transport == "usb":
            if FNB58USBReader is None and FNB58ExporterReader is None:
                return jsonify({
                    "success": False,
                    "error": "FNB58 USB readers are not available in the agent image."
                }), 500
            reader = FNB58USBReader(resolved_port) if FNB58USBReader is not None else None
        else:
            if FNB58SerialReader is None:
                return jsonify({
                    "success": False,
                    "error": "FNB58 serial reader is not available in the agent image."
                }), 500
            reader = FNB58SerialReader(resolved_port)
        
        if reader is None or not reader.start():
            if transport == "usb" and FNB58ExporterReader is not None:
                reader = FNB58ExporterReader(resolved_port, stream_seconds=duration_s)
            if reader is None or not reader.start():
                return jsonify({
                    "success": False,
                    "error": f"Failed to connect to {resolved_port}: {getattr(reader, 'connection_error', None)}"
                }), 400

        if not getattr(reader, "running", False) and getattr(reader, "sample_count", 0) == 0:
            return jsonify({
                "success": False,
                "error": f"Failed to connect to {resolved_port}: {getattr(reader, 'connection_error', None)}"
            }), 400

        # Wait for measurement
        if getattr(reader, "transport", "") != "usb_exporter":
            time.sleep(duration_s)
        result = reader.stop()
        
        if not result.get('success'):
            return jsonify({
                "success": False,
                "error": f"FNB58 measurement failed: {result.get('error')}"
            }), 400
        
        # Build response
        response = {
            "success": True,
            "sensor_type": "fnb58",
            "port": resolved_port,
            "duration_s": duration_s,
            "samples_count": result.get('samples_count'),
            "actual_energy_mwh": result.get('total_energy_mwh'),
            "avg_power_mw": result.get('avg_power_mw'),
            "last_values": result.get('last_values'),
            "meter_source": result.get("meter_source"),
            "transport": result.get("transport"),
        }
        _update_meter_snapshot(result, status="connected", connected=True, port=resolved_port, error=None)
        
        # Post to controller if available
        try:
            controller = data.get("controller_url") or STATE.get("controller_url") or CONTROLLER_URL
            if controller and response.get('actual_energy_mwh'):
                payload = {
                    "device_type": "jetson_nano",
                    "device_id": os.getenv("BALENA_DEVICE_UUID"),
                    "model_name": STATE.get("model_name"),
                    "actual_energy_mwh": response['actual_energy_mwh'],
                    "predicted_mwh": (STATE.get("energy_metrics") or {}).get("predicted_mwh"),
                    "avg_power_mw": response.get('avg_power_mw'),
                    "duration_s": duration_s,
                    "sensor_type": "fnb58",
                    "model_info": STATE.get("model_info"),
                }
                try:
                    requests.post(f"{controller.rstrip('/')}/api/energy/report", json=payload, timeout=10)
                    response["posted_to_controller"] = True
                except Exception as post_err:
                    response["post_warning"] = f"Failed to post to controller: {post_err}"
        except Exception as e:
            response["post_warning"] = str(e)
        
        return jsonify(response)
    
    except Exception as e:
        log(f"[FNB58] Error: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route("/telemetry", methods=["POST"])
def telemetry():
    """
    Receive live telemetry (energy, power, latency) from Raspberry Pi sensors
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

    energy, power, parsed_from = _derive_energy_payload(data)
    if energy is None:
        return jsonify({"error": "No supported energy fields found in telemetry payload"}), 400

    latency = data.get("latency_s")
    note = data.get("note") or data.get("meter_note")
    timestamp = data.get("timestamp")
    source = data.get("source") or data.get("meter_source") or parsed_from

    within_budget = record_energy_sample(
        energy,
        power_mw=power,
        latency_s=latency,
        note=note,
        timestamp=timestamp,
        source=source
    )

    if any(key in data for key in ("voltage_v", "current_a", "power_w", "energy_wh")):
        _update_meter_snapshot({
            "last_values": {
                "voltage_v": data.get("voltage_v"),
                "current_a": data.get("current_a"),
                "power_w": data.get("power_w") or ((power or 0.0) / 1000.0 if power is not None else None),
                "energy_wh": data.get("energy_wh") or (energy / 1000.0),
            }
        }, status="connected", connected=True, port=STATE.get("meter_metrics", {}).get("port"), error=None)

    message = "Telemetry ingested"
    if not within_budget:
        message = "Telemetry ingested - energy budget exceeded, inference halted"

    return jsonify({
        "message": message,
        "parsed_from": parsed_from,
        "energy_metrics": STATE["energy_metrics"],
        "meter_metrics": STATE["meter_metrics"],
        "status": STATE["status"]
    })


def start_model_inference():
    """
    Start model inference (runs in background thread)
    
    Loads TFLite model from CURRENT_MODEL_PATH and runs warmup forward.
    Keeps interpreter in memory for /predict.
    """
    try:
        global LOADED_INTERPRETER, LOADED_ONNX_SESSION, LOADED_RUNTIME
        global LOADED_MODEL_NAME, LOADED_MODEL_PATH, LOADED_INPUT_SIZE, LOADED_INPUT_LAYOUT, LOADED_INPUT_META, LOADED_AT

        log("Starting model inference...")
        model_name = STATE.get("model_name")
        model_info = STATE.get("model_info") or {}
        if not model_name:
            raise RuntimeError("No model_name present in state")
        model_path = _get_active_model_path()
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        input_size = _parse_input_size(model_info.get("input_size"))
        with MODEL_LOCK:
            # Ensure any previous cache is dropped before loading.
            _unload_loaded_model("start")
            if model_path.endswith(".onnx"):
                session, input_meta, normalized_input_size, layout = _load_onnx_model(model_path, input_size)
                LOADED_ONNX_SESSION = session
                LOADED_RUNTIME = "onnx"
                LOADED_INPUT_META = input_meta
            else:
                interpreter, input_meta, normalized_input_size, layout = _load_tflite_model(model_path, input_size)
                LOADED_INTERPRETER = interpreter
                LOADED_RUNTIME = "tflite"
                LOADED_INPUT_META = input_meta
            LOADED_MODEL_NAME = model_name
            LOADED_MODEL_PATH = model_path
            LOADED_INPUT_SIZE = normalized_input_size
            LOADED_INPUT_LAYOUT = layout
            LOADED_AT = _now_iso()

        set_state(status="running", inference_active=True, runtime=LOADED_RUNTIME, artifact_path=model_path, error=None)

        log(f"Model '{STATE['model_name']}' is now running (runtime: {LOADED_RUNTIME})")
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
    log("Jetson ML Agent starting...")
    log(f"Model directory: {MODEL_DIR}")
    
    # Restore state from previous session
    load_state_from_disk()
    
    # Check if there's an existing model
    active_model_path = _get_active_model_path()
    if active_model_path and os.path.exists(active_model_path):
        STATE["artifact_path"] = active_model_path
        STATE["runtime"] = os.path.splitext(active_model_path)[1].lstrip(".")
        size = os.path.getsize(active_model_path)
        log(f"Found existing model: {size} bytes")
        if STATE.get("model_name"):
            log(f"Current model: {STATE['model_name']}")

    if FNB58_AUTO_START:
        start_fnb58_monitor()
        ensure_fnb58_retry_loop()
    
    # Start Flask server
    app.run(host="0.0.0.0", port=8000)
