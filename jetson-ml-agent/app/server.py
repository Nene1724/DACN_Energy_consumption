import os
import threading
import uuid
from datetime import datetime, timezone
from urllib.parse import urlparse
from flask import Flask, request, jsonify, Response
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
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ImportError:
    trt = None
    TENSORRT_AVAILABLE = False

try:
    from cuda import cudart
    CUDA_RUNTIME_AVAILABLE = True
except ImportError:
    cudart = None
    CUDA_RUNTIME_AVAILABLE = False

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

try:
    from movenet_fall_detection import (
        analyze_pose,
        capture_camera_snapshot,
        extract_keypoints,
        open_camera,
        preprocess_frame_bgr,
        summarize_detection_window,
    )
except ImportError:
    analyze_pose = None
    capture_camera_snapshot = None
    extract_keypoints = None
    open_camera = None
    preprocess_frame_bgr = None
    summarize_detection_window = None

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
CAMERA_DEVICE = (os.getenv("CAMERA_DEVICE") or "/dev/video0").strip()
CAMERA_WIDTH = int(os.getenv("CAMERA_WIDTH", "640") or "640")
CAMERA_HEIGHT = int(os.getenv("CAMERA_HEIGHT", "480") or "480")
FALL_DETECT_DURATION_S = max(0.5, float(os.getenv("FALL_DETECT_DURATION_S", "2.5")))
FALL_DETECT_MAX_FRAMES = max(4, int(os.getenv("FALL_DETECT_MAX_FRAMES", "16") or "16"))
FALL_DETECT_MAX_READ_FAILURES = max(1, int(os.getenv("FALL_DETECT_MAX_READ_FAILURES", "6") or "6"))
TENSORRT_ENGINE_ENABLED = (os.getenv("TENSORRT_ENGINE_ENABLED", "1").strip().lower() in ("1", "true", "yes", "on"))
TENSORRT_FP16_ENABLED = (os.getenv("TENSORRT_FP16_ENABLED", "1").strip().lower() in ("1", "true", "yes", "on"))
TENSORRT_WORKSPACE_BYTES = max(1 << 20, int(os.getenv("TENSORRT_WORKSPACE_BYTES", str(512 * 1024 * 1024))))
VIDEO_UPLOAD_DIR = (os.getenv("VIDEO_UPLOAD_DIR") or "/data/uploaded_videos").strip()

# Used only for simulated fallback telemetry when the OS doesn't expose temp sensors.
_SIM_TEMP_START_TS = time.time()


def _safe_video_upload_dir() -> str:
    base_dir = VIDEO_UPLOAD_DIR or "/data/uploaded_videos"
    try:
        os.makedirs(base_dir, exist_ok=True)
        return base_dir
    except Exception:
        fallback = "/tmp/uploaded_videos"
        os.makedirs(fallback, exist_ok=True)
        return fallback


def _is_file_camera_source(camera_source: object) -> bool:
    if camera_source is None:
        return False
    text = str(camera_source).strip()
    if not text:
        return False
    if os.path.isfile(text):
        return True
    return text.lower().endswith(".mp4")


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
        _current_model_path_for_ext(".engine"),
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


def _create_fall_detection_metrics():
    return {
        "enabled": False,
        "camera_device": CAMERA_DEVICE,
        "camera_source": None,
        "camera_ready": False,
        "fall_detected": None,
        "fall_score": None,
        "label": None,
        "frames_analyzed": 0,
        "updated_at": None,
        "last_error": None,
        "details": {},
    }


def _create_benchmark_metrics():
    return {
        "status": "idle",
        "runtime": None,
        "warmup_runs": 0,
        "benchmark_runs": 0,
        "latency_avg_s": None,
        "latency_std_s": None,
        "latency_p50_s": None,
        "latency_p95_s": None,
        "throughput_iter_per_s": None,
        "iterations": 0,
        "updated_at": None,
        "last_error": None,
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
    "fall_detection": _create_fall_detection_metrics(),
    "benchmark_metrics": _create_benchmark_metrics(),
}

# In-memory runtime cache
LOADED_INTERPRETER = None
LOADED_ONNX_SESSION = None
LOADED_TRT_RUNNER = None
LOADED_RUNTIME = None
LOADED_MODEL_NAME = None
LOADED_MODEL_PATH = None
LOADED_INPUT_SIZE = None
LOADED_INPUT_LAYOUT = None
LOADED_INPUT_META = None
LOADED_AT = None
MODEL_LOCK = threading.RLock()
CAMERA_LOCK = threading.RLock()
FNB58_MONITOR = {"reader": None, "thread": None}
FNB58_RETRY_LOOP = {"thread": None}
CAMERA_FEED = {
    "cap": None,
    "source": None,
    "requested_source": None,
    "is_file_source": False,
    "last_error": None,
    "latest_frame": None,
    "latest_frame_ts": None,
    "last_good_frame": None,
    "last_good_frame_ts": None,
    "worker": None,
    "running": False,
    "condition": threading.Condition(CAMERA_LOCK),
}


def _get_cv2_module():
    try:
        import movenet_fall_detection as _mfd

        return getattr(_mfd, "cv2", None)
    except Exception:
        return None


def _release_camera_feed_locked():
    CAMERA_FEED["running"] = False
    cap = CAMERA_FEED.get("cap")
    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass
    CAMERA_FEED["cap"] = None
    CAMERA_FEED["source"] = None
    CAMERA_FEED["requested_source"] = None
    CAMERA_FEED["latest_frame"] = None
    CAMERA_FEED["latest_frame_ts"] = None


def _drop_camera_capture_locked():
    """Release only the active capture handle, keep worker state for auto-retry."""
    cap = CAMERA_FEED.get("cap")
    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass
    CAMERA_FEED["cap"] = None
    CAMERA_FEED["source"] = None


def _camera_worker(requested_source):
    source = requested_source or CAMERA_DEVICE
    backoff_s = 0.05
    while True:
        with CAMERA_LOCK:
            if not CAMERA_FEED.get("running") or CAMERA_FEED.get("requested_source") != source:
                break

            cap = CAMERA_FEED.get("cap")
            if cap is None or not getattr(cap, "isOpened", lambda: False)():
                try:
                    cap, actual_source = open_camera(source, CAMERA_WIDTH, CAMERA_HEIGHT)
                    CAMERA_FEED["cap"] = cap
                    CAMERA_FEED["source"] = actual_source
                    CAMERA_FEED["requested_source"] = source
                    CAMERA_FEED["is_file_source"] = _is_file_camera_source(source)
                    CAMERA_FEED["last_error"] = None
                except Exception as exc:
                    CAMERA_FEED["last_error"] = str(exc)
                    cap = None

            if cap is None:
                if CAMERA_FEED.get("latest_frame") is None:
                    # Nothing to serve yet, keep retrying.
                    pass
                else:
                    backoff_s = 0.05

        if cap is None:
            time.sleep(backoff_s)
            backoff_s = min(backoff_s * 1.5, 0.5)
            continue

        try:
            ok, frame_bgr = cap.read()
            if ok and frame_bgr is not None:
                with CAMERA_LOCK:
                    CAMERA_FEED["latest_frame"] = frame_bgr.copy()
                    CAMERA_FEED["latest_frame_ts"] = time.time()
                    CAMERA_FEED["last_good_frame"] = frame_bgr.copy()
                    CAMERA_FEED["last_good_frame_ts"] = CAMERA_FEED["latest_frame_ts"]
                    CAMERA_FEED["last_error"] = None
                    if CAMERA_FEED.get("source") is None:
                        CAMERA_FEED["source"] = source
                    with CAMERA_FEED["condition"]:
                        CAMERA_FEED["condition"].notify_all()
                backoff_s = 0.05
            else:
                with CAMERA_LOCK:
                    CAMERA_FEED["last_error"] = f"Camera frame read failed for source: {source}"
                time.sleep(backoff_s)
                backoff_s = min(backoff_s * 1.5, 0.5)
        except Exception as exc:
            with CAMERA_LOCK:
                CAMERA_FEED["last_error"] = str(exc)
                _drop_camera_capture_locked()
            time.sleep(backoff_s)
            backoff_s = min(backoff_s * 1.5, 0.5)


def _ensure_camera_worker_locked(requested_source):
    source = requested_source or CAMERA_DEVICE
    current_source = CAMERA_FEED.get("requested_source")
    worker = CAMERA_FEED.get("worker")
    if worker is not None and worker.is_alive() and current_source == source and CAMERA_FEED.get("running"):
        return

    _release_camera_feed_locked()
    CAMERA_FEED["requested_source"] = source
    CAMERA_FEED["running"] = True
    worker = threading.Thread(target=_camera_worker, args=(source,), daemon=True)
    CAMERA_FEED["worker"] = worker
    worker.start()


def _ensure_camera_feed_locked(requested_source):
    source = requested_source or CAMERA_DEVICE
    _ensure_camera_worker_locked(source)
    return CAMERA_FEED.get("cap"), CAMERA_FEED.get("source")


def _read_camera_frame(requested_source=None, wait_timeout_s=0.35):
    with CAMERA_LOCK:
        def _fallback_last_good(actual_source, err):
            stale_frame = CAMERA_FEED.get("last_good_frame")
            stale_ts = CAMERA_FEED.get("last_good_frame_ts")
            if stale_frame is None or stale_ts is None:
                return None, actual_source, err
            age_s = max(0.0, time.time() - float(stale_ts))
            # Use recent cached frame to avoid no-frames spikes during brief camera reopen failures.
            if age_s <= 2.5:
                return stale_frame.copy(), actual_source, err
            return None, actual_source, err

        if open_camera is None:
            err = "Camera helper open_camera is unavailable in the agent image"
            CAMERA_FEED["last_error"] = err
            return _fallback_last_good(requested_source or CAMERA_DEVICE, err)

        try:
            _ensure_camera_feed_locked(requested_source)
            latest_frame = CAMERA_FEED.get("latest_frame")
            actual_source = CAMERA_FEED.get("source") or requested_source or CAMERA_DEVICE
            if latest_frame is not None:
                CAMERA_FEED["last_error"] = None
                return latest_frame.copy(), actual_source, None

            # Wait briefly for the worker to produce its first frame.
            condition = CAMERA_FEED["condition"]
            with condition:
                condition.wait(timeout=max(0.05, float(wait_timeout_s or 0.35)))
            latest_frame = CAMERA_FEED.get("latest_frame")
            actual_source = CAMERA_FEED.get("source") or requested_source or CAMERA_DEVICE
            if latest_frame is not None:
                CAMERA_FEED["last_error"] = None
                return latest_frame.copy(), actual_source, None

            err = CAMERA_FEED.get("last_error") or f"Camera frame read failed for source: {requested_source or CAMERA_DEVICE}"
            return _fallback_last_good(actual_source, err)
        except Exception as exc:
            err = f"Unable to open camera source: {requested_source or CAMERA_DEVICE} ({exc})"
            CAMERA_FEED["last_error"] = err
            return _fallback_last_good(requested_source or CAMERA_DEVICE, err)


def _get_cached_camera_frame(requested_source=None, max_age_s=8.0):
    with CAMERA_LOCK:
        actual_source = CAMERA_FEED.get("source") or requested_source or CAMERA_DEVICE
        now_ts = time.time()

        latest = CAMERA_FEED.get("latest_frame")
        latest_ts = CAMERA_FEED.get("latest_frame_ts")
        if latest is not None and latest_ts is not None and (now_ts - float(latest_ts)) <= float(max_age_s):
            return latest.copy(), actual_source, None

        stale = CAMERA_FEED.get("last_good_frame")
        stale_ts = CAMERA_FEED.get("last_good_frame_ts")
        if stale is not None and stale_ts is not None and (now_ts - float(stale_ts)) <= float(max_age_s):
            return stale.copy(), actual_source, None

        return None, actual_source, CAMERA_FEED.get("last_error") or "No cached camera frame available"


def _unload_loaded_model(reason: str = ""):
    """Drop the in-memory model cache to avoid mixing weights across deploys."""
    global LOADED_INTERPRETER, LOADED_ONNX_SESSION, LOADED_TRT_RUNNER, LOADED_RUNTIME
    global LOADED_MODEL_NAME, LOADED_MODEL_PATH, LOADED_INPUT_SIZE, LOADED_INPUT_LAYOUT, LOADED_INPUT_META, LOADED_AT
    with MODEL_LOCK:
        if LOADED_INTERPRETER is not None or LOADED_ONNX_SESSION is not None or LOADED_TRT_RUNNER is not None:
            log(f"Unloading cached model{': ' + reason if reason else ''}")
        if LOADED_TRT_RUNNER is not None:
            try:
                LOADED_TRT_RUNNER.close()
            except Exception:
                pass
        LOADED_INTERPRETER = None
        LOADED_ONNX_SESSION = None
        LOADED_TRT_RUNNER = None
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

    try:
        usb_target = detect_fnb58_usb() if FNB58USBReader is not None else None
    except Exception as exc:
        log(f"[FNB58] USB detection skipped: {exc}")
        usb_target = None
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


def _normalized_text(value):
    return str(value or "").strip().lower()


def _is_fall_detection_model_info(model_info=None):
    info = model_info if isinstance(model_info, dict) else (STATE.get("model_info") or {})
    markers = {
        _normalized_text(info.get("use_case")),
        _normalized_text(info.get("task")),
        _normalized_text(info.get("task_type")),
        _normalized_text(info.get("source_type")),
        _normalized_text(info.get("model_name")),
        _normalized_text(info.get("artifact_file")),
        _normalized_text(info.get("pose_model")),
    }

    for marker in markers:
        if not marker:
            continue
        if marker in {"fall_detection_pose", "fall_detection", "camera_fall_detection"}:
            return True
        if "movenet" in marker:
            return True
    return False


def _update_fall_detection_snapshot(result=None, *, error=None):
    metrics = STATE.get("fall_detection") or _create_fall_detection_metrics()
    metrics["enabled"] = _is_fall_detection_model_info()
    metrics["camera_device"] = CAMERA_DEVICE
    metrics["updated_at"] = _now_iso()

    if result:
        metrics["camera_source"] = result.get("camera_source")
        metrics["camera_ready"] = bool(result.get("camera_ready", True))
        metrics["fall_detected"] = result.get("fall_detected")
        metrics["fall_score"] = result.get("fall_score")
        metrics["label"] = result.get("label")
        metrics["frames_analyzed"] = result.get("frames_analyzed", 0)
        metrics["details"] = result.get("details") or {}
        metrics["last_error"] = None

    if error is not None:
        metrics["camera_ready"] = False
        metrics["last_error"] = str(error)

    STATE["fall_detection"] = metrics


def reset_fall_detection_metrics(error=None):
    metrics = _create_fall_detection_metrics()
    metrics["enabled"] = _is_fall_detection_model_info()
    metrics["camera_device"] = CAMERA_DEVICE
    metrics["updated_at"] = _now_iso()
    metrics["last_error"] = str(error) if error else None
    STATE["fall_detection"] = metrics


def _update_benchmark_snapshot(result=None, *, status=None, error=None):
    metrics = STATE.get("benchmark_metrics") or _create_benchmark_metrics()
    if status is not None:
        metrics["status"] = status
    if error is not None:
        metrics["last_error"] = str(error)
    metrics["updated_at"] = _now_iso()

    if result:
        for key in (
            "runtime",
            "warmup_runs",
            "benchmark_runs",
            "latency_avg_s",
            "latency_std_s",
            "latency_p50_s",
            "latency_p95_s",
            "throughput_iter_per_s",
            "iterations",
        ):
            if key in result:
                metrics[key] = result.get(key)
        if error is None:
            metrics["last_error"] = None

    STATE["benchmark_metrics"] = metrics


def reset_benchmark_metrics(error=None):
    metrics = _create_benchmark_metrics()
    metrics["runtime"] = STATE.get("runtime")
    metrics["updated_at"] = _now_iso()
    metrics["last_error"] = str(error) if error else None
    STATE["benchmark_metrics"] = metrics


def _mark_camera_ready(camera_source=None):
    metrics = STATE.get("fall_detection") or _create_fall_detection_metrics()
    metrics["enabled"] = _is_fall_detection_model_info()
    metrics["camera_device"] = CAMERA_DEVICE
    metrics["camera_source"] = str(camera_source) if camera_source is not None else metrics.get("camera_source")
    metrics["camera_ready"] = True
    metrics["updated_at"] = _now_iso()
    if metrics.get("last_error") == "Camera frame read failed":
        metrics["last_error"] = None
    STATE["fall_detection"] = metrics


def _build_camera_overlay_lines():
    fall_metrics = STATE.get("fall_detection") or {}
    lines = [
        f"Model: {STATE.get('model_name') or 'none'}",
        f"Device: {CAMERA_DEVICE}",
    ]
    label = fall_metrics.get("label")
    score = _safe_float(fall_metrics.get("fall_score"))
    if label:
        if score is not None:
            lines.append(f"Fall: {label} ({score:.2f})")
        else:
            lines.append(f"Fall: {label}")
    return lines


def _draw_overlay_lines(frame_bgr, lines):
    cv2 = _get_cv2_module()
    if cv2 is None or frame_bgr is None:
        return frame_bgr

    origin_y = 28
    for idx, line in enumerate((lines or [])[:6]):
        y = origin_y + (idx * 24)
        cv2.putText(
            frame_bgr,
            str(line),
            (14, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame_bgr,
            str(line),
            (14, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (24, 24, 24),
            1,
            cv2.LINE_AA,
        )
    return frame_bgr


def _capture_camera_snapshot_result(camera_source=None, annotate=True):
    cv2 = _get_cv2_module()
    if cv2 is None:
        raise RuntimeError("OpenCV is not available in the agent image.")

    frame_bgr, actual_source, err = _read_camera_frame(camera_source or CAMERA_DEVICE)
    if frame_bgr is None:
        raise RuntimeError(err or "Unable to capture camera frame")

    if annotate:
        frame_bgr = _draw_overlay_lines(frame_bgr, _build_camera_overlay_lines())

    ok, encoded = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise RuntimeError("Failed to encode camera frame as JPEG")

    frame_bytes = encoded.tobytes()
    _mark_camera_ready(actual_source)
    return frame_bytes, actual_source


def _run_camera_fall_detection(duration_s=None, max_frames=None, camera_source=None, fast_mode=False):
    if not _is_fall_detection_model_info():
        return {
            "success": False,
            "error": "Current deployed model is not configured for camera fall detection.",
        }, 400

    if open_camera is None or preprocess_frame_bgr is None or extract_keypoints is None or analyze_pose is None:
        return {
            "success": False,
            "error": "MoveNet fall-detection helpers are not available in the agent image.",
        }, 500

    with MODEL_LOCK:
        runtime = LOADED_RUNTIME
        interpreter = LOADED_INTERPRETER
        input_meta = LOADED_INPUT_META
        input_size = LOADED_INPUT_SIZE

    if not TFLITE_AVAILABLE:
        return {
            "success": False,
            "error": "TFLite runtime is not available. Install tensorflow or tflite_runtime to run camera fall detection.",
        }, 500

    if runtime != "tflite" or interpreter is None or input_size is None:
        return {
            "success": False,
            "error": "Fall detection currently requires a loaded TFLite MoveNet model.",
        }, 400

    duration_s = max(0.5, float(duration_s if duration_s is not None else FALL_DETECT_DURATION_S))
    max_frames = max(4, int(max_frames if max_frames is not None else FALL_DETECT_MAX_FRAMES))
    fast_mode = bool(fast_mode)
    requested_source = camera_source or CAMERA_DEVICE
    is_file_source = _is_file_camera_source(requested_source)
    c, h, w = input_size
    target_size = int(max(h, w))

    frame_results = []
    frame_count = 0
    frame_read_failures = 0
    max_read_failures = max(FALL_DETECT_MAX_READ_FAILURES, 6)
    start_ts = time.time()
    last_error = None
    actual_source = requested_source
    first_frame_wait_s = max(1.2, min(3.0, duration_s + 0.4))
    detection_window_s = max(duration_s, first_frame_wait_s + 0.35)

    if fast_mode:
        frame_bgr, actual_source, frame_err = _get_cached_camera_frame(requested_source, max_age_s=1.6)
        if frame_bgr is None:
            frame_bgr, actual_source, frame_err = _read_camera_frame(requested_source, wait_timeout_s=0.22)
        if frame_bgr is None:
            last_error = frame_err or f"Camera frame read failed for source: {requested_source}"
        else:
            try:
                input_details = input_meta if isinstance(input_meta, dict) else interpreter.get_input_details()[0]
                output_details = interpreter.get_output_details()[0]
                input_dtype = input_details.get("dtype", np.uint8)

                input_tensor, _ = preprocess_frame_bgr(frame_bgr, target_size, input_dtype=input_dtype)
                with MODEL_LOCK:
                    interpreter.set_tensor(input_details["index"], input_tensor)
                    interpreter.invoke()
                    output = interpreter.get_tensor(output_details["index"])

                keypoints = extract_keypoints(output)
                pose_result = analyze_pose(keypoints)
                pose_result["frame_index"] = 0
                frame_results.append(pose_result)
                frame_count = 1
            except Exception as exc:
                last_error = f"Fall inference failed: {exc}"

        try:
            summary = summarize_detection_window(frame_results) if summarize_detection_window is not None else {
                "frames_analyzed": len(frame_results),
                "fall_detected": False,
                "fall_score": 0.0,
                "label": "unsupported",
            }
        except Exception as exc:
            last_error = f"Fall summary failed: {exc}"
            summary = {
                "frames_analyzed": len(frame_results),
                "fall_detected": False,
                "fall_score": 0.0,
                "label": "no_frames" if not frame_results else "uncertain",
                "avg_fall_score": None,
                "fall_frames": 0,
                "fall_frame_ratio": None,
                "max_consecutive_fall_frames": 0,
                "best_frame": None,
            }

        summary = summary if isinstance(summary, dict) else {
            "frames_analyzed": len(frame_results),
            "fall_detected": False,
            "fall_score": 0.0,
            "label": "unsupported",
        }
        result = {
            "success": True,
            "timestamp": _now_iso(),
            "model": STATE.get("model_name"),
            "runtime": runtime,
            "camera_source": str(actual_source),
            "camera_ready": bool(frame_results),
            "duration_s": round(time.time() - start_ts, 3),
            "frames_requested": 1,
            "frames_analyzed": summary.get("frames_analyzed", len(frame_results)),
            "fall_detected": summary.get("fall_detected", False),
            "fall_score": summary.get("fall_score", 0.0),
            "label": summary.get("label"),
            "details": {
                "avg_fall_score": summary.get("avg_fall_score"),
                "fall_frames": summary.get("fall_frames"),
                "fall_frame_ratio": summary.get("fall_frame_ratio"),
                "max_consecutive_fall_frames": summary.get("max_consecutive_fall_frames"),
                "best_frame": summary.get("best_frame"),
                "last_frame_error": last_error or ("No camera frame available for fast-mode inference" if not frame_results else None),
                "frame_read_failures": frame_read_failures,
                "detection_mode": "fast_cached_frame",
            },
        }
        if not frame_results and last_error:
            result["error"] = last_error
        elif not frame_results:
            result["error"] = "No camera frame available for fast-mode inference"
        _update_fall_detection_snapshot(result)
        return result, 200

    def _capture_frame_with_fallback():
        shared_wait_s = 0.35 if frame_count == 0 else 0.2
        frame_bgr, frame_source, frame_err = _read_camera_frame(requested_source, wait_timeout_s=shared_wait_s)
        if frame_bgr is not None:
            return frame_bgr, frame_source, frame_err

        # For live camera devices (e.g. /dev/video0), avoid opening a second
        # capture handle because it can contend with the shared feed.
        if not is_file_source:
            return None, frame_source, frame_err

        if open_camera is None:
            return None, frame_source, frame_err

        try:
            cap, fallback_source = open_camera(requested_source, CAMERA_WIDTH, CAMERA_HEIGHT)
        except Exception as exc:
            return None, frame_source, frame_err or str(exc)

        try:
            fallback_frame = None
            fallback_error = frame_err
            for _ in range(3):
                ok, candidate_frame = cap.read()
                if ok and candidate_frame is not None:
                    fallback_frame = candidate_frame
                    break
                time.sleep(0.03)
            if fallback_frame is None:
                return None, fallback_source, fallback_error or f"Camera frame read failed for source: {requested_source}"
            return fallback_frame, fallback_source, None
        finally:
            try:
                cap.release()
            except Exception:
                pass

    while (time.time() - start_ts) < detection_window_s and frame_count < max_frames:
        frame_bgr, actual_source, frame_err = _capture_frame_with_fallback()
        if frame_bgr is None:
            frame_read_failures += 1
            last_error = frame_err or f"Camera frame read failed ({frame_read_failures}/{max_read_failures})"

            # When no frame has been captured yet, allow a longer grace period
            # before aborting the detection window.
            elapsed = time.time() - start_ts
            if frame_count == 0 and elapsed < first_frame_wait_s:
                time.sleep(0.03 if is_file_source else 0.07)
                continue

            # After at least one frame is seen, stop earlier on sustained failures.
            if frame_count > 0 and frame_read_failures >= max_read_failures:
                break

            # If no frame ever arrived, keep trying a bit longer before giving up.
            if frame_count == 0 and frame_read_failures >= (max_read_failures * 3):
                break

            time.sleep(0.01 if is_file_source else 0.05)
            continue

        try:
            frame_read_failures = 0

            input_details = input_meta if isinstance(input_meta, dict) else interpreter.get_input_details()[0]
            output_details = interpreter.get_output_details()[0]
            input_dtype = input_details.get("dtype", np.uint8)

            input_tensor, _ = preprocess_frame_bgr(frame_bgr, target_size, input_dtype=input_dtype)

            # Keep lock scope minimal: protect only interpreter state mutations.
            with MODEL_LOCK:
                interpreter.set_tensor(input_details["index"], input_tensor)
                interpreter.invoke()
                output = interpreter.get_tensor(output_details["index"])

            keypoints = extract_keypoints(output)
            pose_result = analyze_pose(keypoints)
            pose_result["frame_index"] = frame_count
            frame_results.append(pose_result)
            frame_count += 1
        except Exception as exc:
            frame_read_failures += 1
            last_error = f"Fall inference failed: {exc}"
            if frame_count > 0 and frame_read_failures >= max_read_failures:
                break
            time.sleep(0.01 if is_file_source else 0.03)
            continue

    # Final safety net: if no frame was analyzed in the window, try one cached frame.
    if frame_count == 0:
        cached_frame, cached_source, cached_err = _get_cached_camera_frame(requested_source)
        if cached_frame is not None:
            try:
                input_details = input_meta if isinstance(input_meta, dict) else interpreter.get_input_details()[0]
                output_details = interpreter.get_output_details()[0]
                input_dtype = input_details.get("dtype", np.uint8)

                input_tensor, _ = preprocess_frame_bgr(cached_frame, target_size, input_dtype=input_dtype)
                with MODEL_LOCK:
                    interpreter.set_tensor(input_details["index"], input_tensor)
                    interpreter.invoke()
                    output = interpreter.get_tensor(output_details["index"])

                keypoints = extract_keypoints(output)
                pose_result = analyze_pose(keypoints)
                pose_result["frame_index"] = frame_count
                frame_results.append(pose_result)
                frame_count += 1
                actual_source = cached_source
            except Exception as exc:
                last_error = f"Fall inference failed on cached frame: {exc}"
        elif not last_error:
            last_error = cached_err

    # Last-resort fallback: for live sources, attempt a one-shot direct capture.
    if frame_count == 0 and not is_file_source and open_camera is not None:
        cap = None
        try:
            cap, fallback_source = open_camera(requested_source, CAMERA_WIDTH, CAMERA_HEIGHT)
            direct_frame = None
            for _ in range(4):
                ok, candidate = cap.read()
                if ok and candidate is not None:
                    direct_frame = candidate
                    break
                time.sleep(0.03)

            if direct_frame is not None:
                input_details = input_meta if isinstance(input_meta, dict) else interpreter.get_input_details()[0]
                output_details = interpreter.get_output_details()[0]
                input_dtype = input_details.get("dtype", np.uint8)

                input_tensor, _ = preprocess_frame_bgr(direct_frame, target_size, input_dtype=input_dtype)
                with MODEL_LOCK:
                    interpreter.set_tensor(input_details["index"], input_tensor)
                    interpreter.invoke()
                    output = interpreter.get_tensor(output_details["index"])

                keypoints = extract_keypoints(output)
                pose_result = analyze_pose(keypoints)
                pose_result["frame_index"] = frame_count
                frame_results.append(pose_result)
                frame_count += 1
                actual_source = fallback_source
            elif not last_error:
                last_error = f"Direct camera fallback could not read frame from source: {requested_source}"
        except Exception as exc:
            if not last_error:
                last_error = f"Direct camera fallback failed: {exc}"
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass

    try:
        summary = summarize_detection_window(frame_results) if summarize_detection_window is not None else {
            "frames_analyzed": len(frame_results),
            "fall_detected": False,
            "fall_score": 0.0,
            "label": "unsupported",
        }
    except Exception as exc:
        last_error = f"Fall summary failed: {exc}"
        summary = {
            "frames_analyzed": len(frame_results),
            "fall_detected": False,
            "fall_score": 0.0,
            "label": "no_frames" if not frame_results else "uncertain",
            "avg_fall_score": None,
            "fall_frames": 0,
            "fall_frame_ratio": None,
            "max_consecutive_fall_frames": 0,
            "best_frame": None,
        }

    summary = summary if isinstance(summary, dict) else {
        "frames_analyzed": len(frame_results),
        "fall_detected": False,
        "fall_score": 0.0,
        "label": "unsupported",
    }
    result = {
        "success": True,
        "timestamp": _now_iso(),
        "model": STATE.get("model_name"),
        "runtime": runtime,
        "camera_source": str(actual_source),
        "camera_ready": bool(frame_results),
        "duration_s": round(time.time() - start_ts, 3),
        "frames_requested": max_frames,
        "frames_analyzed": summary.get("frames_analyzed", len(frame_results)),
        "fall_detected": summary.get("fall_detected", False),
        "fall_score": summary.get("fall_score", 0.0),
        "label": summary.get("label"),
        "details": {
            "avg_fall_score": summary.get("avg_fall_score"),
            "fall_frames": summary.get("fall_frames"),
            "fall_frame_ratio": summary.get("fall_frame_ratio"),
            "max_consecutive_fall_frames": summary.get("max_consecutive_fall_frames"),
            "best_frame": summary.get("best_frame"),
            "last_frame_error": last_error or ("No camera frames captured within detection window" if not frame_results else None),
            "frame_read_failures": frame_read_failures,
        },
    }
    if not frame_results and last_error:
        result["error"] = last_error
    elif not frame_results:
        result["error"] = "No camera frames captured within detection window"
    _update_fall_detection_snapshot(result)
    return result, 200


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
        "tensorrt_available": TENSORRT_AVAILABLE,
        "cuda_runtime_available": CUDA_RUNTIME_AVAILABLE,
        "tensorrt_engine_enabled": TENSORRT_ENGINE_ENABLED,
        "numpy_available": np is not None
    }
    response["fall_detection"]["enabled"] = _is_fall_detection_model_info()
    response["benchmark_metrics"]["runtime"] = STATE.get("runtime")
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
        "fall_detection": {
            "enabled": STATE.get("fall_detection", {}).get("enabled"),
            "camera_ready": STATE.get("fall_detection", {}).get("camera_ready"),
            "camera_source": STATE.get("fall_detection", {}).get("camera_source"),
            "fall_detected": STATE.get("fall_detection", {}).get("fall_detected"),
            "fall_score": STATE.get("fall_detection", {}).get("fall_score"),
            "label": STATE.get("fall_detection", {}).get("label"),
            "updated_at": STATE.get("fall_detection", {}).get("updated_at"),
            "last_error": STATE.get("fall_detection", {}).get("last_error"),
        },
        "benchmark": {
            "status": STATE.get("benchmark_metrics", {}).get("status"),
            "runtime": STATE.get("benchmark_metrics", {}).get("runtime"),
            "latency_avg_s": STATE.get("benchmark_metrics", {}).get("latency_avg_s"),
            "latency_p95_s": STATE.get("benchmark_metrics", {}).get("latency_p95_s"),
            "throughput_iter_per_s": STATE.get("benchmark_metrics", {}).get("throughput_iter_per_s"),
            "benchmark_runs": STATE.get("benchmark_metrics", {}).get("benchmark_runs"),
            "updated_at": STATE.get("benchmark_metrics", {}).get("updated_at"),
            "last_error": STATE.get("benchmark_metrics", {}).get("last_error"),
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
        if not (TENSORRT_ENGINE_ENABLED and TENSORRT_AVAILABLE and CUDA_RUNTIME_AVAILABLE):
            return jsonify({"error": "ONNX Runtime not installed and TensorRT engine path unavailable"}), 500

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
        reset_fall_detection_metrics()
        reset_benchmark_metrics()

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
        reset_fall_detection_metrics()
        reset_benchmark_metrics()

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


@app.route("/camera/fall-detect", methods=["POST"])
def camera_fall_detect():
    """Run MoveNet-based fall detection on live webcam frames."""
    try:
        data = request.get_json(silent=True) or {}
        result, status_code = _run_camera_fall_detection(
            duration_s=data.get("duration_s"),
            max_frames=data.get("max_frames"),
            camera_source=data.get("camera_device") or data.get("camera_source"),
            fast_mode=bool(data.get("fast_mode")),
        )
        return jsonify(result), status_code
    except Exception as exc:
        reset_fall_detection_metrics(error=exc)
        return jsonify({
            "success": False,
            "error": f"Camera fall-detect endpoint failed: {exc}",
            "frames_analyzed": 0,
            "fall_detected": False,
            "fall_score": 0.0,
            "label": "no_frames",
            "details": {
                "last_frame_error": str(exc),
            },
        }), 500


@app.route("/camera/upload-video", methods=["POST"])
def camera_upload_video():
    file = request.files.get("file")
    if file is None or not getattr(file, "filename", ""):
        return jsonify({"success": False, "error": "Missing video file"}), 400

    filename = str(file.filename)
    if not filename.lower().endswith(".mp4"):
        return jsonify({"success": False, "error": "Only .mp4 files are supported"}), 400

    upload_dir = _safe_video_upload_dir()
    safe_name = f"{int(time.time())}_{uuid.uuid4().hex}.mp4"
    save_path = os.path.join(upload_dir, safe_name)

    try:
        file.save(save_path)
    except Exception as exc:
        return jsonify({"success": False, "error": f"Failed to save uploaded video: {exc}"}), 500

    return jsonify({"success": True, "video_path": save_path})


@app.route("/camera/snapshot", methods=["GET"])
def camera_snapshot():
    """Return a JPEG snapshot from the USB camera for monitoring."""
    try:
        annotate = str(request.args.get("annotate", "1")).strip().lower() not in {"0", "false", "no", "off"}
        frame_bytes, actual_source = _capture_camera_snapshot_result(
            camera_source=request.args.get("camera_device") or request.args.get("camera_source") or CAMERA_DEVICE,
            annotate=annotate,
        )
        response = Response(frame_bytes, mimetype="image/jpeg")
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["X-Camera-Source"] = str(actual_source)
        return response
    except Exception as exc:
        reset_fall_detection_metrics(error=exc)
        return jsonify({
            "success": False,
            "error": f"Unable to capture camera snapshot: {exc}",
        }), 500


@app.route("/camera/stream", methods=["GET"])
def camera_stream():
    """Return a multipart MJPEG stream from the USB camera."""
    annotate = str(request.args.get("annotate", "1")).strip().lower() not in {"0", "false", "no", "off"}
    camera_source = request.args.get("camera_device") or request.args.get("camera_source") or CAMERA_DEVICE
    try:
        fps = float(request.args.get("fps", "5"))
    except (TypeError, ValueError):
        fps = 5.0

    try:
        jpeg_quality = int(request.args.get("quality", "75"))
    except (TypeError, ValueError):
        jpeg_quality = 75

    fps = max(0.5, min(fps, 24.0))
    jpeg_quality = max(50, min(jpeg_quality, 90))
    boundary = "frame"
    cv2 = _get_cv2_module()

    if cv2 is None:
        return jsonify({
            "success": False,
            "error": "OpenCV is not available in the agent image.",
        }), 500

    def generate():
        frame_delay = 1.0 / fps if fps > 0 else 0.2
        while True:
            try:
                frame_bgr, actual_source, frame_err = _read_camera_frame(camera_source)
                if frame_bgr is None:
                    reset_fall_detection_metrics(error=frame_err or "Camera frame read failed")
                    time.sleep(0.2)
                    continue

                if annotate:
                    frame_bgr = _draw_overlay_lines(frame_bgr, _build_camera_overlay_lines())

                ok, encoded = cv2.imencode(
                    ".jpg",
                    frame_bgr,
                    [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
                )
                if not ok:
                    time.sleep(0.05)
                    continue

                _mark_camera_ready(actual_source)
                frame_bytes = encoded.tobytes()
                header = (
                    f"--{boundary}\r\n"
                    "Content-Type: image/jpeg\r\n"
                    f"Content-Length: {len(frame_bytes)}\r\n"
                    f"X-Camera-Source: {actual_source}\r\n"
                    "\r\n"
                ).encode("utf-8")
                yield header + frame_bytes + b"\r\n"
            except GeneratorExit:
                return
            except Exception:
                time.sleep(0.25)
            finally:
                time.sleep(frame_delay)

    response = Response(generate(), mimetype=f"multipart/x-mixed-replace; boundary={boundary}")
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    return response


@app.route("/benchmark", methods=["POST"])
def benchmark_model():
    """Benchmark the currently deployed model on this edge device."""
    data = request.get_json(silent=True) or {}

    try:
        warmup_runs = int(data.get("warmup_runs", 5))
        benchmark_runs = int(data.get("benchmark_runs", 30))
    except (TypeError, ValueError):
        return jsonify({"success": False, "error": "warmup_runs and benchmark_runs must be numeric"}), 400

    try:
        _update_benchmark_snapshot(status="running", error=None)
        result = benchmark_loaded_model(warmup_runs=warmup_runs, benchmark_runs=benchmark_runs)
        return jsonify(result)
    except Exception as exc:
        _update_benchmark_snapshot(status="error", error=exc)
        return jsonify({
            "success": False,
            "error": f"Unable to benchmark current model: {exc}",
        }), 500


@app.route("/predict", methods=["POST"])
def predict():
    """
    Inference endpoint: runs a forward pass using TFLite (cached after /deploy or /start).
    """
    if STATE.get("status") not in ("running", "ready"):
        return jsonify({"error": "Model not running"}), 400

    if _is_fall_detection_model_info():
        data = request.get_json(silent=True) or {}
        result, status_code = _run_camera_fall_detection(
            duration_s=data.get("duration_s"),
            max_frames=data.get("max_frames"),
            camera_source=data.get("camera_device") or data.get("camera_source"),
        )
        return jsonify(result), status_code

    _ensure_numpy()

    try:
        with MODEL_LOCK:
            if LOADED_INPUT_SIZE is None or (LOADED_INTERPRETER is None and LOADED_ONNX_SESSION is None and LOADED_TRT_RUNNER is None):
                return jsonify({"error": "Model not loaded in memory"}), 400
            interpreter = LOADED_INTERPRETER
            session = LOADED_ONNX_SESSION
            trt_runner = LOADED_TRT_RUNNER
            input_size = LOADED_INPUT_SIZE
            input_layout = LOADED_INPUT_LAYOUT
            input_meta = LOADED_INPUT_META
            runtime = LOADED_RUNTIME

        c, h, w = input_size
        if runtime == "onnx":
            dummy_input = _build_dummy_input(input_layout or "nchw", input_size, getattr(input_meta, "shape", None), np.float32)
            outputs = session.run(None, {input_meta.name: dummy_input})
            output = outputs[0]
        elif runtime == "tensorrt":
            if trt_runner is None:
                raise RuntimeError("TensorRT runner is not loaded")
            output = trt_runner.infer()
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


def _run_loaded_model_once_locked(runtime, interpreter, session, trt_runner, input_size, input_layout, input_meta, dummy_input=None):
    if runtime == "tensorrt":
        if trt_runner is None:
            raise RuntimeError("TensorRT runner is not loaded")
        return trt_runner.infer(dummy_input)

    if runtime == "onnx":
        if session is None or input_meta is None:
            raise RuntimeError("ONNX session is not loaded")
        actual_input = dummy_input
        if actual_input is None:
            actual_input = _build_dummy_input(input_layout or "nchw", input_size, getattr(input_meta, "shape", None), np.float32)
        outputs = session.run(None, {input_meta.name: actual_input})
        return outputs[0] if outputs else None

    if interpreter is None:
        raise RuntimeError("TFLite interpreter is not loaded")
    input_details = interpreter.get_input_details()[0]
    actual_input = dummy_input
    if actual_input is None:
        actual_input = _build_dummy_input(input_layout or "nhwc", input_size, input_details.get("shape"), input_details.get("dtype", np.float32))
    interpreter.set_tensor(input_details["index"], actual_input)
    interpreter.invoke()
    output_details = interpreter.get_output_details()[0]
    return interpreter.get_tensor(output_details["index"])


def benchmark_loaded_model(warmup_runs: int = 5, benchmark_runs: int = 30):
    """
    Benchmark the currently loaded model on-device using the exact runtime in memory.
    """
    _ensure_numpy()

    warmup_runs = max(1, int(warmup_runs))
    benchmark_runs = max(4, int(benchmark_runs))

    if STATE.get("status") not in ("running", "ready"):
        raise RuntimeError("Model is not ready for benchmarking")

    with MODEL_LOCK:
        runtime = LOADED_RUNTIME
        interpreter = LOADED_INTERPRETER
        session = LOADED_ONNX_SESSION
        trt_runner = LOADED_TRT_RUNNER
        input_size = LOADED_INPUT_SIZE
        input_layout = LOADED_INPUT_LAYOUT
        input_meta = LOADED_INPUT_META

        if input_size is None or (interpreter is None and session is None and trt_runner is None):
            raise RuntimeError("No loaded model is available for benchmarking")

        if runtime == "onnx":
            dummy_input = _build_dummy_input(input_layout or "nchw", input_size, getattr(input_meta, "shape", None), np.float32)
        elif runtime == "tensorrt":
            dummy_input = None
        else:
            input_details = interpreter.get_input_details()[0]
            dummy_input = _build_dummy_input(
                input_layout or "nhwc",
                input_size,
                input_details.get("shape"),
                input_details.get("dtype", np.float32),
            )

        for _ in range(warmup_runs):
            _run_loaded_model_once_locked(runtime, interpreter, session, trt_runner, input_size, input_layout, input_meta, dummy_input)

        samples_s = []
        for _ in range(benchmark_runs):
            started = time.perf_counter()
            _run_loaded_model_once_locked(runtime, interpreter, session, trt_runner, input_size, input_layout, input_meta, dummy_input)
            ended = time.perf_counter()
            samples_s.append(max(ended - started, 0.0))

    if not samples_s:
        raise RuntimeError("Benchmark produced no samples")

    lat_avg = float(np.mean(samples_s))
    lat_std = float(np.std(samples_s))
    lat_p50 = float(np.percentile(samples_s, 50))
    lat_p95 = float(np.percentile(samples_s, 95))
    throughput = float(1.0 / lat_avg) if lat_avg > 0 else None

    result = {
        "success": True,
        "timestamp": _now_iso(),
        "model_name": STATE.get("model_name"),
        "runtime": runtime,
        "warmup_runs": warmup_runs,
        "benchmark_runs": benchmark_runs,
        "iterations": len(samples_s),
        "latency_avg_s": round(lat_avg, 6),
        "latency_std_s": round(lat_std, 6),
        "latency_p50_s": round(lat_p50, 6),
        "latency_p95_s": round(lat_p95, 6),
        "throughput_iter_per_s": round(throughput, 4) if throughput is not None else None,
        "input_size": list(input_size),
        "input_layout": input_layout,
        "sample_latencies_ms": [round(sample * 1000.0, 3) for sample in samples_s[: min(10, len(samples_s))]],
    }

    model_info = STATE.get("model_info") or {}
    if isinstance(model_info, dict):
        model_info["measured_latency_avg_s"] = result["latency_avg_s"]
        model_info["measured_latency_p95_s"] = result["latency_p95_s"]
        model_info["measured_throughput_iter_per_s"] = result["throughput_iter_per_s"]
        model_info["benchmark_runs"] = result["benchmark_runs"]
        model_info["benchmark_updated_at"] = result["timestamp"]
        set_state(model_info=model_info)

    _update_benchmark_snapshot(result, status="completed", error=None)
    return result


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


def _check_cuda_status(status, action: str):
    ok_value = getattr(cudart.cudaError_t, "cudaSuccess", 0) if CUDA_RUNTIME_AVAILABLE else 0
    if int(status) != int(ok_value):
        raise RuntimeError(f"CUDA call failed while {action}: {status}")


def _to_concrete_shape(shape, fallback_shape):
    concrete = []
    for idx, dim in enumerate(shape):
        try:
            value = int(dim)
        except Exception:
            value = -1
        if value <= 0:
            value = int(fallback_shape[idx]) if idx < len(fallback_shape) else 1
        concrete.append(max(1, value))
    return tuple(concrete)


class _TensorRTRunner:
    """Minimal TensorRT engine runner using CUDA runtime API."""

    def __init__(self, engine_path, input_size):
        if not TENSORRT_AVAILABLE:
            raise RuntimeError("TensorRT Python package is not available")
        if not CUDA_RUNTIME_AVAILABLE:
            raise RuntimeError("cuda-python runtime bindings are not available")
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"TensorRT engine file not found: {engine_path}")

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            serialized_engine = f.read()
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        if engine is None:
            raise RuntimeError("Failed to deserialize TensorRT engine")

        context = engine.create_execution_context()
        if context is None:
            raise RuntimeError("Failed to create TensorRT execution context")

        self.logger = logger
        self.runtime = runtime
        self.engine = engine
        self.context = context
        self.engine_path = engine_path
        self.input_size = input_size
        self.input_index = None
        self.output_indices = []

        for binding_idx in range(self.engine.num_bindings):
            if self.engine.binding_is_input(binding_idx):
                self.input_index = binding_idx
            else:
                self.output_indices.append(binding_idx)

        if self.input_index is None:
            raise RuntimeError("TensorRT engine does not expose an input binding")
        if not self.output_indices:
            raise RuntimeError("TensorRT engine does not expose any output bindings")

        raw_shape = tuple(self.engine.get_binding_shape(self.input_index))
        self.layout, self.normalized_input_size = _infer_layout_and_size(raw_shape, input_size)

    def close(self):
        self.context = None
        self.engine = None
        self.runtime = None

    def infer(self, dummy_input=None):
        if self.context is None or self.engine is None:
            raise RuntimeError("TensorRT runner is closed")

        if dummy_input is None:
            dummy_input = _build_dummy_input(
                self.layout or "nchw",
                self.normalized_input_size,
                self.engine.get_binding_shape(self.input_index),
                np.float32,
            )

        input_array = np.ascontiguousarray(dummy_input)
        input_dtype = np.dtype(trt.nptype(self.engine.get_binding_dtype(self.input_index)))
        if input_array.dtype != input_dtype:
            input_array = input_array.astype(input_dtype, copy=False)

        if any(int(dim) <= 0 for dim in self.engine.get_binding_shape(self.input_index)):
            if not self.context.set_binding_shape(self.input_index, tuple(int(dim) for dim in input_array.shape)):
                raise RuntimeError("Failed to set dynamic input shape for TensorRT context")

        device_ptrs = {}
        bindings = [0] * self.engine.num_bindings
        output_buffers = {}

        try:
            for binding_idx in range(self.engine.num_bindings):
                dtype = np.dtype(trt.nptype(self.engine.get_binding_dtype(binding_idx)))
                if binding_idx == self.input_index:
                    host_array = input_array
                else:
                    raw_shape = tuple(int(x) for x in self.context.get_binding_shape(binding_idx))
                    concrete_shape = _to_concrete_shape(raw_shape, [1] * len(raw_shape))
                    host_array = np.empty(concrete_shape, dtype=dtype)
                    output_buffers[binding_idx] = host_array

                nbytes = int(host_array.nbytes)
                status, ptr = cudart.cudaMalloc(nbytes)
                _check_cuda_status(status, f"allocating binding {binding_idx}")
                device_ptrs[binding_idx] = ptr
                bindings[binding_idx] = int(ptr)

                if binding_idx == self.input_index:
                    status = cudart.cudaMemcpy(
                        ptr,
                        input_array.ctypes.data,
                        nbytes,
                        cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                    )[0]
                    _check_cuda_status(status, "copying input to device")

            if not self.context.execute_v2(bindings):
                raise RuntimeError("TensorRT execution failed")

            first_output_idx = self.output_indices[0]
            output_array = output_buffers[first_output_idx]
            out_bytes = int(output_array.nbytes)
            status = cudart.cudaMemcpy(
                output_array.ctypes.data,
                device_ptrs[first_output_idx],
                out_bytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
            )[0]
            _check_cuda_status(status, "copying output to host")
            return output_array
        finally:
            for ptr in device_ptrs.values():
                try:
                    cudart.cudaFree(ptr)
                except Exception:
                    pass


def _build_tensorrt_engine(onnx_path, engine_path, input_size):
    if not TENSORRT_AVAILABLE:
        raise RuntimeError("TensorRT Python package is not available")

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errors = []
            for idx in range(parser.num_errors):
                errors.append(str(parser.get_error(idx)))
            detail = " | ".join(errors) if errors else "unknown ONNX parser error"
            raise RuntimeError(f"TensorRT ONNX parse failed: {detail}")

    config = builder.create_builder_config()
    if hasattr(config, "set_memory_pool_limit"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(TENSORRT_WORKSPACE_BYTES))
    else:
        config.max_workspace_size = int(TENSORRT_WORKSPACE_BYTES)

    if TENSORRT_FP16_ENABLED and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    has_dynamic_input = False
    profile = builder.create_optimization_profile()
    for input_idx in range(network.num_inputs):
        tensor = network.get_input(input_idx)
        shape = tuple(int(dim) for dim in tensor.shape)
        if any(dim <= 0 for dim in shape):
            has_dynamic_input = True
            layout, normalized = _infer_layout_and_size(shape, input_size)
            c, h, w = normalized
            fallback_shape = (1, h, w, c) if layout == "nhwc" else (1, c, h, w)
            concrete = _to_concrete_shape(shape, fallback_shape)
            profile.set_shape(tensor.name, concrete, concrete, concrete)

    if has_dynamic_input:
        config.add_optimization_profile(profile)

    if hasattr(builder, "build_serialized_network"):
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("TensorRT build_serialized_network returned None")
        engine_bytes = bytes(serialized)
    else:
        engine = builder.build_engine(network, config)
        if engine is None:
            raise RuntimeError("TensorRT build_engine returned None")
        engine_bytes = bytes(engine.serialize())

    with open(engine_path, "wb") as f:
        f.write(engine_bytes)


def _load_tensorrt_engine(engine_path, input_size):
    _ensure_numpy()
    runner = _TensorRTRunner(engine_path, input_size)
    _ = runner.infer()
    input_meta = {
        "name": str(runner.engine.get_binding_name(runner.input_index)),
        "shape": list(runner.engine.get_binding_shape(runner.input_index)),
        "_trt_runner": runner,
    }
    return runner, input_meta, runner.normalized_input_size, runner.layout

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
        runtime = LOADED_RUNTIME
        interpreter = LOADED_INTERPRETER
        session = LOADED_ONNX_SESSION
        trt_runner = LOADED_TRT_RUNNER
        input_meta = LOADED_INPUT_META
        input_size = LOADED_INPUT_SIZE
        input_layout = LOADED_INPUT_LAYOUT
    if input_size is None or (interpreter is None and session is None and trt_runner is None):
        return {"success": False, "error": "Runtime is not loaded"}

    start_uj = _read_uint(energy_file)
    start_ts = time.time()
    iters = 0
    while (time.time() - start_ts) < duration_s:
        try:
            with MODEL_LOCK:
                if runtime == "tflite":
                    _run_single_inference_locked(interpreter, input_size)
                else:
                    _run_loaded_model_once_locked(
                        runtime,
                        interpreter,
                        session,
                        trt_runner,
                        input_size,
                        input_layout,
                        input_meta,
                        None,
                    )
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

    Loads the active model artifact (.tflite/.onnx/.engine) and runs warmup forward.
    Keeps the selected runtime handle in memory for /predict.
    """
    try:
        global LOADED_INTERPRETER, LOADED_ONNX_SESSION, LOADED_TRT_RUNNER, LOADED_RUNTIME
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
            if model_path.endswith(".engine"):
                try:
                    trt_runner, input_meta, normalized_input_size, layout = _load_tensorrt_engine(model_path, input_size)
                    LOADED_TRT_RUNNER = trt_runner
                    LOADED_RUNTIME = "tensorrt"
                    LOADED_INPUT_META = input_meta
                except Exception as trt_engine_exc:
                    onnx_fallback_path = os.path.splitext(model_path)[0] + ".onnx"
                    if not os.path.exists(onnx_fallback_path):
                        raise RuntimeError(f"Failed to load TensorRT engine and no ONNX fallback found: {trt_engine_exc}")
                    log(f"Failed to load TensorRT engine, falling back to ONNX artifact: {trt_engine_exc}")
                    model_path = onnx_fallback_path
                    session, input_meta, normalized_input_size, layout = _load_onnx_model(model_path, input_size)
                    LOADED_ONNX_SESSION = session
                    LOADED_RUNTIME = "onnx"
                    LOADED_INPUT_META = input_meta
            elif model_path.endswith(".onnx"):
                trt_ready = TENSORRT_ENGINE_ENABLED and TENSORRT_AVAILABLE and CUDA_RUNTIME_AVAILABLE
                if trt_ready:
                    engine_path = os.path.splitext(model_path)[0] + ".engine"
                    try:
                        if not os.path.exists(engine_path) or os.path.getmtime(engine_path) < os.path.getmtime(model_path):
                            log("Building TensorRT engine from ONNX...")
                            _build_tensorrt_engine(model_path, engine_path, input_size)
                        trt_runner, input_meta, normalized_input_size, layout = _load_tensorrt_engine(engine_path, input_size)
                        LOADED_TRT_RUNNER = trt_runner
                        LOADED_RUNTIME = "tensorrt"
                        LOADED_INPUT_META = input_meta
                        model_path = engine_path
                    except Exception as trt_exc:
                        log(f"TensorRT engine path unavailable, falling back to ONNX Runtime: {trt_exc}")
                        session, input_meta, normalized_input_size, layout = _load_onnx_model(model_path, input_size)
                        LOADED_ONNX_SESSION = session
                        LOADED_RUNTIME = "onnx"
                        LOADED_INPUT_META = input_meta
                else:
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
        reset_fall_detection_metrics()
        reset_benchmark_metrics()

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
        ext = os.path.splitext(active_model_path)[1].lower()
        STATE["runtime"] = "tensorrt" if ext == ".engine" else ext.lstrip(".")
        size = os.path.getsize(active_model_path)
        log(f"Found existing model: {size} bytes")
        if STATE.get("model_name"):
            log(f"Current model: {STATE['model_name']}")
    reset_fall_detection_metrics()
    reset_benchmark_metrics()

    if FNB58_AUTO_START:
        start_fnb58_monitor()
        ensure_fnb58_retry_loop()
    
    # Start Flask server (threaded to keep stream responsive during fall checks)
    app.run(host="0.0.0.0", port=8000, threaded=True)
