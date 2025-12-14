import os
import threading
from datetime import datetime
from flask import Flask, request, jsonify
import requests

import shutil
import time

app = Flask(__name__)

# Persistent storage on BBB
MODEL_DIR = "/data/models"
os.makedirs(MODEL_DIR, exist_ok=True)
CURRENT_MODEL_PATH = os.path.join(MODEL_DIR, "current_model.bin")
ENERGY_HISTORY_LIMIT = 40


def _create_energy_metrics(budget=None):
    return {
        "budget_mwh": budget,
        "latest_mwh": None,
        "avg_mwh": None,
        "status": "unknown",
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


def set_state(**kwargs):
    """Update agent state"""
    for k, v in kwargs.items():
        STATE[k] = v
    STATE["last_update"] = datetime.utcnow().isoformat() + "Z"


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
    raw = _read_text("/sys/class/thermal/thermal_zone0/temp")
    if not raw:
        return None
    try:
        v = float(raw)
        return round(v / 1000.0, 1) if v > 200 else round(v, 1)
    except Exception:
        return None


@app.route("/status", methods=["GET"])
def status():
    """Return current agent status"""
    return jsonify(STATE)


@app.route("/metrics", methods=["GET"])
def metrics():
    """Lightweight device metrics without native deps (works on armv7)."""
    cpu = _cpu_percent_from_proc()
    mem = _memory_from_proc()
    storage = _storage_for_path("/")
    temp = _temperature_c()

    return jsonify({
        "success": True,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "cpu": {"percent": round(cpu, 2) if cpu is not None else None},
        "memory": mem,
        "storage": storage,
        "temperature_c": temp,
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

    try:
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

        # Replace current model
        os.replace(tmp_path, CURRENT_MODEL_PATH)
        file_size = os.path.getsize(CURRENT_MODEL_PATH)
        
        log(f"Model downloaded: {file_size} bytes")
        set_state(
            status="ready",
            model_name=model_name,
            model_info=model_info,
            error=None,
            energy_metrics=STATE["energy_metrics"]
        )

        # Auto-start model inference in background
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
    if STATE["status"] != "ready":
        return jsonify({"error": "Model not ready"}), 400
    
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
    Inference endpoint (placeholder)
    In real implementation, load model and run inference
    """
    if STATE["status"] != "running":
        return jsonify({"error": "Model not running"}), 400
    
    # Placeholder: Real implementation would process input and return prediction
    return jsonify({
        "model": STATE["model_name"],
        "status": "inference_placeholder",
        "message": "Implement actual inference logic here"
    })


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
    
    In real implementation:
    1. Load model from CURRENT_MODEL_PATH
    2. Initialize inference engine (PyTorch, ONNX, TFLite, etc.)
    3. Set status to "running"
    4. Keep model loaded for predictions
    """
    try:
        log("Starting model inference...")
        set_state(status="running", inference_active=True, error=None)
        
        # Placeholder for actual model loading
        # Example:
        # import torch
        # model = torch.load(CURRENT_MODEL_PATH)
        # model.eval()
        
        log(f"Model '{STATE['model_name']}' is now running")
        log(f"Estimated energy draw: {STATE['model_info'].get('energy_avg_mwh', 'N/A')} mWh")
        budget = STATE.get("energy_metrics", {}).get("budget_mwh")
        if budget is not None:
            log(f"Energy budget assigned: {budget} mWh")
        
        # In real app, keep model loaded and wait for inference requests
        # For now, just mark as running
        
    except Exception as e:
        log(f"Error starting inference: {str(e)}")
        set_state(status="error", error=str(e), inference_active=False)


if __name__ == "__main__":
    log("BBB ML Agent starting...")
    log(f"Model directory: {MODEL_DIR}")
    
    # Check if there's an existing model
    if os.path.exists(CURRENT_MODEL_PATH):
        size = os.path.getsize(CURRENT_MODEL_PATH)
        log(f"Found existing model: {size} bytes")
    
    # Start Flask server
    app.run(host="0.0.0.0", port=8000)
