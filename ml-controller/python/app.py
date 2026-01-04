import os
import re
from urllib.parse import urlparse
from flask import Flask, request, render_template, jsonify, send_from_directory, make_response
import requests
from requests.utils import requote_uri
from model_analyzer import ModelAnalyzer
from energy_predictor_service import EnergyPredictorService
from log_manager import LogManager
from yolo_models import (
    get_all_yolo_models,
    get_yolo_model,
    get_yolo_models_by_family,
    get_recommended_yolo_models
)

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates'))

# Load environment variables from .env (non-destructive: keep existing env vars)
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # ml-controller root
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
CSV_PATH = os.path.join(DATA_DIR, "247_models_benchmark_jetson.csv")  # Updated to full dataset
RPI5_CSV_PATH = os.path.join(DATA_DIR, "27_models_benchmark_rpi5.csv")  # Raspberry Pi 5 dataset
LOG_FILE_PATH = os.path.join(DATA_DIR, "deployment_logs.json")
PREFERRED_ARTIFACT_EXTS = [".pth", ".pt", ".onnx", ".tflite", ".bin"]
BALENA_API_BASE = os.getenv("BALENA_API_BASE", "https://api.balena-cloud.com")
BALENA_DEFAULT_TIMEOUT = int(os.getenv("BALENA_API_TIMEOUT", "30"))

# Initialize predictor + analyzer + log manager
predictor_service = EnergyPredictorService(ARTIFACTS_DIR)
analyzer = ModelAnalyzer(CSV_PATH, predictor_service=predictor_service, model_store_dir=MODEL_STORE_DIR, rpi5_csv_path=RPI5_CSV_PATH)
log_manager = LogManager(LOG_FILE_PATH, max_logs=500)


def _normalize_key(value: str) -> str:
    return "".join(ch.lower() for ch in value if ch.isalnum())


def resolve_model_artifact(model_name: str):
    """
    Find a matching model artifact file inside MODEL_STORE_DIR.
    Accepts variations in casing/spacing and tries preferred extensions.
    """
    if not model_name:
        return None

    normalized = _normalize_key(model_name)
    if not normalized:
        return None

    if not os.path.isdir(MODEL_STORE_DIR):
        return None

    # Exact match ignoring casing / non-alphanumerics
    for filename in os.listdir(MODEL_STORE_DIR):
        base, _ = os.path.splitext(filename)
        if _normalize_key(base) == normalized:
            return filename

    # Try preferred extensions with sanitized base names
    provided_base, provided_ext = os.path.splitext(model_name)
    if provided_ext:
        direct_path = os.path.join(MODEL_STORE_DIR, model_name)
        if os.path.exists(direct_path):
            return model_name

    slug = re.sub(r"[^a-zA-Z0-9]+", "_", model_name).strip("_") or model_name
    for ext in PREFERRED_ARTIFACT_EXTS:
        if model_name.lower().endswith(ext.lower()):
            candidate = model_name
        else:
            candidate = f"{model_name}{ext}"

        candidate_path = os.path.join(MODEL_STORE_DIR, candidate)
        if os.path.exists(candidate_path):
            return candidate

        slug_candidate = f"{slug}{ext}"
        slug_path = os.path.join(MODEL_STORE_DIR, slug_candidate)
        if os.path.exists(slug_path):
            return slug_candidate

    return None


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
    endpoint = raw.get("vpn_address") or raw.get("ip_address")
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
        "webconsole_url": raw.get("webconsole_url")
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
    token = token or os.getenv("BALENA_API_TOKEN")
    if not token:
        raise ValueError("BALENA_API_TOKEN chưa được cấu hình")

    limit = max(1, min(int(limit or 50), 200))

    # Build OData query parameters (don't use dict to avoid double encoding)
    select_fields = [
        "id", "device_name", "uuid", "status", "is_online",
        "ip_address", "vpn_address", "os_version",
        "supervisor_version", "last_connectivity_event", "webconsole_url"
    ]
    
    query_parts = [
        f"$select={','.join(select_fields)}",
        "$expand=belongs_to__application($select=app_name,slug)",
        "$orderby=device_name asc",
        f"$top={limit}"
    ]

    filters = []
    if app_slug:
        # If app_slug is numeric, treat as app ID; otherwise use slug filter
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


@app.route("/analytics")
def analytics():
    """Analytics view - now integrated as partial in index.html"""
    return _render_with_cache("index.html", initial_view="analytics")


@app.route("/favicon.ico")
def favicon():
    """Return empty response for favicon to avoid 404 errors"""
    return "", 204


@app.route("/api/models/all", methods=["GET"])
def get_all_models():
    """API: Lấy danh sách tất cả models"""
    try:
        models = analyzer.get_all_models()
        return jsonify({"success": True, "models": models})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/models/recommended", methods=["GET"])
def get_recommended_models():
    """API: Lấy models được recommend cho BBB"""
    try:
        device_type = request.args.get("device_type", "BBB")
        max_energy = float(request.args.get("max_energy", 100))
        
        recommendations = analyzer.get_recommended_models(device_type, max_energy)
        stats = analyzer.get_energy_stats()
        
        return jsonify({
            "success": True,
            "recommendations": recommendations,
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


# ============================================================================
# YOLO DETECTION MODELS API
# ============================================================================

@app.route("/api/yolo/models", methods=["GET"])
def api_get_yolo_models():
    """
    API: Get all YOLO detection models
    
    Query params:
        family (optional): Filter by family (yolov5, yolov8)
    
    Returns:
        {
            "success": true,
            "models": [...],
            "count": 10
        }
    """
    try:
        family = request.args.get("family")
        
        if family:
            models = get_yolo_models_by_family(family)
        else:
            models = get_all_yolo_models()
        
        # Add energy predictions for each model
        for model in models:
            try:
                # Prepare payload for energy prediction
                payload = {
                    "device_type": "jetson_nano_2gb",  # Default to Jetson
                    "params_m": model["params_m"],
                    "gflops": model["gflops"],
                    "gmacs": model["gmacs"],
                    "size_mb": model["size_mb"],
                    "latency_avg_s": model["latency_avg_s"],
                    "throughput_iter_per_s": model["throughput_iter_per_s"]
                }
                
                # Predict energy
                predictions = predictor_service.predict([payload])
                if predictions and predictions[0].get("prediction_mwh"):
                    model["energy_prediction_mwh"] = predictions[0]["prediction_mwh"]
                    model["energy_ci_lower_mwh"] = predictions[0]["ci_lower_mwh"]
                    model["energy_ci_upper_mwh"] = predictions[0]["ci_upper_mwh"]
            except Exception as e:
                # If prediction fails, just skip it
                model["energy_prediction_mwh"] = None
                model["energy_error"] = str(e)
        
        return jsonify({
            "success": True,
            "models": models,
            "count": len(models)
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/yolo/models/<model_id>", methods=["GET"])
def api_get_yolo_model(model_id):
    """
    API: Get specific YOLO model details
    
    Query params:
        device_type (optional): Specify device for energy prediction (default: jetson_nano_2gb)
    
    Returns:
        {
            "success": true,
            "model": {...}
        }
    """
    try:
        model = get_yolo_model(model_id)
        
        if model is None:
            return jsonify({
                "success": False,
                "error": f"YOLO model '{model_id}' not found"
            }), 404
        
        # Add energy prediction
        device_type = request.args.get("device_type", "jetson_nano_2gb")
        
        try:
            payload = {
                "device_type": device_type,
                "params_m": model["params_m"],
                "gflops": model["gflops"],
                "gmacs": model["gmacs"],
                "size_mb": model["size_mb"],
                "latency_avg_s": model["latency_avg_s"],
                "throughput_iter_per_s": model["throughput_iter_per_s"]
            }
            
            predictions = predictor_service.predict([payload])
            if predictions and predictions[0].get("prediction_mwh"):
                model["energy_prediction"] = {
                    "device_type": device_type,
                    "prediction_mwh": predictions[0]["prediction_mwh"],
                    "ci_lower_mwh": predictions[0]["ci_lower_mwh"],
                    "ci_upper_mwh": predictions[0]["ci_upper_mwh"],
                    "model_used": predictions[0].get("model_used"),
                    "mape_pct": predictions[0].get("mape_pct")
                }
        except Exception as e:
            model["energy_prediction"] = {"error": str(e)}
        
        return jsonify({"success": True, "model": model})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/yolo/recommend", methods=["GET"])
def api_recommend_yolo_models():
    """
    API: Get recommended YOLO models based on constraints
    
    Query params:
        max_latency_s (optional): Maximum latency in seconds
        min_accuracy (optional): Minimum mAP@50
        max_params_m (optional): Maximum parameters in millions
        max_energy_mwh (optional): Maximum energy budget in mWh
        device_type (optional): Device type for energy prediction
    
    Returns:
        {
            "success": true,
            "recommendations": [...],
            "filters_applied": {...}
        }
    """
    try:
        # Get filter parameters
        max_latency_s = request.args.get("max_latency_s", type=float)
        min_accuracy = request.args.get("min_accuracy", type=float)
        max_params_m = request.args.get("max_params_m", type=float)
        max_energy_mwh = request.args.get("max_energy_mwh", type=float)
        device_type = request.args.get("device_type", "jetson_nano_2gb")
        
        # Get filtered models
        models = get_recommended_yolo_models(
            max_latency_s=max_latency_s,
            min_accuracy_map50=min_accuracy,
            max_params_m=max_params_m
        )
        
        # Add energy predictions and filter by energy if needed
        filtered_models = []
        for model in models:
            try:
                payload = {
                    "device_type": device_type,
                    "params_m": model["params_m"],
                    "gflops": model["gflops"],
                    "gmacs": model["gmacs"],
                    "size_mb": model["size_mb"],
                    "latency_avg_s": model["latency_avg_s"],
                    "throughput_iter_per_s": model["throughput_iter_per_s"]
                }
                
                predictions = predictor_service.predict([payload])
                if predictions and predictions[0].get("prediction_mwh"):
                    energy_mwh = predictions[0]["prediction_mwh"]
                    model["energy_prediction_mwh"] = energy_mwh
                    model["energy_ci_lower_mwh"] = predictions[0]["ci_lower_mwh"]
                    model["energy_ci_upper_mwh"] = predictions[0]["ci_upper_mwh"]
                    
                    # Filter by energy if max_energy_mwh is specified
                    if max_energy_mwh is None or energy_mwh <= max_energy_mwh:
                        filtered_models.append(model)
                else:
                    # If prediction fails, still include if no energy filter
                    if max_energy_mwh is None:
                        filtered_models.append(model)
            except Exception:
                # Skip models with prediction errors
                if max_energy_mwh is None:
                    filtered_models.append(model)
        
        return jsonify({
            "success": True,
            "recommendations": filtered_models,
            "count": len(filtered_models),
            "filters_applied": {
                "max_latency_s": max_latency_s,
                "min_accuracy_map50": min_accuracy,
                "max_params_m": max_params_m,
                "max_energy_mwh": max_energy_mwh,
                "device_type": device_type
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ============================================================================
# END YOLO API
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
            artifact = resolve_model_artifact(model_name + preferred_ext)
            if artifact:
                break
        
        # Fallback: tìm bất kỳ artifact nào
        if not artifact:
            artifact = resolve_model_artifact(model_name)
        
        if not artifact:
            return jsonify({
                "success": False,
                "error": (
                    f"Không tìm thấy file artifact cho model '{model_name}'. "
                    "Hãy thêm file .tflite hoặc .onnx vào thư mục model_store/ "
                    "hoặc convert model từ PyTorch (.pth) sang TFLite."
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

        # Tạo URL tải model - sử dụng IP thực tế của server này
        # Lấy IP của server từ request hoặc socket
        import socket
        try:
            # Thử lấy IP từ request headers (nếu có reverse proxy)
            pc_ip = request.host.split(':')[0]
            if pc_ip in ('localhost', '127.0.0.1', '0.0.0.0'):
                # Lấy IP thực tế của máy trong cùng subnet với BBB
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect((bbb_ip, 80))
                pc_ip = s.getsockname()[0]
                s.close()
        except:
            # Fallback: giả định cùng subnet, thay .34 -> .36 (hoặc IP mặc định)
            pc_ip = bbb_ip.rsplit('.', 1)[0] + '.36'
        
        model_url = f"http://{pc_ip}:5000/models/{artifact}"
        log_manager.add_log(
            log_type="info",
            message=f"Model URL: {model_url}",
            metadata={"pc_ip": pc_ip, "bbb_ip": bbb_ip, "artifact": artifact}
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
    """API: Lấy trạng thái của BBB"""
    bbb_ip = request.args.get("bbb_ip")
    if not bbb_ip:
        return jsonify({"success": False, "error": "bbb_ip is required"}), 400

    try:
        resp = requests.get(f"http://{bbb_ip}:8000/status", timeout=10)
        resp.raise_for_status()
        return jsonify({
            "success": True,
            "status": resp.json()
        })
    except requests.exceptions.ConnectionError:
        return jsonify({
            "success": False,
            "error": f"Unable to connect to device {bbb_ip}",
            "device_offline": True
        }), 200
    except requests.exceptions.Timeout:
        return jsonify({
            "success": False,
            "error": f"Device {bbb_ip} did not respond (timeout)",
            "device_timeout": True
        }), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Error: {str(e)}"
        }), 200

@app.route("/api/device/metrics", methods=["GET"])
def get_device_metrics():
    """API: Lấy system metrics từ thiết bị BBB (CPU, Memory, Storage, Temperature)"""
    bbb_ip = request.args.get("bbb_ip")
    if not bbb_ip:
        return jsonify({"success": False, "error": "bbb_ip is required"}), 400

    try:
        resp = requests.get(f"http://{bbb_ip}:8000/metrics", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        return jsonify(data)
        
    except requests.exceptions.ConnectionError:
        return jsonify({
            "success": False,
            "error": f"Unable to connect to device {bbb_ip}",
            "device_offline": True
        }), 200
    except requests.exceptions.Timeout:
        return jsonify({
            "success": False,
            "error": f"Device {bbb_ip} did not respond (timeout)",
            "device_timeout": True
        }), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Error: {str(e)}"
        }), 200


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
    token = os.getenv("BALENA_API_TOKEN")
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
    token = os.getenv("BALENA_API_TOKEN")
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
        device_uuid = data.get("device_uuid")
        device_endpoint = data.get("device_endpoint")
        fleet = data.get("fleet")
        
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
        for ext in [".tflite", ".onnx", ".pth", ".pt"]:
            potential_path = os.path.join(MODEL_STORE_DIR, f"{model_name}{ext}")
            if os.path.exists(potential_path):
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
        
        # Deploy to device via HTTP
        if not device_endpoint:
            # Need to get device IP from Balena API
            log_manager.add_log(
                "error",
                f"device_endpoint required for deployment (device UUID not sufficient)",
                {"device_uuid": device_uuid}
            )
            return jsonify({
                "success": False,
                "error": "device_endpoint required for deployment"
            }), 400
        
        # Construct model URL for device to download
        # Device will download from controller's /models/<filename> endpoint
        host = request.host  # e.g., "192.168.137.1:5000"
        model_filename = os.path.basename(model_path)
        model_url = f"http://{host}/models/{model_filename}"
        
        # Get energy budget if specified
        energy_budget = data.get("energy_budget_mwh")
        
        # Prepare deployment payload for device
        deploy_payload = {
            "model_name": model_name,
            "model_url": model_url,
            "model_info": model_info
        }
        
        if energy_budget is not None:
            deploy_payload["energy_budget_mwh"] = energy_budget
        
        # Call device's /deploy endpoint
        device_url = f"http://{device_endpoint}/deploy"
        print(f"[INFO] Deploying to device: {device_url}")
        
        try:
            response = requests.post(
                device_url,
                json=deploy_payload,
                timeout=60
            )
            response.raise_for_status()
            device_response = response.json()
            
            log_manager.add_log(
                "success",
                f"Deploy thành công model '{model_name}' lên thiết bị {device_display}",
                {
                    "model_name": model_name,
                    "device_uuid": device_uuid,
                    "device_endpoint": device_endpoint,
                    "model_path": model_path,
                    "fleet": fleet,
                    "device_response": device_response
                }
            )
            
            return jsonify({
                "success": True,
                "message": f"Model {model_name} deployed successfully",
                "device_uuid": device_uuid,
                "device_endpoint": device_endpoint,
                "model_path": model_path,
                "model_url": model_url,
                "device_response": device_response
            })
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Failed to deploy to device: {str(e)}"
            log_manager.add_log(
                "error",
                error_msg,
                {
                    "model_name": model_name,
                    "device_endpoint": device_endpoint,
                    "error": str(e)
                }
            )
            return jsonify({
                "success": False,
                "error": error_msg
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
        return jsonify({
            "success": True,
            "count": len(predictions),
            "predictions": predictions,
            "model_info": predictor_service.model_info,
        })
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        return jsonify({"success": False, "error": f"Lỗi khi dự đoán: {e}"}), 500


@app.route("/models/<path:filename>")
def download_model(filename):
    """Endpoint để BBB tải model files"""
    return send_from_directory(MODEL_STORE_DIR, filename, as_attachment=False)


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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
