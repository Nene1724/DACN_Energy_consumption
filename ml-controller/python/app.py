import os
import re
import json
from urllib.parse import urlparse
from flask import Flask, request, render_template, jsonify, send_from_directory, make_response
import requests
from requests.utils import requote_uri
from model_analyzer import ModelAnalyzer
from energy_predictor_service import EnergyPredictorService
from log_manager import LogManager

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
ENERGY_REPORTS_PATH = os.path.join(DATA_DIR, "energy_reports.json") #sosanh kq dudoan
PREFERRED_ARTIFACT_EXTS = [".pth", ".pt", ".onnx", ".tflite", ".bin"]
BALENA_API_BASE = os.getenv("BALENA_API_BASE", "https://api.balena-cloud.com")
BALENA_DEFAULT_TIMEOUT = int(os.getenv("BALENA_API_TIMEOUT", "30"))

# Initialize predictor + analyzer + log manager
predictor_service = EnergyPredictorService(ARTIFACTS_DIR)
analyzer = ModelAnalyzer(CSV_PATH, predictor_service=predictor_service, model_store_dir=MODEL_STORE_DIR, rpi5_csv_path=RPI5_CSV_PATH)
log_manager = LogManager(LOG_FILE_PATH, max_logs=50)


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
    token = token or os.getenv("BALENA_API_TOKEN")
    if not token:
        raise ValueError("BALENA_API_TOKEN chưa được cấu hình")

    limit = max(1, min(int(limit or 50), 200))

    # Build OData query parameters (don't use dict to avoid double encoding)
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


def get_device_public_url(device_uuid: str, token=None) -> str:
    """
    Fetch the Public Device URL for a device by its UUID.
    Returns: https://<uuid>.balena-devices.com or empty string if not enabled.
    """
    token = token or os.getenv("BALENA_API_TOKEN")
    if not token or not device_uuid:
        return ""

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
            return f"https://{device_uuid}.balena-devices.com"
    except Exception as e:
        print(f"[WARN] Could not fetch public URL for device {device_uuid}: {e}")
    
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
        
        # Try to download using timm
        try:
            import timm
            import torch
            
            # Create model_store directory if not exists
            os.makedirs(MODEL_STORE_DIR, exist_ok=True)
            
            # Download model from timm
            print(f"Downloading model: {model_name}")
            model = timm.create_model(model_name, pretrained=True)
            
            # Save as .pth file
            output_filename = f"{model_name}.pth"
            output_path = os.path.join(MODEL_STORE_DIR, output_filename)
            
            torch.save(model.state_dict(), output_path)
            
            # Verify file was created
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
                return jsonify({
                    "success": True,
                    "message": f"Successfully downloaded {model_name} ({file_size:.2f} MB)",
                    "artifact": output_filename
                })
            else:
                return jsonify({
                    "success": False,
                    "error": "Model download failed - file not created"
                }), 500
                
        except ImportError:
            return jsonify({
                "success": False,
                "error": "timm library not installed. Install with: pip install timm"
            }), 500
        except Exception as download_error:
            return jsonify({
                "success": False,
                "error": f"Failed to download model: {str(download_error)}"
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
            return jsonify({
                "success": True,
                "status": resp.json()
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

@app.route("/api/device/measure-energy", methods=["POST"]) #ss kq
def device_measure_energy():
    """Proxy a measurement request to an agent and ensure results are reported back.

    JSON body:
    {
      "device_url": "http://<agent>:<port>",
      "duration_s": 10.0
    }
    """
    try:
        data = request.get_json(force=True, silent=False) or {}
        device_url = data.get("device_url")
        duration_s = float(data.get("duration_s", 10.0))
        if not device_url:
            return jsonify({"success": False, "error": "device_url is required"}), 400

        controller_url = request.host_url.rstrip('/')
        payload = {"duration_s": duration_s, "controller_url": controller_url}
        resp = requests.post(f"{device_url.rstrip('/')}/measure_energy", json=payload, timeout=duration_s + 10)
        resp.raise_for_status()
        result = resp.json()
        return jsonify({"success": True, "agent_result": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

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
        
        # Construct model URL for device to download
        # Priority: 1) CONTROLLER_PUBLIC_URL env, 2) ngrok URL, 3) local IP
        import socket
        
        public_base_url = os.getenv("CONTROLLER_PUBLIC_URL", "").strip()
        if public_base_url:
            # Use configured public URL (ngrok/cloudflare tunnel)
            model_url = f"{public_base_url.rstrip('/')}/models/{os.path.basename(model_path)}"
            print(f"[INFO] Using public controller URL: {public_base_url}")
        else:
            # Fallback to local IP (may fail for cross-network deployment)
            controller_ip = socket.gethostbyname(socket.gethostname())
            controller_port = request.environ.get('SERVER_PORT', '5000')
            model_filename = os.path.basename(model_path)
            model_url = f"http://{controller_ip}:{controller_port}/models/{model_filename}"
            print(f"[WARN] Using local IP for model URL. Set CONTROLLER_PUBLIC_URL env for cross-network deployment")
        
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
                
                return jsonify({
                    "success": True,
                    "message": f"Model {model_name} deployed successfully",
                    "device_uuid": device_uuid,
                    "device_endpoint": device_endpoint,
                    "model_path": model_path,
                    "model_url": model_url,
                    "device_response": device_response,
                    "used_url_type": url_type
                })
                
            except requests.exceptions.RequestException as e:
                last_error = str(e)
                print(f"[WARN] Failed to deploy via {url_type} URL ({device_url}): {e}")
                # Try next URL
                continue
        
        # All URLs failed
        error_msg = f"Failed to deploy to device (tried {len(urls_to_try)} URLs). Last error: {last_error}"
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
                device_key = "jetson_nano" if "jetson" in device_type.lower() else "raspberry_pi5"
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
    """Endpoint để BBB tải model files"""
    return send_from_directory(MODEL_STORE_DIR, filename, as_attachment=False)


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
                        "metrics": {"test_mape": 21.5, "test_r2": 0.96},
                        "model_name": "Gradient Boosting"
                    },
                    "rpi5_model": {
                        "metrics": {"loo_mape": 14.2, "loo_r2": 0.95},
                        "model_name": "Gradient Boosting"
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


# -------- Energy measurement reporting & comparison --------
def _load_json_safe(path: str, default):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return default


def _save_json_safe(path: str, data) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[WARN] Failed to write {path}: {e}")


@app.route("/api/energy/report", methods=["POST"])
def api_energy_report():
    """Receive actual energy measurement from agents and compute comparison.

    Expected JSON body:
    {
      "device_type": "jetson_nano" | "raspberry_pi5" | ...,
      "device_id": "<optional>",
      "model_name": "edgenext_xx_small",
      "actual_energy_mwh": 17.5,
      "avg_power_mw": 950.2,
      "duration_s": 62.3,
      "timestamp": "ISO8601" (optional),
      "sensor_type": "powercap|tegrastats|ina219|unknown" (optional)
    }
    If features are not provided, attempt lookup by model_name+device from benchmark CSVs.
    """
    from datetime import datetime
    try:
        payload = request.get_json(force=True, silent=False) or {}
        device_type = (payload.get("device_type") or payload.get("device") or "unknown").strip()
        model_name = (payload.get("model_name") or payload.get("model") or "").strip()
        actual_energy_mwh = float(payload.get("actual_energy_mwh"))
        duration_s = float(payload.get("duration_s", 0))
        avg_power_mw = payload.get("avg_power_mw")
        avg_power_mw = float(avg_power_mw) if avg_power_mw is not None else None
        sensor_type = payload.get("sensor_type") or "unknown"
        ts = payload.get("timestamp") or datetime.utcnow().isoformat()

        # Build predictor input from CSVs if possible
        feat = {
            "device_type": device_type,
            "model": model_name,
        }

        # Try find row in analyzer.df
        row = None
        if model_name:
            try:
                def _norm(s: str) -> str:
                    return "".join(ch.lower() for ch in s if ch.isalnum())
                mnorm = _norm(model_name)
                df = analyzer.df
                # Filter by device first
                if any(k in device_type.lower() for k in ["jetson", "nano"]):
                    df = df[df["device"] == "jetson_nano_2gb"]
                elif any(k in device_type.lower() for k in ["rasp", "rpi", "pi5"]):
                    df = df[df["device"] == "raspberry_pi5"]
                for _, r in df.iterrows():
                    if _norm(str(r.get("model_name", ""))) == mnorm:
                        row = r
                        break
            except Exception:
                row = None

        if row is not None:
            feat.update({
                "params_m": float(row.get("params_m", 0) or 0),
                "gflops": float(row.get("gflops", 0) or 0),
                "gmacs": float(row.get("gmacs", 0) or 0),
                "size_mb": float(row.get("size_mb", 0) or 0),
                "latency_avg_s": float(row.get("latency_avg_s", 0) or 0),
                "throughput_iter_per_s": float(row.get("throughput_iter_per_s", 0) or 0),
            })
        else:
            for k in ["params_m","gflops","gmacs","size_mb","latency_avg_s","throughput_iter_per_s"]:
                if k in payload:
                    feat[k] = float(payload.get(k))

        predicted_mwh = None
        ci_lower = None
        ci_upper = None
        model_used = None
        mape_pct = None
        if all(k in feat for k in ["params_m","gflops","gmacs","size_mb","latency_avg_s","throughput_iter_per_s"]):
            pred = predictor_service.predict([feat])[0]
            if not pred.get("error"):
                predicted_mwh = pred.get("prediction_mwh")
                ci_lower = pred.get("ci_lower_mwh")
                ci_upper = pred.get("ci_upper_mwh")
                model_used = pred.get("model_used")
                mape_pct = pred.get("mape_pct")

        abs_err = None
        pct_err = None
        if (predicted_mwh is not None) and (actual_energy_mwh is not None):
            abs_err = abs(actual_energy_mwh - predicted_mwh)
            if actual_energy_mwh > 1e-9:
                pct_err = abs_err / actual_energy_mwh * 100.0

        reports = _load_json_safe(ENERGY_REPORTS_PATH, default=[])
        entry = {
            "timestamp": ts,
            "device_type": device_type,
            "device_id": payload.get("device_id"),
            "model_name": model_name,
            "sensor_type": sensor_type,
            "duration_s": duration_s,
            "avg_power_mw": avg_power_mw,
            "actual_energy_mwh": actual_energy_mwh,
            "predicted_mwh": predicted_mwh,
            "ci_lower_mwh": ci_lower,
            "ci_upper_mwh": ci_upper,
            "abs_error_mwh": abs_err,
            "pct_error": pct_err,
        }
        reports.append(entry)
        _save_json_safe(ENERGY_REPORTS_PATH, reports)

        return jsonify({
            "success": True,
            "comparison": entry,
            "count": len(reports)
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/energy/recent", methods=["GET"])
def api_energy_recent():
    try:
        n = int(request.args.get("n", 20))
        reports = _load_json_safe(ENERGY_REPORTS_PATH, default=[])
        return jsonify({
            "success": True,
            "items": reports[-n:][::-1],
            "total": len(reports)
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

# ss kq dd

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
    
    # Validate required fields
    required_fields = ["device_type", "params_m", "gflops", "gmacs", "size_mb", 
                      "latency_avg_s", "throughput_iter_per_s"]
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        return jsonify({
            "success": False,
            "error": f"Missing required fields: {', '.join(missing_fields)}"
        }), 400
    
    try:
        # Prepare payload for energy predictor
        payload = {
            "device_type": data["device_type"],
            "params_m": float(data["params_m"]),
            "gflops": float(data["gflops"]),
            "gmacs": float(data["gmacs"]),
            "size_mb": float(data["size_mb"]),
            "latency_avg_s": float(data["latency_avg_s"]),
            "throughput_iter_per_s": float(data["throughput_iter_per_s"])
        }
        
        # Predict energy
        predictions = predictor_service.predict([payload])
        
        if not predictions or predictions[0].get("prediction_mwh") is None:
            error_msg = predictions[0].get("error", "Unknown error") if predictions else "No predictions returned"
            print(f"[ERROR] Energy prediction failed: {error_msg}")
            print(f"[DEBUG] Payload: {payload}")
            return jsonify({
                "success": False,
                "error": f"Energy prediction failed: {error_msg}"
            }), 500
        
        pred = predictions[0]
        
        # Load thresholds for categorization
        thresholds_path = os.path.join(ARTIFACTS_DIR, "energy_thresholds.json")
        thresholds_data = {}
        if os.path.exists(thresholds_path):
            with open(thresholds_path, 'r', encoding='utf-8') as f:
                thresholds_data = json.load(f)
        
        # Determine device key
        device_type = data["device_type"].lower()
        device_key = "jetson_nano" if "jetson" in device_type else "raspberry_pi5"
        thresholds = thresholds_data.get(device_key, {})
        
        p25 = thresholds.get("p25", 50)
        p50 = thresholds.get("p50", 85)
        p75 = thresholds.get("p75", 150)
        
        # Categorize energy
        energy = pred["prediction_mwh"]
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
        
        # Check if model is downloaded
        model_name = data.get("model_name", "unknown")
        model_downloaded = False
        if model_name and model_name != "unknown":
            artifact = resolve_model_artifact(model_name)
            model_downloaded = artifact is not None
        
        result = {
            "model_name": model_name,
            "device_type": data["device_type"],
            "predicted_energy_mwh": round(energy, 2),
            "ci_lower_mwh": round(pred.get("ci_lower_mwh", 0), 2),
            "ci_upper_mwh": round(pred.get("ci_upper_mwh", 0), 2),
            "energy_category": energy_category,
            "recommendation": recommendation,
            "reason": reason,
            "thresholds": {
                "p25": p25,
                "p50": p50,
                "p75": p75
            },
            "model_info": {
                "params_m": data["params_m"],
                "gflops": data["gflops"],
                "size_mb": data["size_mb"],
                "latency_avg_s": data["latency_avg_s"]
            },
            "model_downloaded": model_downloaded,
            "model_used_for_prediction": pred.get("model_used"),
            "prediction_mape_pct": pred.get("mape_pct")
        }
        
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
    try:
        device_url = None
        
        # Check if device_endpoint looks like a UUID (32+ hex chars)
        if len(device_endpoint) >= 32 and all(c in "0123456789abcdef" for c in device_endpoint.lower()):
            # It's a UUID, try to get public URL
            public_url = get_device_public_url(device_endpoint)
            if public_url:
                device_url = f"{public_url}/status"
            else:
                return jsonify({
                    "success": False,
                    "error": "Public URL not available for this device"
                }), 503
        else:
            # Treat as IP address
            device_url = f"http://{device_endpoint}:8000/status"
        
        response = requests.get(device_url, timeout=5)
        
        if response.status_code == 200:
            return jsonify({
                "success": True,
                "data": response.json()
            })
        else:
            return jsonify({
                "success": False,
                "error": f"HTTP {response.status_code}"
            }), response.status_code
            
    except requests.exceptions.Timeout:
        return jsonify({
            "success": False,
            "error": "Request timeout"
        }), 504
    except requests.exceptions.ConnectionError:
        return jsonify({
            "success": False,
            "error": "Connection failed"
        }), 503
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
