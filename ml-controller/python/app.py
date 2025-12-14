import os
import re
from flask import Flask, request, render_template, jsonify, send_from_directory
import requests
from model_analyzer import ModelAnalyzer
from energy_predictor_service import EnergyPredictorService
from log_manager import LogManager

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates'))

# Balena API Configuration (hardcoded for development)
# TODO: Move to environment variables in production
if not os.getenv("BALENA_API_TOKEN"):
    # Try Session Token first (JWT format)
    session_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6MzY0MTIyLCJleHAiOjE3NjU4MDkyOTksImp3dF9zZWNyZXQiOiJXRVMzRE1DNjMyRVhXWE1YN0RSU1JJUVBEM0pLT1YyTiIsImF1dGhUaW1lIjoxNzY1Njc5Njk5Mjg4LCJpYXQiOjE3NjU2ODA4MDd9.O5uXxj2byOiIFaVZd94C3sdgi9Oh1LD-2AUGexKHSZg"
    # Fallback to API Key
    api_key = "KTfVKRmgeSxIgFxFq9aAdstuOAHSWAFu"
    
    # Use session token (usually more permissive)
    os.environ["BALENA_API_TOKEN"] = session_token

BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # ml-controller root
DATA_DIR = os.path.join(BASE_DIR, "data")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
MODEL_STORE_DIR = os.path.join(BASE_DIR, "model_store")
CSV_PATH = os.path.join(DATA_DIR, "126_benchmark_final.csv")
LOG_FILE_PATH = os.path.join(DATA_DIR, "deployment_logs.json")
PREFERRED_ARTIFACT_EXTS = [".pth", ".pt", ".onnx", ".tflite", ".bin"]
BALENA_API_BASE = os.getenv("BALENA_API_BASE", "https://api.balena-cloud.com")
BALENA_DEFAULT_TIMEOUT = int(os.getenv("BALENA_API_TIMEOUT", "30"))

# Initialize predictor + analyzer + log manager
predictor_service = EnergyPredictorService(ARTIFACTS_DIR)
analyzer = ModelAnalyzer(CSV_PATH, predictor_service=predictor_service, model_store_dir=MODEL_STORE_DIR)
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
        filters.append(
            f"belongs_to__application/any(a: a/slug eq '{_sanitize_filter_value(app_slug)}')"
        )
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


@app.route("/")
def index():
    return render_template("index.html")


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


@app.route("/api/deploy", methods=["POST"])
def deploy():
    """API: Deploy model xuống BBB"""
    try:
        data = request.get_json()
        bbb_ip = data.get("bbb_ip")
        model_name = data.get("model_name")
        max_energy = data.get("max_energy")
        force = bool(data.get("force", False))

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
        artifact = resolve_model_artifact(model_name)
        if not artifact:
            return jsonify({
                "success": False,
                "error": (
                    f"Không tìm thấy file artifact cho model '{model_name}'. "
                    "Hãy thêm file vào thư mục model_store/ hoặc đặt tên phù hợp."
                )
            }), 404

        # Tạo URL tải model (giả sử model file nằm trong folder models/)
        # Trong thực tế, bạn sẽ cần download model từ model hub
        pc_ip = bbb_ip.rsplit('.', 1)[0] + '.1'
        model_url = f"http://{pc_ip}:5000/models/{artifact}"

        # Gửi lệnh deploy xuống BBB
        deploy_url = f"http://{bbb_ip}:8000/deploy"
        payload = {
            "model_name": model_name,
            "model_url": model_url,
            "model_info": model_info
        }
        if energy_budget is not None:
            payload["energy_budget_mwh"] = energy_budget

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
    try:
        bbb_ip = request.args.get("bbb_ip")
        if not bbb_ip:
            return jsonify({"success": False, "error": "bbb_ip is required"}), 400

        resp = requests.get(f"http://{bbb_ip}:8000/status", timeout=10)
        resp.raise_for_status()
        
        return jsonify({
            "success": True,
            "status": resp.json()
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

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

@app.route("/api/balena/fleets", methods=["GET"])
def get_balena_fleets():
    """API: Liệt kê các fleet/application trên Balena Cloud"""
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
        url = f"{BALENA_API_BASE.rstrip('/')}/v6/application?$select=id,app_name,slug&$orderby=app_name asc"
        resp = requests.get(url, headers=headers, timeout=BALENA_DEFAULT_TIMEOUT)
        resp.raise_for_status()
        payload = resp.json()
        apps = payload.get("d") or payload.get("value") or []
        fleets = [{"id": app.get("id"), "name": app.get("app_name"), "slug": app.get("slug")} for app in apps]
        return jsonify({"success": True, "fleets": fleets})
    except requests.HTTPError as e:
        status = e.response.status_code if e.response else 502
        try:
            msg = e.response.json()
        except Exception:
            msg = e.response.text if e.response else str(e)
        return jsonify({"success": False, "error": f"Balena API error ({status}): {msg}"}), status
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
        print(f"[ERROR] Balena API HTTP Error {status}: {msg}")
        
        return jsonify({
            "success": False,
            "error": f"Lỗi Balena API ({status}): {msg}",
            "details": str(msg)
        }), status
    except requests.RequestException as e:
        print(f"[ERROR] Balena Connection Error: {str(e)}")
        return jsonify({
            "success": False,
            "error": f"Lỗi kết nối Balena: {str(e)}"
        }), 502
    except Exception as e:
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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
