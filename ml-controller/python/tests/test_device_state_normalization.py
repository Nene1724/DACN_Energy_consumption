import os
import sys

TEST_DIR = os.path.dirname(__file__)
PY_DIR = os.path.abspath(os.path.join(TEST_DIR, ".."))
if PY_DIR not in sys.path:
    sys.path.insert(0, PY_DIR)

import app as controller_app


def test_device_state_offline_fallback_has_complete_shape(monkeypatch):
    def fake_fetch_device_json(bbb_ip, device_uuid, suffix, timeout=10):
        return None, "Failed to fetch"

    monkeypatch.setattr(controller_app, "_fetch_device_json", fake_fetch_device_json)

    client = controller_app.app.test_client()
    resp = client.get("/api/device/state", query_string={"bbb_ip": "10.0.0.2"})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["success"] is True
    state = data["device_state"]

    assert state["connectivity_state"] == "offline"
    assert state["model_name"] is None
    assert state["inference_state"] == "unknown"
    assert state["last_error"]
    assert "energy" in state
    assert "latest_inference_delta_mwh" in state["energy"]


def test_device_state_online_normalized_shape(monkeypatch):
    def fake_fetch_device_json(bbb_ip, device_uuid, suffix, timeout=10):
        if suffix == "/status":
            return {
                "status": "running",
                "model_name": "yolo11n",
                "inference_active": True,
                "energy_semantics": {
                    "latest_inference_delta_mwh": 11.2,
                    "session_energy_mwh": 130.0,
                    "total_energy_wh": 2.1,
                    "latest_timed_measurement_mwh": 10.9,
                },
                "meter_metrics": {"connected": True},
            }, None
        if suffix == "/metrics":
            return {
                "success": True,
                "cpu": {"percent": 42.0},
                "memory": {"used_percent": 31.0},
                "storage": {"used_percent": 40.0},
                "temperature_c": 58.0,
            }, None
        return None, None

    monkeypatch.setattr(controller_app, "_fetch_device_json", fake_fetch_device_json)

    client = controller_app.app.test_client()
    resp = client.get("/api/device/state", query_string={"bbb_ip": "10.0.0.2", "device_uuid": "abc123"})
    assert resp.status_code == 200
    data = resp.get_json()
    state = data["device_state"]

    assert state["connectivity_state"] == "online"
    assert state["deployment_state"] == "running"
    assert state["inference_state"] == "running"
    assert state["model_name"] == "yolo11n"
    assert state["energy"]["latest_inference_delta_mwh"] == 11.2
