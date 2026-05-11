import os
import sys
import time

TEST_DIR = os.path.dirname(__file__)
APP_DIR = os.path.abspath(os.path.join(TEST_DIR, "..", "app"))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

import server


def test_delta_labeled_but_cumulative_rejected():
    # Reset energy metrics and meter
    with server.METER_LOCK:
        server.STATE["energy_metrics"] = server._create_energy_metrics()
        server.STATE["meter_metrics"] = server._create_meter_metrics()
        m = server.STATE["meter_metrics"]
        m["total_energy_wh"] = 50.0
        m["baseline_energy_wh"] = 50.0
        m["last_sample_ts"] = time.monotonic()
        server.STATE["meter_metrics"] = m

    client = server.app.test_client()

    # Post a payload that includes cumulative energy_wh but claims to be a delta
    resp = client.post("/telemetry", json={
        "energy_kind": "delta",
        "energy_wh": 150.0,  # cumulative large value
        "power_w": 1.0
    })
    assert resp.status_code == 400
    payload = resp.get_json()
    assert payload["error"] == "Invalid telemetry payload"
    assert "cannot be used with energy_kind='delta'" in payload["reason"]

    # Rejected payload must not append to energy history or alter meter semantics.
    history = (server.STATE.get("energy_metrics") or {}).get("history") or []
    assert len(history) == 0

    # Meter total_energy_wh should remain unchanged on rejection.
    assert server.STATE.get("meter_metrics", {}).get("total_energy_wh") == 50.0


def test_valid_delta_mwh_is_recorded():
    with server.METER_LOCK:
        server.STATE["energy_metrics"] = server._create_energy_metrics()
        server.STATE["meter_metrics"] = server._create_meter_metrics()
        meter = server.STATE["meter_metrics"]
        meter["total_energy_wh"] = 10.0
        meter["baseline_energy_wh"] = 10.0
        meter["last_sample_ts"] = time.monotonic()
        server.STATE["meter_metrics"] = meter

    client = server.app.test_client()
    resp = client.post("/telemetry", json={
        "energy_kind": "delta",
        "delta_mwh": 12.5,
        "power_mw": 2500.0,
        "timestamp": "2026-05-11T00:00:00Z",
    })

    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["energy_kind"] == "delta"
    history = (server.STATE.get("energy_metrics") or {}).get("history") or []
    assert len(history) == 1
    assert history[0]["energy_mwh"] == 12.5
    assert server.STATE.get("meter_metrics", {}).get("latest_inference_delta_mwh") is None


def test_cumulative_payload_updates_meter_only():
    with server.METER_LOCK:
        server.STATE["energy_metrics"] = server._create_energy_metrics()
        server.STATE["meter_metrics"] = server._create_meter_metrics()
        meter = server.STATE["meter_metrics"]
        meter["total_energy_wh"] = 8.0
        meter["baseline_energy_wh"] = 8.0
        meter["latest_inference_delta_mwh"] = 6.0
        meter["last_sample_ts"] = time.monotonic()
        server.STATE["meter_metrics"] = meter

    client = server.app.test_client()
    resp = client.post("/telemetry", json={
        "energy_kind": "cumulative",
        "energy_wh": 8.7,
        "power_w": 1.7,
    })

    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["energy_kind"] == "cumulative"
    history = (server.STATE.get("energy_metrics") or {}).get("history") or []
    assert len(history) == 0
    assert server.STATE.get("meter_metrics", {}).get("total_energy_wh") == 8.7
    assert server.STATE.get("meter_metrics", {}).get("latest_inference_delta_mwh") == 6.0


def test_cumulative_reset_does_not_inject_false_delta():
    with server.METER_LOCK:
        server.STATE["energy_metrics"] = server._create_energy_metrics()
        server.STATE["meter_metrics"] = server._create_meter_metrics()
        meter = server.STATE["meter_metrics"]
        meter["total_energy_wh"] = 20.0
        meter["baseline_energy_wh"] = 20.0
        meter["latest_inference_delta_mwh"] = 4.4
        meter["last_sample_ts"] = time.monotonic()
        server.STATE["meter_metrics"] = meter

    client = server.app.test_client()
    resp = client.post("/telemetry", json={
        "energy_kind": "cumulative",
        "energy_wh": 1.2,
        "power_w": 0.3,
        "timestamp": "2026-05-11T00:00:01Z",
    })

    assert resp.status_code == 200
    history = (server.STATE.get("energy_metrics") or {}).get("history") or []
    assert len(history) == 0
    assert server.STATE.get("meter_metrics", {}).get("latest_inference_delta_mwh") == 4.4
