import os
import sys
import time
import threading

import pytest


TEST_DIR = os.path.dirname(__file__)
APP_DIR = os.path.abspath(os.path.join(TEST_DIR, "..", "app"))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

import server


def test_concurrent_inferences_do_not_overlap():
    # Reset state
    with server.METER_LOCK:
        server.STATE["meter_metrics"] = server._create_meter_metrics()
        server.STATE["inference_measurements"] = {}
        m = server.STATE["meter_metrics"]
        m["total_energy_wh"] = 1.0
        m["last_sample_ts"] = time.monotonic()
        server.STATE["meter_metrics"] = m

    id1 = server.start_inference_measurement("a1")
    # advance meter a little
    with server.METER_LOCK:
        server.STATE["meter_metrics"]["total_energy_wh"] = 1.01
        server.STATE["meter_metrics"]["last_sample_ts"] = time.monotonic()

    id2 = server.start_inference_measurement("b2")
    # advance meter further
    with server.METER_LOCK:
        server.STATE["meter_metrics"]["total_energy_wh"] = 1.06
        server.STATE["meter_metrics"]["last_sample_ts"] = time.monotonic()

    r2 = server.end_inference_measurement(id2, wait_for_sample=False)
    r1 = server.end_inference_measurement(id1, wait_for_sample=False)

    assert r2["status"] == "ok"
    assert r1["status"] == "ok"
    # Ensure deltas are non-negative and independent
    assert r2["delta_mwh"] >= 0
    assert r1["delta_mwh"] >= 0


def test_reconnect_resets_measurements():
    with server.METER_LOCK:
        server.STATE["meter_metrics"] = server._create_meter_metrics()
        server.STATE["inference_measurements"] = {}
        m = server.STATE["meter_metrics"]
        m["total_energy_wh"] = 120.0
        m["last_sample_ts"] = time.monotonic()
        server.STATE["meter_metrics"] = m

    iid = server.start_inference_measurement("recon1")
    # Simulate device reconnect: new total drops
    with server.METER_LOCK:
        server.STATE["meter_metrics"]["total_energy_wh"] = 2.0
        server.STATE["meter_metrics"]["device_session_id"] = 1
        server.STATE["meter_metrics"]["last_sample_ts"] = time.monotonic()
        # Invalidate measurements as _update_meter_snapshot would
        server.STATE["inference_measurements"] = {}

    res = server.end_inference_measurement(iid, wait_for_sample=False)
    assert res["status"] in ("device_session_changed", "missing_start")


def test_fast_inference_low_confidence():
    with server.METER_LOCK:
        server.STATE["meter_metrics"] = server._create_meter_metrics()
        m = server.STATE["meter_metrics"]
        m["total_energy_wh"] = 5.0
        m["last_sample_ts"] = time.monotonic()
        server.STATE["meter_metrics"] = m
        server.STATE["inference_measurements"] = {}

    iid = server.start_inference_measurement("fast1")
    # Do not advance last_sample_ts to simulate fast inference
    res = server.end_inference_measurement(iid, wait_for_sample=True)
    assert "low_confidence" in (res.get("note") or "")


def test_long_runtime_stability():
    with server.METER_LOCK:
        server.STATE["meter_metrics"] = server._create_meter_metrics()
        m = server.STATE["meter_metrics"]
        m["total_energy_wh"] = 0.0
        m["last_sample_ts"] = time.monotonic()
        server.STATE["meter_metrics"] = m
        server.STATE["inference_measurements"] = {}

    for i in range(50):
        iid = server.start_inference_measurement(f"loop{i}")
        with server.METER_LOCK:
            server.STATE["meter_metrics"]["total_energy_wh"] += 0.01
            server.STATE["meter_metrics"]["last_sample_ts"] = time.monotonic()
        res = server.end_inference_measurement(iid, wait_for_sample=False)
        assert res["status"] == "ok"

    # Ensure no entries remain
    assert len(server.STATE.get("inference_measurements") or {}) == 0


def test_repeated_inference_latest_delta_stays_bounded():
    with server.METER_LOCK:
        server.STATE["meter_metrics"] = server._create_meter_metrics()
        meter = server.STATE["meter_metrics"]
        meter["total_energy_wh"] = 20.0
        meter["baseline_energy_wh"] = 20.0
        meter["last_sample_ts"] = time.monotonic()
        server.STATE["meter_metrics"] = meter
        server.STATE["inference_measurements"] = {}
        server.STATE["energy_metrics"] = server._create_energy_metrics()

    deltas = []
    for idx in range(8):
        iid = server.start_inference_measurement(f"bounded-{idx}")
        with server.METER_LOCK:
            server.STATE["meter_metrics"]["total_energy_wh"] = 20.0 + ((idx + 1) * 0.01)
            server.STATE["meter_metrics"]["last_sample_ts"] = time.monotonic()
        result = server.end_inference_measurement(iid, wait_for_sample=False)
        deltas.append(result["delta_mwh"])

    latest = (server.STATE.get("meter_metrics") or {}).get("latest_inference_delta_mwh")
    assert latest == deltas[-1]
    assert max(deltas) <= 100.0
    assert all(delta >= 0 for delta in deltas)


def test_semantic_separation_session_vs_inference_delta():
    with server.METER_LOCK:
        server.STATE["meter_metrics"] = server._create_meter_metrics()
        server.STATE["inference_measurements"] = {}
        meter = server.STATE["meter_metrics"]
        meter["total_energy_wh"] = 10.0
        meter["baseline_energy_wh"] = 10.0
        meter["last_sample_ts"] = time.monotonic()
        server.STATE["meter_metrics"] = meter

    iid = server.start_inference_measurement("sem1")
    with server.METER_LOCK:
        server.STATE["meter_metrics"]["total_energy_wh"] = 10.02
        server.STATE["meter_metrics"]["last_sample_ts"] = time.monotonic()
    result = server.end_inference_measurement(iid, wait_for_sample=False)

    # Session energy should be cumulative and larger/equal over time.
    server._update_meter_snapshot({"last_values": {"energy_wh": 10.05, "power_w": 5.0}}, status="connected", connected=True)
    session_energy = server.STATE["meter_metrics"].get("session_energy_mwh")

    assert result["status"] == "ok"
    assert result["delta_mwh"] <= 30.0
    assert session_energy is not None and session_energy >= result["delta_mwh"]


def test_telemetry_requires_energy_kind_and_cumulative_not_recorded():
    client = server.app.test_client()

    with server.METER_LOCK:
        server.STATE["energy_metrics"] = server._create_energy_metrics()

    # Missing energy_kind should be rejected.
    resp = client.post("/telemetry", json={"energy_mwh": 5.0})
    assert resp.status_code == 400

    # Cumulative telemetry should not be pushed into energy history.
    resp2 = client.post("/telemetry", json={
        "energy_kind": "cumulative",
        "energy_wh": 1.2,
        "power_w": 2.0,
    })
    assert resp2.status_code == 200
    payload = resp2.get_json()
    assert payload["energy_kind"] == "cumulative"
    history = (server.STATE.get("energy_metrics") or {}).get("history") or []
    assert len(history) == 0


def test_measure_energy_runs_exactly_one_inference(monkeypatch):
    calls = {"run": 0}

    monkeypatch.setattr(server, "_find_powercap_energy_file", lambda: "fake_energy_uj")
    monkeypatch.setattr(server, "_read_uint", lambda path: 1000 if calls["run"] == 0 else 1200)
    monkeypatch.setattr(server, "_run_single_inference_locked", lambda interpreter, input_size: calls.__setitem__("run", calls["run"] + 1))

    with server.METER_LOCK:
        server.STATE["status"] = "running"
        server.STATE["meter_metrics"] = server._create_meter_metrics()
        server.STATE["meter_metrics"]["connected"] = True
    with server.MODEL_LOCK:
        server.LOADED_INTERPRETER = object()
        server.LOADED_INPUT_SIZE = (3, 224, 224)

    report = server.measure_energy_during_inference(duration_s=999.0)
    assert report["success"] is True
    assert report["iterations"] == 1
    assert calls["run"] == 1


def test_active_measurement_window_tracks_polling_requests(monkeypatch):
    with server.MEASUREMENT_TRACE_LOCK:
        server.ACTIVE_MEASUREMENT_WINDOW["current"] = {
            "measurement_id": "m1",
            "window_start_ts": server._now_iso(),
            "window_start_monotonic": time.monotonic(),
            "trigger_source": "test",
            "owner_thread_id": -1,
            "requests_during_window": [],
            "inference_count": 1,
            "non_inference_activity_detected": False,
        }

    client = server.app.test_client()
    try:
        resp = client.get("/status")
        assert resp.status_code == 200
        window = server.ACTIVE_MEASUREMENT_WINDOW["current"]
        assert window is not None
        assert len(window["requests_during_window"]) >= 1
        assert window["requests_during_window"][0]["path"] == "/status"
        assert window["requests_during_window"][0]["classification"] == "polling"
        assert window["non_inference_activity_detected"] is True
    finally:
        with server.MEASUREMENT_TRACE_LOCK:
            server.ACTIVE_MEASUREMENT_WINDOW["current"] = None


def test_status_snapshot_is_deepcopy_and_consistent():
    stop = {"value": False}

    def writer():
        while not stop["value"]:
            with server.METER_LOCK:
                m = server.STATE.get("meter_metrics") or server._create_meter_metrics()
                m["total_energy_wh"] = (m.get("total_energy_wh") or 0.0) + 0.001
                m["last_values"] = {"energy_wh": m["total_energy_wh"]}
                server.STATE["meter_metrics"] = m

    t = threading.Thread(target=writer, daemon=True)
    t.start()

    # Snapshot several times while writer mutates state.
    for _ in range(30):
        snap = server._build_state_snapshot()
        assert isinstance(snap, dict)
        assert "meter_metrics" in snap
        # Mutate snapshot and ensure global state is untouched.
        snap["meter_metrics"]["status"] = "mutated_in_snapshot"
        assert server.STATE["meter_metrics"].get("status") != "mutated_in_snapshot"

    stop["value"] = True
    t.join(timeout=1.0)
