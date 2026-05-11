import os
import sys

TEST_DIR = os.path.dirname(__file__)
APP_DIR = os.path.abspath(os.path.join(TEST_DIR, "..", "app"))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

import fnb58_telemetry_collector as collector


def test_collector_normalizes_cumulative_energy_wh():
    payload = collector._normalize_payload(
        {
            "energy_wh": 0.0182,
            "power_w": 2.9,
            "duration_s": 22.6,
            "timestamp": "2026-03-30T15:10:00Z",
        },
        "fnb58",
    )

    assert payload is not None
    assert payload["energy_kind"] == "cumulative"
    assert payload["energy_wh"] == 0.0182
    assert "delta_mwh" not in payload


def test_collector_normalizes_explicit_delta_mwh():
    payload = collector._normalize_payload(
        {
            "delta_mwh": 12.5,
            "power_mw": 2900.0,
            "timestamp": "2026-03-30T15:10:00Z",
        },
        "fnb58",
    )

    assert payload is not None
    assert payload["energy_kind"] == "delta"
    assert payload["delta_mwh"] == 12.5
    assert "energy_wh" not in payload
