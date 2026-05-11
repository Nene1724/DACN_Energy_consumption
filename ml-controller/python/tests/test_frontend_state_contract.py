import os


def test_frontend_offline_fallback_not_undefined():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    path = os.path.join(root, "templates", "index.html")
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()

    # Offline fallback object must include explicit semantic placeholders.
    assert "model_name: null" in src
    assert "inference_state: 'unknown'" in src
    assert "connectivity_state: 'offline'" in src


def test_frontend_uses_normalized_device_state_endpoint():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    path = os.path.join(root, "templates", "index.html")
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()

    # Centralized device state endpoint should be the shared source.
    assert "/api/device/state?" in src
