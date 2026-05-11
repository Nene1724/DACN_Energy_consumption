import importlib.util
import os, sys, json, time

MODULE_PATH = os.path.join(os.path.dirname(__file__), '..', 'ml-controller', 'python', 'app.py')
MODULE_PATH = os.path.abspath(MODULE_PATH)

spec = importlib.util.spec_from_file_location('controller_app', MODULE_PATH)
controller = importlib.util.module_from_spec(spec)
# Ensure local package path is available for imports
controller_dir = os.path.dirname(MODULE_PATH)
if controller_dir not in sys.path:
    sys.path.insert(0, controller_dir)
spec.loader.exec_module(controller)

# Build a fake device_state to cache
device_key = 'fake-device-1'
fake_state = {
    'device_id': device_key,
    'connectivity_state': 'online',
    'deployment_state': 'running',
    'model_name': 'test-model',
    'inference_state': 'running',
    'energy': {
        'latest_inference_delta_mwh': 12.34,
        'session_energy_mwh': 56.78,
        'total_energy_wh': 123.456,
    },
    'last_update': 'now',
    'last_error': None,
}
controller.DEVICE_STATE_CACHE[device_key] = {'device_state': fake_state, 'timestamp': time.time()}
print('[SIM] Cached device state set')

# Monkeypatch _fetch_device_json to simulate failures
original_fetch = controller._fetch_device_json

def fake_fetch(bbb_ip, device_uuid, suffix, timeout=10):
    return None, 'simulated network failure'

controller._fetch_device_json = fake_fetch

# Use Flask test_request_context to call get_device_state
with controller.app.test_request_context(f"/api/device/state?device_uuid={device_key}"):
    resp = controller.get_device_state()
    print('[SIM] get_device_state response:')
    try:
        # resp may be a Flask Response; try to get json
        data = resp.get_json()
    except Exception:
        data = str(resp)
    print(json.dumps(data, indent=2))

# Restore
controller._fetch_device_json = original_fetch
