import os
import sys
import time

TEST_DIR = os.path.dirname(__file__)
APP_DIR = os.path.abspath(os.path.join(TEST_DIR, ".."))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

import app as controller


def test_cached_state_returned_on_fetch_failure():
    device_key = 'unit-test-device-123'
    fake_state = {
        'device_id': device_key,
        'connectivity_state': 'online',
        'deployment_state': 'running',
        'model_name': 'test-model',
        'inference_state': 'running',
        'energy': {
            'latest_inference_delta_mwh': 7.77,
            'session_energy_mwh': 33.3,
            'total_energy_wh': 11.11,
        },
        'last_update': 'now',
        'last_error': None,
    }
    controller.DEVICE_STATE_CACHE[device_key] = {'device_state': fake_state, 'timestamp': time.time()}

    # Monkeypatch fetch to simulate failure
    orig = controller._fetch_device_json
    def fail_fetch(bbb_ip, device_uuid, suffix, timeout=10):
        return None, 'simulated_failure'
    controller._fetch_device_json = fail_fetch

    # Call endpoint within Flask test_request_context
    with controller.app.test_request_context(f"/api/device/state?device_uuid={device_key}"):
        resp = controller.get_device_state()
        # resp is a Flask Response or tuple
        try:
            data = resp.get_json()
        except Exception:
            # If flask returns tuple
            data = resp[0].get_json() if isinstance(resp, tuple) else None

    # Restore
    controller._fetch_device_json = orig

    assert data is not None
    assert data.get('success') is True
    assert data.get('cached') is True
    device_state = data.get('device_state')
    assert device_state.get('connectivity_state') == 'stale'
    assert device_state.get('energy', {}).get('latest_inference_delta_mwh') == 7.77


def test_partial_fetch_failure_preserves_live_state():
    device_key = 'unit-test-device-partial'
    controller.DEVICE_STATE_CACHE[device_key] = {
        'device_state': {
            'device_id': device_key,
            'connectivity_state': 'online',
            'deployment_state': 'running',
            'model_name': 'persisted-model',
            'inference_state': 'running',
            'inference_active': True,
            'energy': {
                'latest_inference_delta_mwh': 9.99,
                'session_energy_mwh': 42.0,
                'total_energy_wh': 7.5,
                'latest_timed_measurement_mwh': 8.8,
            },
            'last_update': 'cached',
            'last_error': None,
        },
        'timestamp': time.time(),
    }

    original_fetch = controller._fetch_device_json

    def mixed_fetch(bbb_ip, device_uuid, suffix, timeout=10):
        if suffix == '/status':
            return {
                'status': 'running',
                'model_name': 'persisted-model',
                'inference_active': True,
                'energy_semantics': {
                    'latest_inference_delta_mwh': 9.99,
                    'session_energy_mwh': 42.0,
                    'total_energy_wh': 7.5,
                },
                'meter_metrics': {'connected': True},
            }, None
        if suffix == '/metrics':
            return None, 'simulated_metrics_failure'
        return None, 'unexpected'

    controller._fetch_device_json = mixed_fetch
    try:
        with controller.app.test_request_context(f"/api/device/state?device_uuid={device_key}"):
            resp = controller.get_device_state()
            data = resp.get_json()
    finally:
        controller._fetch_device_json = original_fetch

    assert data is not None
    state = data['device_state']
    assert state['connectivity_state'] == 'online'
    assert state['deployment_state'] == 'running'
    assert state['model_name'] == 'persisted-model'
    assert state['energy']['latest_inference_delta_mwh'] == 9.99
    assert state['energy']['session_energy_mwh'] == 42.0
