import importlib.util
import sys
import os
import time
import json

MODULE_PATH = os.path.join(os.path.dirname(__file__), '..', 'jetson-ml-agent', 'app', 'server.py')
MODULE_PATH = os.path.abspath(MODULE_PATH)

spec = importlib.util.spec_from_file_location('agent_server', MODULE_PATH)
agent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent)

print('Loaded agent module:', agent)

# Helper to pretty print state
def pp(obj):
    print(json.dumps(obj, indent=2, default=str))

# Ensure meter baseline
agent._update_meter_snapshot({'last_values': {'energy_wh': 100.0, 'power_w': 0.5, 'voltage_v': 5.0, 'current_a': 0.1}}, status='connected', connected=True)
print('\n--- After initial meter baseline ---')
pp(agent.STATE['meter_metrics'])

# Start inference measurement
iid = agent.start_inference_measurement('sim1')
print('\nStarted inference id=', iid)

# Simulate slight delay and meter update (increase total energy)
for step, extra_wh in enumerate([0.01, 0.009, 0.0]):
    # simulate new cumulative total
    current_total = (agent.STATE.get('meter_metrics') or {}).get('total_energy_wh') or 100.0
    new_total = current_total + extra_wh
    agent._update_meter_snapshot({'last_values': {'energy_wh': new_total, 'power_w': 0.55 + 0.01*step, 'voltage_v': 5.0, 'current_a': 0.11}}, status='connected', connected=True)
    print(f'Updated meter total to {new_total} Wh')
    time.sleep(0.05)

# End inference measurement (wait_for_sample True)
res = agent.end_inference_measurement(iid)
print('\n--- end_inference_measurement result ---')
pp(res)
print('\n--- energy_metrics after measurement ---')
pp(agent.STATE.get('energy_metrics'))
print('\n--- meter_metrics after measurement ---')
pp(agent.STATE.get('meter_metrics'))

# Now simulate second inference without baseline reset to see accumulation
iid2 = agent.start_inference_measurement('sim2')
print('\nStarted inference id=', iid2)
# No meter update this time (simulate missing sample)
res2 = agent.end_inference_measurement(iid2, wait_for_sample=False)
print('\n--- second end_inference_measurement result (no sample) ---')
pp(res2)
print('\n--- final meter_metrics ---')
pp(agent.STATE.get('meter_metrics'))
