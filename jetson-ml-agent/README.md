# Jetson Nano ML Agent

ML inference agent for Jetson Nano DevKit 2GB running on Balena Cloud.

## Features
- TensorFlow Lite inference
- Auto-restart on reboot
- Energy monitoring
- Remote deployment from controller

## Deployment

```bash
balena push Jetson_Nano
```

## Device Requirements
- Jetson Nano DevKit 2GB
- Balena OS
- Minimum 4GB storage

## API Endpoints
- `GET /status` - Agent status
- `GET /metrics` - Device metrics
- `POST /deploy` - Deploy new model
- `POST /start` - Start inference
- `POST /stop` - Stop inference
- `POST /predict` - Run inference
- `POST /camera/fall-detect` - Run live webcam fall detection when a MoveNet fall model is deployed
- `POST /telemetry` - Report energy metrics

## Camera Fall Detection Model

The repo now includes a ready-to-deploy official TensorFlow Hub pose model:

- `ml-controller/new_models/movenet_singlepose_lightning_f16.tflite`

This model is intended for Jetson Nano webcam fall detection. During deploy, the
controller tags MoveNet uploads with `use_case=fall_detection_pose`, so the
agent automatically switches `/predict` into camera mode instead of dummy-input
mode.

### Expected camera behavior

- Input source: USB webcam on `/dev/video0`
- Default capture size: `640x480`
- Default detection window: `2.5s`
- Default max frames per request: `16`

### Quick test after deploy

```bash
curl http://127.0.0.1:8000/status

curl -X POST http://127.0.0.1:8000/camera/fall-detect \
  -H "Content-Type: application/json" \
  -d '{"duration_s": 2.5, "max_frames": 16}'
```

You can also call:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"duration_s": 2.5}'
```

When the deployed model is MoveNet, `/predict` will return `fall_detected`,
`fall_score`, and pose-window statistics from the webcam.

## FNB58 Integration Workflow

### 1. Deploy updated image

```bash
balena push <your_fleet_name>
```

### 2. Verify FNB58 on device host OS

```bash
balena device ssh <device_ip_or_uuid>
lsusb | grep -i fnirsi
dmesg | grep -Ei 'fnirsi|hidraw|usb'
```

### 3. Start telemetry collector in `ml-agent` container

Collector supports any external exporter that can output JSON.

Example payload from exporter command:

```json
{
	"energy_wh": 0.0182,
	"power_w": 2.9,
	"duration_s": 22.6,
	"timestamp": "2026-03-30T15:10:00Z"
}
```

Run collector:

```bash
balena device ssh <device_ip_or_uuid> ml-agent
python3 app/fnb58_telemetry_collector.py \
	--agent-url http://127.0.0.1:8000 \
	--json-command "python3 /data/fnb58_exporter.py" \
	--interval 3 \
	--source fnb58
```

Alternative (read from JSON file updated by exporter):

```bash
python3 app/fnb58_telemetry_collector.py \
	--agent-url http://127.0.0.1:8000 \
	--json-file /data/fnb58_latest.json \
	--interval 3 \
	--source fnb58
```

### 4. Run 10-30 inference cycles and compute MAPE

```bash
python3 app/benchmark_mape.py --runs 30 --delay 2 --csv /data/mape_report.csv
```

The script reads `predicted_mwh` from deployed model metadata and compares with measured telemetry from `/telemetry` history.
