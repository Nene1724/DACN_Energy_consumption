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
- `POST /telemetry` - Report energy metrics
