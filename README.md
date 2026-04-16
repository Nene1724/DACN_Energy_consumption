
# DACN / KLTN — Energy Consumption (ML Controller + Device Agent)

This repo contains:

- `ml-controller/`: Flask web controller + analytics UI for model selection/deployment and energy prediction.
- `jetson-ml-agent/`: Device-side inference/telemetry agent (Jetson Nano) packaged for Balena/Docker.

## Quick start (Windows)

Start the controller web UI locally:

```powershell
./start_web.ps1
```

Then open:

- http://localhost:5000

Notes:

- `start_web.ps1` creates/uses a `.venv` with **Python 3.11** and installs `ml-controller/requirements.txt`.
- If you have exported device-specific energy models, place them under `ml-controller/artifacts/`.

## Run the controller manually (any OS)

From repo root:

**Windows (PowerShell)**

```powershell
py -3.11 -m venv .venv
./.venv/Scripts/python -m pip install -U pip
./.venv/Scripts/python -m pip install -r ml-controller/requirements.txt

cd ml-controller/python
../../.venv/Scripts/python app.py
```

**macOS / Linux**

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -U pip
./.venv/bin/python -m pip install -r ml-controller/requirements.txt

cd ml-controller/python
../../.venv/bin/python app.py
```

Controller binds to `0.0.0.0:5000`.

Optional environment variables (set in `ml-controller/.env` or your shell):

- `BALENA_API_TOKEN` (or `BALENA_API_KEY` / `BALENA_TOKEN`)
- `CONTROLLER_PUBLIC_URL` (useful when devices need to download models from a public URL)

## Device agent (Jetson)

The Jetson agent lives in `jetson-ml-agent/`.

- See `jetson-ml-agent/README.md` for Balena deployment instructions and the agent API endpoints.
- Local development is available via the `jetson-ml-agent/docker-compose.yml`.

## Ready-to-deploy camera fall model

An official TensorFlow Hub MoveNet TFLite model is bundled for Jetson Nano fall
detection experiments:

- `ml-controller/new_models/movenet_singlepose_lightning_f16.tflite`

Use the deployment UI to upload this file and deploy it to Jetson. The agent
will treat MoveNet uploads as `fall_detection_pose` models and expose live
camera inference through `/predict` and `/camera/fall-detect`.

## Test the deployed fall model with UR Fall dataset (MP4)

The UR Fall Detection Dataset page provides direct MP4 downloads (cam0/cam1 for
fall sequences; cam0 for ADL).

This repo includes a small batch runner that will:

1) download selected MP4s to a local cache (`.cache/ur_fall/`)
2) upload each MP4 to your Jetson agent (`/camera/upload-video`)
3) run fall detection on the uploaded file (`/camera/fall-detect`)
4) write a JSON report with TP/TN/FP/FN + timing

Run it from repo root (after you can reach the agent at port 8000):

```powershell
./.venv/Scripts/python ml-controller/python/ur_fall_batch_test.py \
	--agent http://<JETSON_IP>:8000 \
	--kind all --start 1 --end 5
```

Run the full dataset (can take a while):

```powershell
./.venv/Scripts/python ml-controller/python/ur_fall_batch_test.py \
	--agent http://<JETSON_IP>:8000 \
	--kind all --start 1 --end 40
```

## Repo structure

- `ml-controller/python/`: Flask app (`app.py`) + services
- `ml-controller/templates/`: HTML templates for the UI
- `ml-controller/artifacts/`: exported predictors/metadata (thresholds, model metadata, etc.)
- `ml-controller/model_store/`, `ml-controller/new_models/`: model files used by the analyzer
- `ml-controller/data/`: benchmark CSVs and sample measurements/logs

