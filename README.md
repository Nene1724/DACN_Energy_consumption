
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

## Repo structure

- `ml-controller/python/`: Flask app (`app.py`) + services
- `ml-controller/templates/`: HTML templates for the UI
- `ml-controller/artifacts/`: exported predictors/metadata (thresholds, model metadata, etc.)
- `ml-controller/model_store/`, `ml-controller/new_models/`: model files used by the analyzer
- `ml-controller/data/`: benchmark CSVs and sample measurements/logs

