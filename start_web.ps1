# IoT ML Energy Manager - Quick Start Script
# Start Flask web server

Write-Host "======================================" -ForegroundColor Cyan
Write-Host " IoT ML Energy Manager - Quick Start" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

# Get script directory
$ROOT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$PYTHON_DIR = Join-Path $ROOT_DIR "ml-controller\python"

# Check if artifacts exist
$ARTIFACTS_DIR = Join-Path $ROOT_DIR "ml-controller\artifacts"
if (-not (Test-Path $ARTIFACTS_DIR)) {
    Write-Host "[WARNING] Artifacts directory not found!" -ForegroundColor Yellow
    Write-Host "          Creating artifacts directory..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Path $ARTIFACTS_DIR | Out-Null
}

# Check for device-specific models
$JETSON_MODEL = Join-Path $ARTIFACTS_DIR "jetson_energy_model.pkl"
$RPI5_MODEL = Join-Path $ARTIFACTS_DIR "rpi5_energy_model.pkl"

if (Test-Path $JETSON_MODEL) {
    Write-Host "[OK] Jetson model found" -ForegroundColor Green
} else {
    Write-Host "[WARNING] Jetson model not found - run notebook to export models" -ForegroundColor Yellow
}

if (Test-Path $RPI5_MODEL) {
    Write-Host "[OK] RPi5 model found" -ForegroundColor Green
} else {
    Write-Host "[WARNING] RPi5 model not found - run notebook to export models" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Working directory: $PYTHON_DIR" -ForegroundColor Cyan
Write-Host ""

# Change to python directory
Set-Location $PYTHON_DIR

# Check if Flask is installed
try {
    $flaskCheck = python -c "import flask; print(flask.__version__)" 2>&1
    Write-Host "[OK] Flask version: $flaskCheck" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Flask not installed! Installing..." -ForegroundColor Red
    python -m pip install flask requests
}

# Set Flask environment variables
$env:FLASK_APP = "app.py"
$env:FLASK_ENV = "development"
$env:FLASK_DEBUG = "1"

Write-Host ""
Write-Host "======================================" -ForegroundColor Cyan
Write-Host " Starting Flask Server" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host "URL: http://localhost:5000" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop" -ForegroundColor Yellow
Write-Host ""

# Start Flask app
python app.py
