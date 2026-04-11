# IoT ML Energy Manager - Quick Start Script
# Start Flask web server

Write-Host "======================================" -ForegroundColor Cyan
Write-Host " IoT ML Energy Manager - Quick Start" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

# Get script directory
$ROOT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$PYTHON_DIR = Join-Path $ROOT_DIR "ml-controller\python"
$VENV_DIR = Join-Path $ROOT_DIR ".venv"
$VENV_PYTHON = Join-Path $VENV_DIR "Scripts\python.exe"
$REQ_FILE = Join-Path $ROOT_DIR "ml-controller\requirements.txt"

function Resolve-Python311 {
    # Prefer Python 3.11 (needed for numpy/pandas wheels)
    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        try {
            $exe = & py -3.11 -c "import sys; print(sys.executable)" 2>$null
            if ($LASTEXITCODE -eq 0 -and $exe) { return $exe.Trim() }
        } catch {}
    }

    $candidates = @(
        "python3.11",
        "python"
    )

    foreach ($cmd in $candidates) {
        $found = Get-Command $cmd -ErrorAction SilentlyContinue
        if ($found) {
            try {
                $ver = & $cmd -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')" 2>$null
                if ($LASTEXITCODE -eq 0 -and $ver.Trim() -eq "3.11") {
                    $exe = & $cmd -c "import sys; print(sys.executable)" 2>$null
                    if ($LASTEXITCODE -eq 0 -and $exe) { return $exe.Trim() }
                }
            } catch {}
        }
    }

    return $null
}

function Ensure-Venv($pythonExe) {
    if (-not $pythonExe) {
        Write-Host "[ERROR] Python 3.11 not found. Install Python 3.11 and re-run." -ForegroundColor Red
        Write-Host "        Tip: winget install -e --id Python.Python.3.11 --scope user" -ForegroundColor Yellow
        exit 1
    }

    $needCreate = $false
    if (-not (Test-Path $VENV_PYTHON)) {
        $needCreate = $true
    } else {
        try {
            $ver = & $VENV_PYTHON -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')" 2>$null
            if ($LASTEXITCODE -ne 0 -or $ver.Trim() -ne "3.11") {
                $needCreate = $true
                Write-Host "[WARNING] Existing .venv is not Python 3.11 (found: $ver). Will rebuild venv." -ForegroundColor Yellow
            }
        } catch {
            $needCreate = $true
        }
    }

    if ($needCreate) {
        if (Test-Path $VENV_DIR) {
            $bak = "$VENV_DIR.bak-" + (Get-Date -Format "yyyyMMdd-HHmmss")
            Write-Host "[INFO] Backing up existing venv to $bak" -ForegroundColor Cyan
            Move-Item -Force $VENV_DIR $bak
        }
        Write-Host "[OK] Creating virtual environment (.venv) with Python 3.11..." -ForegroundColor Green
        & $pythonExe -m venv $VENV_DIR
        if ($LASTEXITCODE -ne 0) {
            Write-Host "[ERROR] Failed to create venv." -ForegroundColor Red
            exit 1
        }
    }
}

$python311 = Resolve-Python311
Ensure-Venv $python311

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

$pyExe = & $VENV_PYTHON -c "import sys; print(sys.executable)" 2>$null
$pyVer = & $VENV_PYTHON -c "import sys; print(sys.version.split()[0])" 2>$null
Write-Host "[OK] Using Python: $pyExe" -ForegroundColor Green
Write-Host "[OK] Python version: $pyVer" -ForegroundColor Green

# Install requirements
if (-not (Test-Path $REQ_FILE)) {
    Write-Host "[WARNING] Requirements file not found: $REQ_FILE" -ForegroundColor Yellow
    Write-Host "          Installing minimal deps (flask, requests)" -ForegroundColor Yellow
    & $VENV_PYTHON -m pip install --upgrade pip
    & $VENV_PYTHON -m pip install flask requests
} else {
    Write-Host "[OK] Installing requirements from: $REQ_FILE" -ForegroundColor Green
    & $VENV_PYTHON -m pip install --upgrade pip
    & $VENV_PYTHON -m pip install --upgrade -r $REQ_FILE
}

# Set Flask environment variables
$env:FLASK_APP = "app.py"
$env:FLASK_ENV = "development"
$env:FLASK_DEBUG = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host ""
Write-Host "======================================" -ForegroundColor Cyan
Write-Host " Starting Flask Server" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host "URL: http://localhost:5000" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop" -ForegroundColor Yellow
Write-Host ""

# Start Flask app
& $VENV_PYTHON app.py
