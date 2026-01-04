# Deploy RPI ML Agent to Balena Cloud
# Fleet: Raspberry_Pi

Write-Host "=== Balena Cloud Deployment Script ===" -ForegroundColor Cyan
Write-Host ""

# Check Balena CLI
Write-Host "Checking Balena CLI..." -ForegroundColor Yellow
try {
    $balenaVersion = balena --version 2>&1
    Write-Host "✓ Balena CLI installed: $balenaVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Balena CLI not found. Please install: https://github.com/balena-io/balena-cli" -ForegroundColor Red
    exit 1
}

# Check login status
Write-Host ""
Write-Host "Checking authentication..." -ForegroundColor Yellow
$whoami = balena whoami 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Not logged in to Balena Cloud" -ForegroundColor Red
    Write-Host "Please run: balena login" -ForegroundColor Yellow
    exit 1
}
Write-Host "✓ Logged in as: $whoami" -ForegroundColor Green

# Set fleet name
$fleet = "Raspberry_Pi"

# Get current directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

Write-Host ""
Write-Host "=== Deployment Configuration ===" -ForegroundColor Cyan
Write-Host "  Fleet: $fleet" -ForegroundColor White
Write-Host "  Directory: $scriptDir" -ForegroundColor White
Write-Host "  Service: ml-agent (port 8000)" -ForegroundColor White
Write-Host ""

# Confirm
$confirm = Read-Host "Deploy to fleet '$fleet'? (y/N)"
if ($confirm -ne 'y' -and $confirm -ne 'Y') {
    Write-Host "Deployment cancelled." -ForegroundColor Yellow
    exit 0
}

# Push to Balena
Write-Host ""
Write-Host "=== Pushing to Balena Cloud ===" -ForegroundColor Cyan
Write-Host "This may take several minutes..." -ForegroundColor Yellow
Write-Host ""

balena push $fleet

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "=== Deployment Successful! ===" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "  1. Wait for devices to download and start the service" -ForegroundColor White
    Write-Host "  2. Check device status: balena devices" -ForegroundColor White
    Write-Host "  3. View logs: balena logs <device-name>" -ForegroundColor White
    Write-Host "  4. Test endpoint: curl http://<device-ip>:8000/status" -ForegroundColor White
    Write-Host ""
    Write-Host "Service endpoints:" -ForegroundColor Cyan
    Write-Host "  Status:  http://<device-ip>:8000/status" -ForegroundColor White
    Write-Host "  Deploy:  http://<device-ip>:8000/deploy" -ForegroundColor White
    Write-Host "  Telemetry: http://<device-ip>:8000/telemetry" -ForegroundColor White
} else {
    Write-Host ""
    Write-Host "=== Deployment Failed ===" -ForegroundColor Red
    Write-Host "Check the error messages above." -ForegroundColor Yellow
    exit 1
}
