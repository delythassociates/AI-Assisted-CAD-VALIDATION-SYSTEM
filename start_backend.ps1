# Start HERMES DFM backend on port 8001 with optional .env secrets
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

# Kill any existing process on port 8001
$conn = Get-NetTCPConnection -LocalPort 8001 -State Listen -ErrorAction SilentlyContinue
if ($conn) {
    $pidToKill = $conn.OwningProcess
    Write-Host "Found existing process on port 8001 (PID $pidToKill). Terminating..." -ForegroundColor Yellow
    Stop-Process -Id $pidToKill -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
}

$envFile = Join-Path $Root ".env"
if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        if ($_ -match '^\s*([^#][^=]+)=(.*)$') {
            $name = $matches[1].Trim()
            $value = $matches[2].Trim().Trim('"').Trim("'")
            [Environment]::SetEnvironmentVariable($name, $value, "Process")
            Set-Item -Path "env:$name" -Value $value
        }
    }
    Write-Host "Loaded environment from .env" -ForegroundColor Green
} else {
    Write-Host "No .env file - copy .env.example to .env and set GEMINI_API_KEY" -ForegroundColor Yellow
}

if (-not $env:EUREKA_API_KEY) { $env:EUREKA_API_KEY = "eureka-dev-key-change-me" }
if (-not $env:EUREKA_DEV_MODE) { $env:EUREKA_DEV_MODE = "true" }

$geminiStatus = "NOT SET -- rules-only mode"
if ($env:GEMINI_API_KEY) {
    $geminiStatus = "configured OK"
}

Write-Host "Starting HERMES DFM API on http://localhost:8001 ..." -ForegroundColor Cyan
Write-Host "  GNN model: $env:EUREKA_GNN_MODEL_PATH (default from config)"
Write-Host "  Gemini:    $geminiStatus"
python -m uvicorn backend.main:app --port 8001 --host 0.0.0.0
