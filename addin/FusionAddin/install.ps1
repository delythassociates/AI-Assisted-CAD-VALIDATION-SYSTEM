# install.ps1 - Deploy Eureka DFM 3.0 Add-in to Autodesk Fusion 360
# Run:  powershell -ExecutionPolicy Bypass -File install.ps1

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$AddInsDir = Join-Path $env:APPDATA "Autodesk\Autodesk Fusion 360\API\AddIns\EurekaDFM"

Write-Host ""
Write-Host "===================================="  -ForegroundColor Cyan
Write-Host "  Eureka DFM 3.0 Fusion Installer"     -ForegroundColor Cyan
Write-Host "===================================="  -ForegroundColor Cyan
Write-Host ""
Write-Host "Source:      $ScriptDir"
Write-Host "Destination: $AddInsDir"
Write-Host ""

# Check Fusion 360 API directory exists
$ApiDir = Join-Path $env:APPDATA "Autodesk\Autodesk Fusion 360\API"
if (-not (Test-Path $ApiDir)) {
    Write-Host "ERROR: Fusion 360 API directory not found." -ForegroundColor Red
    Write-Host "       $ApiDir" -ForegroundColor Red
    Write-Host "Make sure Fusion 360 is installed and launched at least once."
    exit 1
}

# Create AddIns directory if missing
$AddInsParent = Join-Path $ApiDir "AddIns"
if (-not (Test-Path $AddInsParent)) {
    New-Item -ItemType Directory -Path $AddInsParent -Force | Out-Null
    Write-Host "Created AddIns directory" -ForegroundColor Green
}

# Remove existing installation if any
if (Test-Path $AddInsDir) {
    $item = Get-Item $AddInsDir -Force
    if ($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) {
        $null = cmd /c rmdir `"$AddInsDir`"
        Write-Host "Removed existing symlink" -ForegroundColor Yellow
    } else {
        Remove-Item -Recurse -Force $AddInsDir
        Write-Host "Removed existing installation" -ForegroundColor Yellow
    }
}

# Try to create a directory symlink (preferred) or fall back to copy
$UseSymlink = $false
try {
    $null = cmd /c mklink /D `"$AddInsDir`" `"$ScriptDir`"
    if (Test-Path $AddInsDir) {
        $UseSymlink = $true
        Write-Host "Created symlink (changes auto-sync)" -ForegroundColor Green
    }
} catch {}

if (-not $UseSymlink) {
    Copy-Item -Path $ScriptDir -Destination $AddInsDir -Recurse -Force
    Write-Host "Copied files to AddIns folder" -ForegroundColor Green
}

# Verify required files
$RequiredFiles = @("EurekaDFM.py", "EurekaDFM.manifest", "palette.html")
$AllGood = $true
foreach ($f in $RequiredFiles) {
    $fp = Join-Path $AddInsDir $f
    if (Test-Path $fp) {
        Write-Host "    OK: $f" -ForegroundColor Green
    } else {
        Write-Host "    MISSING: $f" -ForegroundColor Red
        $AllGood = $false
    }
}

Write-Host ""
if ($AllGood) {
    Write-Host "===================================="  -ForegroundColor Green
    Write-Host "  Installation Complete"               -ForegroundColor Green
    Write-Host "===================================="  -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "  1. Open Autodesk Fusion 360"
    Write-Host "  2. Go to UTILITIES tab then ADD-INS (or Shift+S)"
    Write-Host "  3. Find EurekaDFM in Add-Ins tab"
    Write-Host "  4. Select it and click Run"
    Write-Host "  5. Eureka DFM button appears in toolbar"
    Write-Host "  6. Open a solid part and click it to validate"
    Write-Host ""
    Write-Host "Tip: Check Run on Startup to auto-load." -ForegroundColor Yellow
} else {
    Write-Host "===================================="  -ForegroundColor Red
    Write-Host "  Installation has missing files"      -ForegroundColor Red
    Write-Host "===================================="  -ForegroundColor Red
}
Write-Host ""
