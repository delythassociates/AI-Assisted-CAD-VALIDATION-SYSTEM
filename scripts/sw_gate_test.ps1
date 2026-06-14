# SolidWorks Gate 2 smoke test - open part and count faces via late-bound COM
$ErrorActionPreference = "Stop"
$stepPath = "F:\Varroc\test_cad\DongleHiderPlus_Shell.step"
$logPath = Join-Path $env:TEMP "eureka_addin.log"

$sw = [System.Runtime.InteropServices.Marshal]::GetActiveObject("SldWorks.Application")
if (-not $sw) { throw "SolidWorks is not running. Launch SW first." }

Write-Host "=== Gate 2: Open test part ==="
# swDocPART = 1
$errors = 0
$warnings = 0
$doc = $sw.OpenDoc6($stepPath, 1, 0, "", [ref]$errors, [ref]$warnings)
if (-not $doc) { throw "Failed to open $stepPath (errors=$errors warnings=$warnings)" }

$part = $sw.ActiveDoc
$bodies = $part.GetBodies2(0, $true)
$body = $bodies | Select-Object -First 1
if (-not $body) { throw "No solid body found" }

$faces = $body.GetFaces()
$faceCount = @($faces).Count
Write-Host "Face count: $faceCount"
Write-Host "Gate 2 PASS: part opened, $faceCount faces detected"

Write-Host ""
Write-Host "=== Checking eureka_addin.log ==="
Start-Sleep -Seconds 2
if (Test-Path $logPath) {
    Get-Content $logPath -Tail 15
    $metaLine = Select-String -Path $logPath -Pattern "BuildPartMetadata" | Select-Object -Last 1
    if ($metaLine) {
        Write-Host "Gate 2 metadata log: $($metaLine.Line)"
    } else {
        Write-Host "NOTE: Click Validate in Eureka task pane to complete Gates 5/6"
    }
} else {
    Write-Host "No eureka_addin.log found"
}

Write-Host ""
Write-Host "SW gate test complete."
