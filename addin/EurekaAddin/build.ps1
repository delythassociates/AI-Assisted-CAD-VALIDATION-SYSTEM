param(
    [string]$Configuration = "Debug",
    [string]$SwInteropPath = "F:\Softwares\Solid Works 25\SOLIDWORKS\api\redist"
)

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$OutputDir = Join-Path $ProjectDir "bin\$Configuration"

# Check prerequisites
$swInterop = @(
    "SolidWorks.Interop.sldworks.dll",
    "SolidWorks.Interop.swconst.dll",
    "SolidWorks.Interop.swcommands.dll",
    "SolidWorks.Interop.swmotionstudy.dll",
    "SolidWorks.Interop.swpublished.dll"
)

$missing = @()
foreach ($dll in $swInterop) {
    $path = Join-Path $SwInteropPath $dll
    if (-not (Test-Path $path)) {
        $missing += $dll
    }
}

if ($missing.Count -gt 0) {
    Write-Host "WARNING: SolidWorks Interop DLLs not found:" -ForegroundColor Yellow
    foreach ($dll in $missing) {
        Write-Host "  $dll (expected at $SwInteropPath)" -ForegroundColor Yellow
    }
    Write-Host "You can install the SolidWorks API SDK or copy the DLLs from a SolidWorks installation." -ForegroundColor Yellow
    Write-Host ""
}

# Check for csc compiler
$csc = Get-Command csc -ErrorAction SilentlyContinue
if (-not $csc) {
    # Try VS Build Tools path
    $cscPaths = @(
        "${env:ProgramFiles}\Microsoft Visual Studio\2022\BuildTools\MSBuild\Current\Bin\Roslyn\csc.exe",
        "${env:ProgramFiles}\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\Roslyn\csc.exe",
        "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\BuildTools\MSBuild\Current\Bin\Roslyn\csc.exe",
        "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\Roslyn\csc.exe"
    )
    foreach ($p in $cscPaths) {
        if (Test-Path $p) {
            $csc = $p
            break
        }
    }
}

if (-not $csc) {
    Write-Host "ERROR: C# compiler (csc.exe) not found." -ForegroundColor Red
    Write-Host "Install Visual Studio 2022 Build Tools with .NET desktop workload:" -ForegroundColor Red
    Write-Host "  vs_BuildTools.exe --add Microsoft.VisualStudio.Workload.ManagedDesktop --includeRecommended --passive"
    exit 1
}

$cscPath = if ($csc -is [System.Management.Automation.CommandInfo]) { $csc.Source } else { $csc }

Write-Host "=== Building Eureka Add-in ===" -ForegroundColor Cyan
Write-Host "Compiler: $cscPath"
Write-Host "Configuration: $Configuration"
Write-Host ""

# Create output directory
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

# Collect source files (exclude dead backup file — same class name causes duplicate type error)
$sourceFiles = @()
Get-ChildItem "$ProjectDir\*.cs" -Exclude "AssemblyInfo.cs", "*.cs.full" | ForEach-Object {
    $sourceFiles += $_.FullName
}

# Collect references
$references = @(
    "System.Net.Http.dll",
    "System.Web.dll",
    "System.Windows.Forms.dll",
    "System.Drawing.dll",
    "System.Runtime.Serialization.dll",
    "System.Web.Extensions.dll",
    "System.Core.dll",
    "Microsoft.CSharp.dll"
)

# Add SW interop if available
if ($missing.Count -eq 0) {
    foreach ($dll in $swInterop) {
        $references += (Join-Path $SwInteropPath $dll)
    }
}

# Build Newtonsoft.Json reference
$nugetPath = Join-Path $env:USERPROFILE ".nuget\packages\newtonsoft.json\13.0.3\lib\net45\Newtonsoft.Json.dll"
if (-not (Test-Path $nugetPath)) {
    Write-Host "Downloading Newtonsoft.Json via NuGet..." -ForegroundColor Yellow
    & dotnet nuget locals all --clear 2>$null
    & dotnet new console -o "$env:TEMP\_eureka_nuget" --force 2>$null | Out-Null
    & dotnet add "$env:TEMP\_eureka_nuget" package Newtonsoft.Json --version 13.0.3 2>$null | Out-Null
    Remove-Item "$env:TEMP\_eureka_nuget" -Recurse -Force -ErrorAction SilentlyContinue
    if (Test-Path $nugetPath) {
        $references += $nugetPath
    }
}
else {
    $references += $nugetPath
}

# Build reference args
$refArgs = $references | ForEach-Object { "/reference:$_" }

# Build (no strong name — PublicKeyToken=null to match SW 2025 working add-ins)
$targetDll = Join-Path $OutputDir "EurekaAddin.dll"
$targetPdb = Join-Path $OutputDir "EurekaAddin.pdb"

$targetType = if ($missing.Count -eq 0) { "library" } else { "module" }

# Check admin (required for regasm /codebase)
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "WARNING: Not running as Administrator. COM registration (regasm) will FAIL." -ForegroundColor Yellow
    Write-Host "Restart PowerShell as Administrator before running this script." -ForegroundColor Yellow
    Write-Host ""
}

# Check if SolidWorks is running (locked DLL)
$swProcess = Get-Process -Name "SLDWORKS" -ErrorAction SilentlyContinue
if ($swProcess) {
    Write-Host "WARNING: SolidWorks is running (PID $($swProcess.Id)). DLL may be locked." -ForegroundColor Yellow
    Write-Host "Close SolidWorks before rebuilding for successful deployment." -ForegroundColor Yellow
    Write-Host ""
}

& $cscPath /target:$targetType /platform:x64 /debug /define:DEBUG `
    /out:$targetDll `
    $refArgs `
    $sourceFiles

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "=== BUILD SUCCESS ===" -ForegroundColor Green
    Write-Host "Output: $targetDll"

    # Copy to SW addins
    $swAddinPath = "F:\Softwares\Solid Works 25\SOLIDWORKS\addins"
    if (Test-Path $swAddinPath) {
        Copy-Item $targetDll (Join-Path $swAddinPath "EurekaAddin.dll") -Force
        if (Test-Path $targetPdb) {
            Copy-Item $targetPdb (Join-Path $swAddinPath "EurekaAddin.pdb") -Force
        }
        Write-Host "Deployed to $swAddinPath" -ForegroundColor Green

        # Register COM
        $regasm = "${env:SystemRoot}\Microsoft.NET\Framework64\v4.0.30319\regasm"
        $dllPath = Join-Path $swAddinPath "EurekaAddin.dll"
        Write-Host "Unregistering previous COM registration..." -ForegroundColor Yellow
        & $regasm /unregister $dllPath /silent
        Write-Host "Registering COM with /codebase..." -ForegroundColor Yellow
        & $regasm $dllPath /codebase /silent
        if ($LASTEXITCODE -eq 0) {
            Write-Host "COM registration complete" -ForegroundColor Green
        }
        else {
            Write-Host "COM registration FAILED (exit code: $LASTEXITCODE). Run as Administrator." -ForegroundColor Red
        }
    }
}
else {
    Write-Host ""
    Write-Host "=== BUILD FAILED ===" -ForegroundColor Red
    exit 1
}
