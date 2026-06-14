param(
    [string]$Configuration = "Debug"
)

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$OutputDir = Join-Path $ProjectDir "bin\$Configuration"

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
    exit 1
}

$cscPath = if ($csc -is [System.Management.Automation.CommandInfo]) { $csc.Source } else { $csc }

Write-Host "=== Building Eureka CATIA Connector ===" -ForegroundColor Cyan
Write-Host "Compiler: $cscPath"
Write-Host "Configuration: $Configuration"
Write-Host ""

# Create output directory
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

# Collect source files (from this folder and shared files from EurekaAddin)
$sourceFiles = @(
    (Join-Path $ProjectDir "CatiaConnector.cs"),
    (Join-Path $ProjectDir "..\EurekaAddin\TaskPane.cs"),
    (Join-Path $ProjectDir "..\EurekaAddin\Models.cs"),
    (Join-Path $ProjectDir "..\EurekaAddin\RestClient.cs")
)

# Collect references
$references = @(
    "System.dll",
    "System.Net.Http.dll",
    "System.Web.dll",
    "System.Windows.Forms.dll",
    "System.Drawing.dll",
    "System.Runtime.Serialization.dll",
    "System.Web.Extensions.dll",
    "System.Core.dll",
    "Microsoft.CSharp.dll",
    "Microsoft.VisualBasic.dll"
)

# Build Newtonsoft.Json reference
$nugetPath = Join-Path $env:USERPROFILE ".nuget\packages\newtonsoft.json\13.0.3\lib\net45\Newtonsoft.Json.dll"
if (-not (Test-Path $nugetPath)) {
    Write-Host "Downloading Newtonsoft.Json via NuGet..." -ForegroundColor Yellow
    & dotnet nuget locals all --clear 2>$null
    & dotnet new console -o "$env:TEMP\_eureka_catia_nuget" --force 2>$null | Out-Null
    & dotnet add "$env:TEMP\_eureka_catia_nuget" package Newtonsoft.Json --version 13.0.3 2>$null | Out-Null
    Remove-Item "$env:TEMP\_eureka_catia_nuget" -Recurse -Force -ErrorAction SilentlyContinue
}

if (Test-Path $nugetPath) {
    $references += $nugetPath
}
else {
    Write-Host "ERROR: Newtonsoft.Json.dll could not be resolved." -ForegroundColor Red
    exit 1
}

# Build reference args
$refArgs = $references | ForEach-Object { "/reference:$_" }

$targetExe = Join-Path $OutputDir "EurekaCatiaConnector.exe"

& $cscPath /target:exe /platform:x64 /debug /define:DEBUG `
    /out:$targetExe `
    $refArgs `
    $sourceFiles

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "=== BUILD SUCCESS ===" -ForegroundColor Green
    Write-Host "Output: $targetExe"
}
else {
    Write-Host ""
    Write-Host "=== BUILD FAILED ===" -ForegroundColor Red
    exit 1
}
