@echo off
title Eureka DFM — Backend Server
color 0A

echo.
echo  ███████╗██╗   ██╗██████╗ ███████╗██╗  ██╗ █████╗     ██████╗ ███████╗███╗   ███╗
echo  ██╔════╝██║   ██║██╔══██╗██╔════╝██║ ██╔╝██╔══██╗    ██╔══██╗██╔════╝████╗ ████║
echo  █████╗  ██║   ██║██████╔╝█████╗  █████╔╝ ███████║    ██║  ██║█████╗  ██╔████╔██║
echo  ██╔══╝  ██║   ██║██╔══██╗██╔══╝  ██╔═██╗ ██╔══██║    ██║  ██║██╔══╝  ██║╚██╔╝██║
echo  ███████╗╚██████╔╝██║  ██║███████╗██║  ██╗██║  ██║    ██████╔╝███████╗██║ ╚═╝ ██║
echo  ╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝    ╚═════╝ ╚══════╝╚═╝     ╚═╝
echo.
echo  AI-Powered Design for Manufacturability — Backend API v3.0
echo  ─────────────────────────────────────────────────────────────
echo.

:: Change to the project root (same folder as this script)
cd /d "%~dp0"

:: ── Kill any process already using port 8001 ──────────────────────────────
echo [1/4] Checking for existing processes on port 8001...
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":8001 " ^| findstr LISTENING') do (
    echo       Found PID %%p — terminating...
    taskkill /PID %%p /F >nul 2>&1
    timeout /t 1 /nobreak >nul
)
echo       Done.

:: ── Load environment variables from .env ──────────────────────────────────
echo [2/4] Loading environment variables from .env...
if exist ".env" (
    for /f "usebackq tokens=1,* delims==" %%a in (".env") do (
        if not "%%a"=="" if not "%%a:~0,1%"=="#" (
            set "%%a=%%b"
        )
    )
    echo       Loaded .env successfully.
) else (
    echo       WARNING: .env not found. Gemini will be unavailable.
)

:: ── Set defaults if not already set ──────────────────────────────────────
if not defined EUREKA_API_KEY set EUREKA_API_KEY=eureka-dev-key-change-me
if not defined EUREKA_DEV_MODE set EUREKA_DEV_MODE=true

:: ── Status report ─────────────────────────────────────────────────────────
echo [3/4] Configuration:
echo       API Key  : %EUREKA_API_KEY:~0,8%...
if defined GEMINI_API_KEY (
    echo       Gemini   : Configured [%GEMINI_API_KEY:~0,10%...]
) else (
    echo       Gemini   : NOT SET — rules-only mode
)
echo       Dev Mode : %EUREKA_DEV_MODE%
echo       Docs URL : http://localhost:8001/docs
echo.

:: ── Start uvicorn ─────────────────────────────────────────────────────────
echo [4/4] Starting Eureka DFM API on http://localhost:8001 ...
echo       Press Ctrl+C to stop the server.
echo  ─────────────────────────────────────────────────────────────
echo.

rem Call PowerShell script to handle environment and process cleanup
powershell -ExecutionPolicy Bypass -File "%~dp0start_backend.ps1"

echo.
echo  Server stopped. Press any key to close.
pause >nul
