@echo off
title Artefex
echo.
echo Starting Artefex...
echo.

REM Try python command first, then py launcher
python --version >nul 2>&1
if %errorlevel% equ 0 (
    python "%~dp0launcher.py"
    goto :end
)

py --version >nul 2>&1
if %errorlevel% equ 0 (
    py "%~dp0launcher.py"
    goto :end
)

echo ERROR: Python was not found on this system.
echo.
echo Please install Python 3.10 or newer from:
echo   https://www.python.org/downloads/
echo.
echo Make sure to check "Add Python to PATH" during install.
echo.
pause

:end
