@echo off
title SafeSight Demo Launcher
if "%~1"=="" (
    echo Usage: run_demo.bat path\to\footage.mp4 [--every-frame] [--loop]
    exit /b 1
)
call .venv\Scripts\activate.bat

:: Dashboard in its own window (open http://localhost:8501, the live view is inside)
start cmd /k "%~dp0.venv\Scripts\activate.bat" ^&^& streamlit run backend/dashboard.py

:: Analyse the recorded footage
python backend/safesight.py --video "%~1" %2 %3 %4

pause