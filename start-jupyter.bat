@echo off
REM Prefer start-jupyter.ps1 — it verifies port 8888 before claiming success.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start-jupyter.ps1"

