@echo off
REM start-jupyter.bat — starts the Jupyter server in the background
REM Run from the workspace root or let the VS Code task run it automatically.

set PORT=8888
set TOKEN=qwendemo
set PYTHON=%~dp0.venv-qwen\Scripts\python.exe

echo.
echo Starting Jupyter server on port %PORT% ...
echo.

start "" /min "%PYTHON%" -m jupyter notebook ^
    --no-browser ^
    --port=%PORT% ^
    --NotebookApp.token=%TOKEN% ^
    --NotebookApp.password= ^
    --notebook-dir="%~dp0"

echo Done. Server starting in the background.
echo.
echo   URL: http://localhost:%PORT%/?token=%TOKEN%
echo.
echo In Cursor: open any notebook - Select Kernel - Existing Jupyter Server - paste the URL above.
echo Cursor remembers this. You only need to do it once.
echo.
