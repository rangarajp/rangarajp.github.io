# start-jupyter.ps1
# Starts the Jupyter server for this workspace using .venv-qwen.

$PORT   = 8888
$TOKEN  = "qwendemo"
$ROOT   = $PSScriptRoot
$PYTHON = Join-Path $ROOT ".venv-qwen\Scripts\python.exe"
$LOG    = Join-Path $ROOT "jupyter.log"

# Check if already running
$listening = netstat -ano 2>$null | Select-String ":$PORT "
if ($listening) {
    Write-Host ""
    Write-Host "Jupyter already running on port $PORT." -ForegroundColor Green
    Write-Host "  URL: http://localhost:${PORT}/?token=${TOKEN}" -ForegroundColor Cyan
    Write-Host ""
    exit 0
}

Write-Host ""
Write-Host "Starting Jupyter server on port $PORT ..." -ForegroundColor Yellow

# Run as a background job so this script returns immediately
$job = Start-Job -ScriptBlock {
    param($python, $port, $token, $root, $log)
    Set-Location $root
    & $python -m jupyter notebook `
        --no-browser `
        "--port=$port" `
        "--NotebookApp.token=$token" `
        "--NotebookApp.password=" `
        "--notebook-dir=$root" *> $log
} -ArgumentList $PYTHON, $PORT, $TOKEN, $ROOT, $LOG

Start-Sleep -Seconds 5

# Verify
$running = netstat -ano 2>$null | Select-String ":$PORT "
if ($running) {
    Write-Host "Jupyter started successfully." -ForegroundColor Green
} else {
    Write-Host "Waiting a few more seconds..." -ForegroundColor Yellow
    Start-Sleep -Seconds 5
    $running = netstat -ano 2>$null | Select-String ":$PORT "
    if ($running) {
        Write-Host "Jupyter started successfully." -ForegroundColor Green
    } else {
        Write-Host "Could not confirm start. Check jupyter.log for errors." -ForegroundColor Red
        if (Test-Path $LOG) { Get-Content $LOG -Tail 10 }
    }
}

Write-Host ""
Write-Host "  URL: http://localhost:${PORT}/?token=${TOKEN}" -ForegroundColor Cyan
Write-Host ""
Write-Host "In Cursor: open any notebook -> Select Kernel -> Existing Jupyter Server -> paste the URL." -ForegroundColor Gray
Write-Host "Cursor remembers this. You only need to do this once." -ForegroundColor Gray
Write-Host ""
