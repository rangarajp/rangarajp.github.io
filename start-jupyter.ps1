# start-jupyter.ps1
# Starts a detached Jupyter server on :8888 using .venv-qwen.
# Uses Start-Process (not Start-Job) so the server survives after this script exits.

$PORT   = 8888
$TOKEN  = "qwendemo"
$ROOT   = $PSScriptRoot
$PYTHON = Join-Path $ROOT ".venv-qwen\Scripts\python.exe"
$LOG    = Join-Path $ROOT "jupyter.log"
$ERR    = Join-Path $ROOT "jupyter.err.log"
$PIDFILE = Join-Path $ROOT "jupyter.pid"

function Test-PortOpen {
    param([int]$Port)
    $null -ne (netstat -ano 2>$null | Select-String ":$Port\s")
}

if (-not (Test-Path $PYTHON)) {
    Write-Host "Missing $PYTHON" -ForegroundColor Red
    exit 1
}

if (Test-PortOpen -Port $PORT) {
    Write-Host ""
    Write-Host "Jupyter already running on port $PORT." -ForegroundColor Green
    Write-Host "  URL: http://localhost:${PORT}/?token=${TOKEN}" -ForegroundColor Cyan
    Write-Host ""
    exit 0
}

Write-Host ""
Write-Host "Starting Jupyter server on port $PORT ..." -ForegroundColor Yellow

# Detached process — survives after this script / task ends
$proc = Start-Process -FilePath $PYTHON -ArgumentList @(
    "-m", "jupyter", "notebook",
    "--no-browser",
    "--port=$PORT",
    "--NotebookApp.token=$TOKEN",
    "--NotebookApp.password=",
    "--notebook-dir=$ROOT"
) -WorkingDirectory $ROOT -WindowStyle Hidden `
  -RedirectStandardOutput $LOG -RedirectStandardError $ERR -PassThru

Set-Content -Path $PIDFILE -Value $proc.Id -Encoding ascii

$ok = $false
for ($i = 0; $i -lt 20; $i++) {
    Start-Sleep -Seconds 1
    if (Test-PortOpen -Port $PORT) {
        $ok = $true
        break
    }
    if ($proc.HasExited) {
        Write-Host "Jupyter exited early (code $($proc.ExitCode)). See jupyter.err.log" -ForegroundColor Red
        if (Test-Path $ERR) { Get-Content $ERR -Tail 20 }
        exit 1
    }
}

if ($ok) {
    Write-Host "Jupyter started successfully (pid $($proc.Id))." -ForegroundColor Green
} else {
    Write-Host "Port $PORT not open yet. Check jupyter.log / jupyter.err.log" -ForegroundColor Red
    if (Test-Path $ERR) { Get-Content $ERR -Tail 20 }
    exit 1
}

Write-Host ""
Write-Host "  URL: http://localhost:${PORT}/?token=${TOKEN}" -ForegroundColor Cyan
Write-Host ""
Write-Host "In Cursor: Select Kernel -> Existing Jupyter Server -> paste the URL." -ForegroundColor Gray
Write-Host ""
