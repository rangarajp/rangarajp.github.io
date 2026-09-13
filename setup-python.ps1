# setup-python.ps1 — create/sync the notebook env with uv (not pip)
$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
Set-Location $Root
. (Join-Path $Root "uv-env.ps1")

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "uv not found. Install: https://docs.astral.sh/uv/getting-started/installation/" -ForegroundColor Red
    exit 1
}

$qwen = Join-Path $Root ".venv-qwen"
$venv = Join-Path $Root ".venv"

function Ensure-VenvJunction {
    if (-not (Test-Path $qwen)) {
        return
    }
    if (Test-Path $venv) {
        $item = Get-Item $venv -Force
        if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) {
            Write-Host ".venv already linked -> .venv-qwen" -ForegroundColor DarkGray
            return
        }
        Write-Host "Found a real .venv (uv default). Removing so we can link to .venv-qwen ..." -ForegroundColor Yellow
        Remove-Item $venv -Recurse -Force
    }
    New-Item -ItemType Junction -Path $venv -Target $qwen | Out-Null
    Write-Host "Linked .venv -> .venv-qwen (uv add/sync will use this)" -ForegroundColor Green
}

Write-Host "Syncing Python deps with uv into .venv-qwen ..." -ForegroundColor Cyan
uv sync
Ensure-VenvJunction

$python = Join-Path $qwen "Scripts\python.exe"
& $python -m ipykernel install --user --name qwen-demo --display-name "Qwen demo"
Write-Host ""
Write-Host "Done. Python: $python" -ForegroundColor Green
Write-Host "Add packages:  uv add <package>   (uses .venv -> .venv-qwen)" -ForegroundColor Gray
Write-Host "Or explicitly: . .\uv-env.ps1; uv add <package>" -ForegroundColor Gray
Write-Host "Never use pip in this repo — use uv." -ForegroundColor Gray
