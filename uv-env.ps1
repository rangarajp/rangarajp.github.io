# Dot-source before uv commands OR just use setup-python.ps1 once.
# uv defaults to `.venv`; this repo's real env is `.venv-qwen`.
# setup-python.ps1 creates a `.venv` → `.venv-qwen` junction so bare `uv add` works.
$env:UV_PROJECT_ENVIRONMENT = Join-Path $PSScriptRoot ".venv-qwen"
Write-Host "UV_PROJECT_ENVIRONMENT=$env:UV_PROJECT_ENVIRONMENT"
