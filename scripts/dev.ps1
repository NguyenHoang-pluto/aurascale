<#
.SYNOPSIS
    Start PixelForge AI for local development (Windows).

.DESCRIPTION
    Launches the FastAPI backend and the Vite dev server in separate windows.
    Run scripts/setup.ps1 first if the virtualenv or node_modules are missing.

.EXAMPLE
    .\scripts\dev.ps1
    .\scripts\dev.ps1 -BackendOnly
#>
[CmdletBinding()]
param(
    [switch]$BackendOnly,
    [switch]$FrontendOnly
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $root 'backend\.venv\Scripts\python.exe'

if (-not $FrontendOnly) {
    if (-not (Test-Path $venvPython)) {
        throw "Backend virtualenv not found. Run .\scripts\setup.ps1 first."
    }
    Write-Host 'Starting backend on http://127.0.0.1:8000 ...' -ForegroundColor Cyan
    Start-Process -FilePath $venvPython `
        -ArgumentList '-m', 'uvicorn', 'app.main:app', '--reload', '--port', '8000' `
        -WorkingDirectory (Join-Path $root 'backend')
}

if (-not $BackendOnly) {
    if (-not (Test-Path (Join-Path $root 'frontend\node_modules'))) {
        throw "Frontend dependencies not installed. Run .\scripts\setup.ps1 first."
    }
    Write-Host 'Starting frontend on http://localhost:5173 ...' -ForegroundColor Cyan
    Start-Process -FilePath 'npm' -ArgumentList 'run', 'dev' `
        -WorkingDirectory (Join-Path $root 'frontend')
}

Write-Host ''
Write-Host 'Frontend  http://localhost:5173' -ForegroundColor Green
Write-Host 'Backend   http://127.0.0.1:8000' -ForegroundColor Green
Write-Host 'Swagger   http://127.0.0.1:8000/docs' -ForegroundColor Green
