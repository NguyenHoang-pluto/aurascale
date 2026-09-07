<#
.SYNOPSIS
    One-time local setup for PixelForge AI (Windows).

.PARAMETER Cpu
    Install the CPU-only PyTorch build instead of the CUDA build.

.EXAMPLE
    .\scripts\setup.ps1
    .\scripts\setup.ps1 -Cpu
#>
[CmdletBinding()]
param([switch]$Cpu)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$backend = Join-Path $root 'backend'
$venvPython = Join-Path $backend '.venv\Scripts\python.exe'

if (-not (Test-Path (Join-Path $root '.env'))) {
    Copy-Item (Join-Path $root '.env.example') (Join-Path $root '.env')
    Write-Host 'Created .env from .env.example' -ForegroundColor Green
}

if (-not (Test-Path $venvPython)) {
    Write-Host 'Creating backend virtualenv ...' -ForegroundColor Cyan
    python -m venv (Join-Path $backend '.venv')
}

Write-Host 'Installing backend dependencies ...' -ForegroundColor Cyan
& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -e "$backend[dev]"

$torchRequirements = if ($Cpu) { 'requirements-cpu.txt' } else { 'requirements-cuda.txt' }
Write-Host "Installing PyTorch from $torchRequirements ..." -ForegroundColor Cyan
& $venvPython -m pip install -r (Join-Path $backend $torchRequirements)

Write-Host 'Installing frontend dependencies ...' -ForegroundColor Cyan
Push-Location (Join-Path $root 'frontend')
try { npm install } finally { Pop-Location }

Write-Host ''
& $venvPython (Join-Path $root 'scripts\check_env.py')
Write-Host ''
Write-Host 'Setup complete. Start the app with .\scripts\dev.ps1' -ForegroundColor Green
