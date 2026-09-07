#!/usr/bin/env bash
# One-time local setup for PixelForge AI (Linux / macOS).
# Usage: scripts/setup.sh [--cpu]
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
backend="$root/backend"
torch_requirements="requirements-cuda.txt"
[[ "${1:-}" == "--cpu" ]] && torch_requirements="requirements-cpu.txt"

[[ -f "$root/.env" ]] || { cp "$root/.env.example" "$root/.env"; echo "Created .env"; }

[[ -x "$backend/.venv/bin/python" ]] || python3 -m venv "$backend/.venv"
venv_python="$backend/.venv/bin/python"

"$venv_python" -m pip install --upgrade pip
"$venv_python" -m pip install -e "$backend[dev]"
"$venv_python" -m pip install -r "$backend/$torch_requirements"

(cd "$root/frontend" && npm install)

echo
"$venv_python" "$root/scripts/check_env.py"
echo
echo "Setup complete. Start the app with scripts/dev.sh"
