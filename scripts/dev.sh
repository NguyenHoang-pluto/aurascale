#!/usr/bin/env bash
# Start PixelForge AI for local development (Linux / macOS).
# Both servers run in the foreground; Ctrl-C stops them together.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv_python="$root/backend/.venv/bin/python"

[[ -x "$venv_python" ]] || { echo "Backend virtualenv missing. Run scripts/setup.sh first." >&2; exit 1; }
[[ -d "$root/frontend/node_modules" ]] || { echo "Frontend deps missing. Run scripts/setup.sh first." >&2; exit 1; }

cleanup() { kill 0 2>/dev/null || true; }
trap cleanup EXIT INT TERM

(cd "$root/backend" && "$venv_python" -m uvicorn app.main:app --reload --port 8000) &
(cd "$root/frontend" && npm run dev) &

echo "Frontend  http://localhost:5173"
echo "Backend   http://127.0.0.1:8000"
echo "Swagger   http://127.0.0.1:8000/docs"
wait
