#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_VERSION="3.11"
UV_BIN="${UV_BIN:-}"

find_uv() {
    if [[ -n "$UV_BIN" ]]; then
        if [[ ! -x "$UV_BIN" ]]; then
            echo "UV_BIN is not executable: $UV_BIN" >&2
            exit 1
        fi
        return
    fi

    if command -v uv >/dev/null 2>&1; then
        UV_BIN="$(command -v uv)"
        return
    fi

    for candidate in "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv"; do
        if [[ -x "$candidate" ]]; then
            UV_BIN="$candidate"
            return
        fi
    done

    if ! command -v curl >/dev/null 2>&1; then
        echo "curl is required to install uv." >&2
        exit 127
    fi

    echo "uv was not found; installing the standalone binary in the user account."
    curl -LsSf https://astral.sh/uv/install.sh | sh

    for candidate in "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv"; do
        if [[ -x "$candidate" ]]; then
            UV_BIN="$candidate"
            return
        fi
    done

    echo "uv installation completed, but the executable was not found." >&2
    exit 127
}

find_uv

echo "Using uv: $UV_BIN"
"$UV_BIN" --version
"$UV_BIN" python install --managed-python "$PYTHON_VERSION"
"$UV_BIN" sync \
    --project "$PROJECT_ROOT" \
    --locked \
    --managed-python \
    --python "$PYTHON_VERSION" \
    --reinstall-package opencv-python

"$PROJECT_ROOT/.venv/bin/python" - <<'PY'
import sys

import cv2
import msgpack
import websockets

print(f"Python: {sys.version.split()[0]}")
print(f"OpenCV: {cv2.__version__}")
print(f"msgpack: {msgpack.version}")
print(f"websockets: {websockets.__version__}")
PY

"$PROJECT_ROOT/.venv/bin/python" -m pytest -q "$PROJECT_ROOT/tests" "$PROJECT_ROOT/deploy_scripts"

echo
echo "VB3 robot server environment is ready: $PROJECT_ROOT/.venv"
echo "Start the server with: bash scripts/bimanual_smolvla.sh"
