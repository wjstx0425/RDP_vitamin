#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${VB3_SERVER_PYTHON:-$PROJECT_ROOT/.venv/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "VB3 server environment not found: $PYTHON_BIN" >&2
    echo "Run 'uv sync --locked --managed-python --python 3.11' first." >&2
    exit 127
fi

arguments=("$@")
default_token_file="${VB3_TOKEN_FILE:-$PROJECT_ROOT/token_list.txt}"
needs_token=true
has_token_file=false
for argument in "$@"; do
    if [[ "$argument" == "-h" || "$argument" == "--help" ]]; then
        needs_token=false
    fi
    if [[ "$argument" == "--token-file" || "$argument" == --token-file=* ]]; then
        has_token_file=true
    fi
done

temporary_token_file=""
cleanup() {
    if [[ -n "$temporary_token_file" ]]; then
        rm -f "$temporary_token_file"
    fi
}
trap cleanup EXIT INT TERM

if [[ "$needs_token" == true && "$has_token_file" == false ]]; then
    if [[ -s "$default_token_file" ]]; then
        arguments+=(--token-file "$default_token_file")
    else
        if [[ -z "${VB_ROBOT_TOKEN:-}" ]]; then
            read -rsp "VB robot token: " VB_ROBOT_TOKEN
            echo
        fi
        if [[ -z "$VB_ROBOT_TOKEN" ]]; then
            echo "VB robot token must not be empty" >&2
            exit 2
        fi
        temporary_token_file="$(mktemp "${TMPDIR:-/tmp}/vb3-server-token.XXXXXX")"
        chmod 600 "$temporary_token_file"
        printf '%s\n' "$VB_ROBOT_TOKEN" > "$temporary_token_file"
        arguments+=(--token-file "$temporary_token_file")
    fi
fi

export PYTHONUNBUFFERED=1
cd "$PROJECT_ROOT"
"$PYTHON_BIN" deploy_scripts/bimanual_smolvla_online.py "${arguments[@]}"
