#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# The connected RDP client negotiates vitac, 224px observations and one-step
# receive-time scheduling through policy_type=rdp.
exec "${SCRIPT_DIR}/bimanual_smolvla.sh" --max-executed-actions 1 "$@"
