#!/usr/bin/env bash
# Platform invocation: bash prepare_model.sh --input <orig> --output <out>
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "${SCRIPT_DIR}/quantize_gptq.py" "$@"
