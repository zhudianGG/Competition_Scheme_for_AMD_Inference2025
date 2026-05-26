#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${INPUT:?INPUT (source model dir) must be set}"
: "${OUTPUT:?OUTPUT (processed model dir) must be set}"

if [[ "${QUANTIZE:-0}" == "1" ]]; then
    echo "[prepare_model] quantizing ${INPUT} -> ${OUTPUT}"
    python "${HERE}/preprocess_model.py" --input "${INPUT}" --output "${OUTPUT}" --quantize
else
    echo "[prepare_model] copying ${INPUT} -> ${OUTPUT}"
    mkdir -p "${OUTPUT}"
    cp -r "${INPUT}/." "${OUTPUT}/"
fi
