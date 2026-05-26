#!/usr/bin/env bash
# Sourced by the eval platform after the base soar-toolkit image boots.
# Two responsibilities:
#   1. Patch the pre-installed sglang so the MiniCPM-SALA attention backends
#      run in float16 (needed by the GPTQ kernels — base image hardcodes bf16).
#   2. Export SGLANG_SERVER_ARGS so the launcher picks up our quant config.
set -euo pipefail

echo "[prepare_env] start $(date '+%F %T')"

# ---- patch hardcoded bf16 in the SALA attention backends ---------------------
BACKEND_DIR=/opt/SGLang-MiniCPM-SALA/packages/sglang-minicpm/python/sglang/srt/layers/attention

for pyfile in "${BACKEND_DIR}/minicpm_backend.py" "${BACKEND_DIR}/minicpm_sparse_utils.py"; do
    if [ -f "${pyfile}" ]; then
        sed -i 's/torch\.bfloat16/torch.float16/g' "${pyfile}"
        sed -i 's/"bfloat16"/"float16"/g'           "${pyfile}"
        echo "[prepare_env] patched $(basename "${pyfile}")"
    else
        echo "[prepare_env] WARNING: ${pyfile} not found — skipping patch"
    fi
done
find "${BACKEND_DIR}/__pycache__" \( -name "minicpm_backend*" -o -name "minicpm_sparse_utils*" \) -delete 2>/dev/null || true

# ---- launch args -------------------------------------------------------------
# Notes vs demo-quant:
#   - keep --quantization gptq + --dtype float16 + --disable-cuda-graph
#     (required while running GPTQ on the SALA sparse-attn path)
#   - add --kv-cache-dtype fp8_e5m2 to cut decode-stage memory bandwidth.
#     Safe with float16 activations; falls back gracefully if unsupported.
#   - chunked-prefill-size 8192 mirrors the demo; tune later once stable.
export SGLANG_SERVER_ARGS="--disable-radix-cache --attention-backend minicpm_flashinfer --chunked-prefill-size 8192 --skip-server-warmup --dense-as-sparse --quantization gptq --dtype float16 --kv-cache-dtype fp8_e5m2 --disable-cuda-graph"

echo "[prepare_env] SGLANG_SERVER_ARGS=${SGLANG_SERVER_ARGS}"
echo "[prepare_env] done"
