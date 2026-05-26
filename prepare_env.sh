#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SGLANG_REPO="${SGLANG_REPO:-/sgl-workspace/sglang}"

echo "[prepare_env] installing sglang editable from ${SGLANG_REPO}"
uv pip install --no-deps -e "${SGLANG_REPO}/python"

echo "[prepare_env] installing benchmark deps"
uv pip install matplotlib pandas aiohttp tqdm requests

echo "[prepare_env] default server args"
export SGLANG_SERVER_ARGS="${SGLANG_SERVER_ARGS:-} --speculative-algorithm EAGLE3 --speculative-num-draft-tokens 5"
echo "  SGLANG_SERVER_ARGS=${SGLANG_SERVER_ARGS}"

echo "[prepare_env] done"
