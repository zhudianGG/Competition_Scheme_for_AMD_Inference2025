#!/usr/bin/env bash
# Run E1-E5 sequentially. Each experiment writes to results/*.csv.
# Quick mode (RUN_QUICK=1) shrinks samples for sanity checking.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

QUICK_FLAG=""
[[ "${RUN_QUICK:-0}" == "1" ]] && QUICK_FLAG="--quick"

MODEL_PATH="${MODEL_PATH:-/sgl-workspace/SpecForge/models/MiniCPM-SALA}"
EVAL_DATA="${EVAL_DATA:-/sgl-workspace/SpecForge/SOAR-Toolkit/eval_dataset/perf_public_set.jsonl}"
EVAL_SCRIPT="${EVAL_SCRIPT:-/sgl-workspace/SpecForge/SOAR-Toolkit/eval_model.py}"

run_step() {
    local label="$1"; shift
    echo ""
    echo "============================================================"
    echo "  ${label}"
    echo "============================================================"
    "$@"
}

cd "${HERE}"

# E4 is offline — no GPU server. Run it first to fail fast if model files are bad.
run_step "E4 — FourOverSix block-scale profile" \
    python experiments/e4_fp4_block_scale_profile.py --model-path "${MODEL_PATH}"

run_step "E1 — Medusa vs EAGLE3 accept-rate sweep" \
    python experiments/e1_accept_rate_sweep.py ${QUICK_FLAG}

run_step "E3 — concurrency break-even" \
    python experiments/e3_concurrency_regression.py ${QUICK_FLAG}

run_step "E2 — long-context decay" \
    python experiments/e2_long_context_decay.py ${QUICK_FLAG}

run_step "E5 — GLA state-fork correctness" \
    python experiments/e5_gla_state_fork_check.py --skip-no-fork

run_step "Plots" \
    python plots/plot_results.py

echo ""
echo "[run_all] DONE. CSVs under results/, plots under plots/."
