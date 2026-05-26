#!/bin/bash
# Sweep e2 accept-length across all available checkpoints to chart head improvement
# vs training steps.
set -euo pipefail

OUT_DIR=${OUT_DIR:-/sgl-workspace/SpecForge/outputs/sala_eagle3_v2}
N_PROMPTS=${N_PROMPTS:-10}
GEN_TOKENS=${GEN_TOKENS:-64}
K=${K:-4}
RESULTS_DIR=$(dirname $(realpath $0))/../results
mkdir -p $RESULTS_DIR

# pair each checkpoint with its own training-time vocab mapping
V1_VM=/sgl-workspace/SpecForge/cache/vocab_mapping/210c72934bd4b0e45cdd3a75d8e01620.pt
V2_VM=/sgl-workspace/SpecForge/cache/vocab_mapping/48548f943af8f668b23ab888755979d8.pt

declare -A CKPT_VM
CKPT_VM[/sgl-workspace/SpecForge/outputs/sala_eagle3_v1/epoch_0_step_1875]=$V1_VM
for d in $OUT_DIR/epoch_*_step_*; do
  [ -d "$d" ] && CKPT_VM[$d]=$V2_VM
done
CKPTS=("${!CKPT_VM[@]}")

SUMMARY=$RESULTS_DIR/e2_sweep_summary.csv
echo "checkpoint,k,mean_accept_per_round,n_rounds" > $SUMMARY

for ckpt in "${CKPTS[@]}"; do
  name=$(basename "$ckpt")
  echo "=== $name ==="
  out=$RESULTS_DIR/e2_${name}.csv
  vm=${CKPT_VM[$ckpt]}
  python $(dirname $(realpath $0))/e2_hf_spec_decode.py \
      --draft "$ckpt" --vocab-mapping "$vm" \
      --n-prompts $N_PROMPTS --gen-tokens $GEN_TOKENS \
      --k $K --skip-greedy --out "$out" 2>&1 | tail -5
  overall=$(tail -1 "$out" | cut -d, -f3)
  rounds=$(tail -1 "$out" | cut -d, -f2)
  echo "$name,$K,$overall,$rounds" >> $SUMMARY
done
echo
echo "=== Summary ==="
cat $SUMMARY
