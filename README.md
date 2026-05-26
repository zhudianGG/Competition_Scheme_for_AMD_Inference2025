# SOAR 2026 Week 8 — MiniCPM-SALA EAGLE3 + Quantization Bench

Bench harness and results for two Week-8 SOAR tasks on MiniCPM-SALA (32-layer
hybrid lightning-linear-attention + sparse-attention model):

1. **EAGLE3 speculative decoding** — train a draft head, measure per-position
   accept length, and probe end-to-end wall-clock speedup.
2. **PTQ quantization** — drop-in FP8 W8A8, MXFP4 W4A4, and NVFP4 FourOverSix
   on the bf16 target, with diagnostics for *why* drafts trained on bf16
   hiddens collapse under quant.

**Canonical writeup:** [`results/SUMMARY.md`](results/SUMMARY.md). The README
below is the index — the SUMMARY is the report.

## Headline findings

- The trained EAGLE3 head works on the bf16 target: per-position accept
  0.515 → **0.838** across 10 epochs of `sala_eagle3_v2`.
- Multi-step HF spec-decode (K=4) reaches **0.64 tokens / round**, ~70% of a
  stock production EAGLE3 head's level on a comparable prompt set.
- End-to-end wall-clock speedup requires sglang. The OpenBMB fork
  (`OpenBMB/sglang@minicpm_sala`) is the only path; on our host stack
  (torch 2.9.1 / CUDA 12.9 / flashinfer 0.6.3 / tilelang 0.1.9) decode
  crashes in the sparse-decode kernels. We have a reproduction kit for the
  fork's native container — see [`sala_docker_bundle/`](sala_docker_bundle/).
- **Drop-in PTQ + reused bf16-trained EAGLE3 draft is unusable.** All seven
  quant variants we tested (full FP8, full MXFP4, NVFP4 FourOverSix, plus
  four diagnostic splits) collapse to **0.087 – 0.102** accept length while
  perplexity remains within 2% of bf16. Both hidden-drift and trajectory-
  drift are *independently* sufficient to break the draft; MLP-only quant
  (attention kept bf16) provides no protection. Drafts must be retrained
  per quant config.

## Layout

```
soar_2026_w8_bench/
├── README.md                    ← you are here
├── results/SUMMARY.md           ← canonical writeup (E1–E9 + fork notes)
├── experiments/                 ← all probe scripts (e1_*.py … e9_*.py)
├── results/                     ← CSV outputs + e8_picks.json
├── logs/                        ← run logs (e6_*.log, e7_mxfp4.log, …)
├── lib/                         ← shared runner / metrics / server helpers
├── plots/                       ← matplotlib helpers + figures
├── data/                        ← (empty — prompts live in SpecForge)
├── sala_docker_bundle/          ← reproducible OpenBMB-fork test kit
├── sala_docker_bundle.zip       ← same, packaged for transfer
├── prepare_env.sh               ← uv-installs sglang editable + deps
├── prepare_model.sh             ← model staging helper
├── preprocess_model.py          ← model-format preprocessing
└── run_all.sh                   ← top-level driver
```

## Experiment index

Each `eN_*.py` is standalone and writes a CSV under `results/`. Scripts share
no internal state across runs — re-runnable in any order against the bf16
target.

| script | purpose | output |
|---|---|---|
| `e1_offline_accept_length.py` | per-position accept length on the bf16 anchor | `results/e1_v2_epoch9.csv` |
| `e1_accept_rate_sweep.py` | top-k accept-rate sweep | — |
| `e2_hf_spec_decode.py` | multi-step HF spec-decode harness (K=4) | `results/e2_hf_spec_decode.csv` |
| `e2_sweep_checkpoints.sh` | sweep across v2 training checkpoints | `results/e2_epoch_*.csv` |
| `e2_long_context_decay.py` | accept-length decay vs prompt length | — |
| `e3_sglang_eagle3_ref.py` | stock gpt-oss EAGLE3 control via sglang | `results/e3_sglang_eagle3.csv` |
| `e3_concurrency_regression.py` | concurrency-vs-tps regression probe | — |
| `e4_fp4_block_scale_profile.py` | per-block FP4 scale distribution | `results/e4_block_scale.csv` |
| `e5_gla_state_fork_check.py` | GLA state-fork consistency for spec verify | — |
| `e6_bf16_nll_probe.py` | bf16 NLL/PPL anchor | logged inline |
| `e6_fp8_quality.py` | FP8 W8A8 quant + E1 protocol + NLL | `results/e6_fp8_quality.csv` |
| `e6_greedy_agreement.py` | bf16 vs FP8 greedy token-agreement | logged inline |
| `e7_mxfp4_quality.py` | MXFP4 W4A4 swap + E1 + NLL + agreement | `results/e7_mxfp4_quality.csv` |
| `e8_nvfp4_foursix.py` | NVFP4 FourOverSix per-tensor split + E1 + NLL | `results/e8_nvfp4_foursix.csv`, `results/e8_picks.json` |
| `e9_quant_diagnostics.py` | hidden-only / traj-only / MLP-only diagnostics | `results/e9_*.csv` |
| `e_sala_bf16_probe.py` | bf16 sanity probe against running sglang server | — |
| `state_fork_switch.patch` | unmerged patch for hybrid-backend state forking | — |

## Reproducing

Pinned environment lives in `mminf` (`/root/venvs/mminf`). For quant runs
you need `torchao` ≥ 0.9 (FP8 + MX kernels) and `flash_attention_2`. The
quant scripts default to:

- target: `/sgl-workspace/SpecForge/models/MiniCPM-SALA`
- draft: `/sgl-workspace/SpecForge/outputs/sala_eagle3_v2/epoch_9_step_18750`
- vocab map: `/sgl-workspace/SpecForge/cache/vocab_mapping/210c72934bd4b0e45cdd3a75d8e01620.pt`
- prompts: `/sgl-workspace/SpecForge/SOAR-Toolkit/eval_dataset/perf_public_set.jsonl`

Override with `--target / --draft / --vocab-mapping / --prompts`.

Examples:

```bash
# bf16 anchor
python experiments/e1_offline_accept_length.py

# FP8 W8A8 quality
python experiments/e6_fp8_quality.py

# NVFP4 FourOverSix (per-tensor MSE-based split)
python experiments/e8_nvfp4_foursix.py

# Diagnostic: which drift breaks the draft?
CUDA_VISIBLE_DEVICES=1 python experiments/e9_quant_diagnostics.py \
    --mode hidden_only --quant fp8 --tag d1_fp8
```

## sglang fork status

The OpenBMB `minicpm_sala` fork (HEAD `d29fb13`) is the only sglang branch
that registers `MiniCPMSALAForCausalLM`. On our host the sparse-decode
kernels crash on prompts > ~2k tokens (`prepare_for_decode` /
`req_to_token` OOB after prefill) and CUDA-graph capture trips a separate
`compressed_attention` topk OOB. We exhausted deep-debug avenues and
WONTFIXed in-tree.

[`sala_docker_bundle/`](sala_docker_bundle/) reproduces the fork in its
pinned `nvcr.io/nvidia/pytorch:25.01-py3` container so the host stack vs
ABI-mismatch question can be settled cleanly. The bundle README has the
build + smoke-test instructions.

## Open tasks (not in this repo)

- **#22** Wire EAGLE3 draft head through sglang and measure end-to-end
  speedup. Blocked on the fork stability question above.
- **#23** Apply `experiments/state_fork_switch.patch` to the hybrid backend
  once a working sglang baseline exists.
- Retrain EAGLE3 draft heads on quantized target hiddens for each
  deployment quant (FP8 / MXFP4 / NVFP4). Required before any quant + spec-
  decode end-to-end demo.
