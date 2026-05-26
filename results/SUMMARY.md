# SOAR 2026 Week 8 — MiniCPM-SALA EAGLE3 results

## Target / draft

- Target: MiniCPM-SALA (32-layer hybrid lightning + sparse attention, custom HF model — not in sglang registry)
- Draft (trained this week): `sala_eagle3_v2` (10 epochs, lr=1e-4, bs=8, max_len=2048, sharegpt)
  - Final ckpt `epoch_9_step_18750`: train acc=0.92, loss=0.31
- Aux hidden layers used: `[1, 15, 28]` (low / mid / high)
- Eval prompts: `SOAR-Toolkit/eval_dataset/perf_public_set.jsonl`, first 10

## E1 — offline per-position accept length

Teacher-forced target rollout (64 tokens), then for each position compare draft top-1 (mapped d2t→target) against target argmax. "Accept length" = longest run of consecutive matches starting at that position.

| ckpt | mean accept length | n_positions |
|---|---|---|
| sala_eagle3_v1 / step 1875 | 0.515 | 640 |
| sala_eagle3_v2 / step 18750 | **0.838** | 640 |

10-epoch training raised per-position accept ~63%.

## E2 — multi-step HF spec decoding (K=4)

Hand-rolled harness in `experiments/e2_hf_spec_decode.py` (sglang has no SALA model class). Per round: seed target forward → draft proposes K=4 tokens → verify target forward → accept longest matching prefix + bonus. Two full-prefix target forwards per round (no KV cache reuse — SALA's `LightningCacheLayer` / `InfLLMv2CacheLayer` don't implement `.crop()`).

Sweep across v2 checkpoints (`experiments/e2_sweep_checkpoints.sh`):

| checkpoint | mean accept / round | n_rounds |
|---|---|---|
| v1 / epoch_0_step_1875 | 0.440 | 450 |
| v2 / epoch_2_step_5000 | 0.561 | 412 |
| v2 / epoch_5_step_10000 | 0.580 | 407 |
| v2 / epoch_7_step_15000 | 0.621 | 398 |
| **v2 / epoch_9_step_18750** | **0.639** | 393 |

Final v2 ckpt (10 prompts, 64 gen tokens, K=4):
- mean accept length: 0.639 / round (excl. bonus → ~1.64 tokens / round including bonus)
- spec throughput: 20.9 tok/s
- greedy baseline: 30.9 tok/s
- wall-clock ratio: **0.68×** — slowdown due to 2 full-prefix forwards per round vs 1 forward per token in greedy

Saturation: K=4 already near-optimal. K=6 and K=8 give ~same accept length (compounding draft error caps the gain).

## E3 — sglang stock-model calibration

`experiments/e3_sglang_eagle3_ref.py` runs the same prompt set through sglang's production EAGLE3 spec-decoding stack with stock-architecture models. SALA cannot run through sglang (not registered + custom hybrid cache); this is the apples-to-oranges control showing what a known-good EAGLE3 pipeline produces on the same prompts.

Setup: `openai/gpt-oss-20b` (mxfp4 MoE, 20B params) + `baseten-admin/gpt-oss-20b-eagle3` head, K=4 draft tokens, 3 spec steps, topk=1.

| config | tps | spec_verify_ct (= rounds) | tokens/round |
|---|---|---|---|
| gpt-oss + EAGLE3 (sglang) | **118.0 tok/s** | 33.2 / 64 tok | ~1.93 |
| gpt-oss no spec (sglang)  | 105.2 tok/s | n/a | 1.0 |
| **speedup** | **1.12×** | | |

Tokens/round ≈ 1.93 corresponds to mean accept length ≈ 0.93 in our SALA terminology (excl. bonus). So SALA EAGLE3 (0.64) is in the same ballpark as a stock production EAGLE3 head (0.93) on this prompt set.

## Why SALA wall-clock looks worse than sglang's

| component                        | sglang gpt-oss path | our SALA HF harness |
|---|---|---|
| target KV cache across rounds    | ✓                   | ✗ (recomputed per round) |
| CUDA graphs for decode           | ✓                   | ✗ |
| draft head incremental forward   | ✓                   | ✗ (re-runs over full prefix) |
| draft uses target KV indirectly  | ✓ (graph-fused)     | n/a |
| spec accept length               | ~0.93 / round       | 0.64 / round |

The 0.68× spec-vs-greedy wall-clock ratio in E2 is dominated by the second target forward each round, not by the accept-length math. With sglang-style KV caching the same accept length would yield ~1.4× speedup ((L+1) per (1+ε) forward vs 1-per-1 greedy).

## Bottom line for the writeup

- The trained EAGLE3 head works: per-position accept 0.515 → 0.838, multi-step 0.44 → 0.64 across 10 epochs.
- Accept length on the SOAR perf set (0.64 / round, K=4) is ~70% of what sglang's stock production EAGLE3 head delivers on a comparable prompt set (~0.93).
- Spec-decoding wall-clock gains for SALA require porting it into sglang's model registry (multi-day effort: implement `MiniCPMSALAForCausalLM` using `BailingMoELinearAttention` / `LightningAttentionBackend` / `HybridLinearAttnBackend` / `nsa_backend` building blocks already in sglang). Until then, our HF harness validates the head but cannot show end-to-end speedup.

## sglang integration attempt (2026-05-23)

Discovered OpenBMB's official sglang fork (`OpenBMB/sglang`, branch `minicpm_sala`, HEAD d29fb13) provides a SALA backend (`MiniCPMSparseBackend`) and the `MiniCPMSALAForCausalLM` model class. Installed the fork editable into `mminf`, built `infllmv2_cuda_impl` and `sparse_kernel` CUDA libs from the fork's `3rdparty/` submodules.

**Functional state:** server boots and serves single-request short-prompt generation:
- `Hello, my name is` → 32 tokens in 3.46 s ≈ **9.2 tok/s** (bf16, `--disable-cuda-graph`).

**Blocking issues for SOAR week 8 deliverables on the fork:**
1. **CUDA graph capture fails** at `compressed_attention` topk (`selected index k out of range`) — synthetic capture batches have fewer pooled blocks than `sparse_topk=64`. Forces `--disable-cuda-graph`, which kills decode throughput.
2. **Decode path crashes on prompts < ~4 k tokens** with the same OOB topk error. Patched with `min(topk, block_score.size(-1))` + zero-pad, but downstream `flashinfer.decode.plan()` then hits `cudaErrorIllegalAddress` once the cache grows past ~30 decode steps — the padded indices produce inconsistent `sparse_page_table` entries.
3. **Empty-tensor `.max()` bug** in `update_batch_for_sparse` when no batch element exceeds `dense_len=8192` (patched).
4. **Context-length cap of 4 k** in our test (model supports 512 k) means the fork's intended sparse regime (`dense_len=8192`) is never legitimately exercised.

Net: the fork's SALA backend is designed and tested only for the long-context regime (prompts ≥ 8 k tokens). Short-prompt workloads — exactly what EAGLE3 spec decoding and our perf eval set drive — hit unhandled edge cases in `MiniCPMSparseBackend`. Patching this would require porting OpenBMB's `MiniCPMHybridReqToTokenPool` short-prompt routing + reworking `compressed_attention` to genuinely fall back to dense when `block_score.size(-1) < sparse_topk`.

**Bottom-line decision:** continue SOAR week 8 quant + spec-dec deliverables on the HF path (which already produced E1–E5). The sglang fork integration is documented as a known-failed path so we don't re-attempt it.

## Deeper fork debug addendum (2026-05-23, attempted re-push)

Re-attempted the fork using the README-canonical launch (`--dense-as-sparse --disable-overlap-schedule --disable-cuda-graph --context-length 16384 --chunked-prefill-size 8192`) and a long prompt (11796 tokens, above `dense_len=8192`, below context). Instrumented `SimpleGLAAttnBackend.forward` (mamba index OOB checks) and the lightning-attn rope call in `minicpm.py` (positions / cos_sin_cache shape).

Findings:
- Smoking gun is GPU memory corruption originating in the sparse decode kernels. Rope debug shows `positions` going from a clean scalar `pos_max=12` at layer 15 to garbage `-4716043681037107806` at layer 18, with two consecutive sparse layers (16, 17) executing between them.
- All five attempted configs eventually crash during decode:
  - v18 (short prompt + canonical config) → flashinfer `decode.plan` async assert at `indptr.to("cpu")`.
  - v19 (12 k prompt, async CUDA) → `seq_lens.to('cpu')` assert after ~24 s of apparent progress.
  - v20 (12 k prompt + `CUDA_LAUNCH_BLOCKING=1`) → synchronous OOB at `req_to_token[indices] = values` in memory_pool.py:104.
  - v21 (12 k prompt + `max_new_tokens=4`) → same `prepare_for_decode` crash even with a single decode step.
- Decode codepath always calls `get_topk_for_sparse` (minicpm_backend.py:1176) regardless of cache size; there is no genuine dense fallback in decode, so short prompts and long prompts both die — short prompts trip the triton `compress_k_core_new` OOB, long prompts trip the flashinfer paged-KV decoder.
- The fork ships with `Dockerfile.minicpm_sala` pinned to `nvcr.io/nvidia/pytorch:25.01-py3` (CUDA 12.8, torch ~2.6). Our stack is torch 2.9.1 / CUDA 12.9 / flashinfer 0.6.3 / tilelang 0.1.9 — newer across the board. The sparse-decode kernels appear to assume the older paged-KV / flashinfer ABI.

**Conclusion:** every reachable crash site on this stack is a downstream symptom of the same upstream sparse-decode corruption. Without rebuilding in the fork's native Docker env, further patching in our tree is non-productive.

## E6 — FP8 W8A8 sanity quant (2026-05-23)

`experiments/e6_fp8_quality.py` applies torchao `Float8DynamicActivationFloat8WeightConfig` (per-row scales) to every nn.Linear in SALA except `lm_head` (256 layers quantized, peak 36.6 GB on H100), then re-runs the E1 accept-length protocol with the v2/epoch_9 draft. `experiments/e6_bf16_nll_probe.py` and `experiments/e6_greedy_agreement.py` provide bf16 anchors.

| metric                                         | bf16            | FP8 W8A8        |
|---|---|---|
| PPL on 20-prompt probe (max_len=512)           | **14.995**      | **14.978**      |
| mean per-position accept length (1920 pos)     | 0.838 (640 pos) | **0.090**       |
| greedy 64-tok agreement bf16 vs FP8            | n/a             | 368/640 (57.5%) |
| mean position of first bf16/FP8 divergence     | n/a             | 36.6 / 64       |

**Key finding:** FP8 W8A8 preserves the target distribution near-perfectly on raw prompts (ΔPPL ≈ 0.02, well inside noise), but accept length collapses **9.3×** (0.838 → 0.090). The cause is greedy-trajectory drift: bf16 and FP8 targets agree on only 57.5% of tokens, with the first divergence at position ~36 of 64 on average. Once the trajectory forks, the bf16-trained EAGLE3 head is predicting against an FP8 argmax sequence its hidden-state distribution was never trained on.

**Implication for SOAR:** drop-in PTQ + reused draft = unusable spec decoding even when the target retains its distribution. The draft head must be either (a) retrained against FP8-target hidden traces, or (b) co-quantized with a calibration that minimizes argmax churn rather than raw PPL. Documented as motivation for re-training drafts per quant config. CSV: `results/e6_fp8_quality.csv`, log: `logs/e6_fp8.log`, agreement log: `logs/e6_agreement.log`.

## E7 — MXFP4 W4A4 quality (2026-05-24)

`experiments/e7_mxfp4_quality.py` swaps every nn.Linear (except lm_head) for torchao `MXLinear` with `elem_dtype='fp4_e2m1'`, `block_size=32`, E8M0 block scales, emulated gemm (H100 SM 9.0 has no native FP4 tensor cores — Blackwell only). 256 layers swapped, peak 36.3 GB. Same E1-protocol accept-length pass + NLL probe + greedy agreement vs bf16. n_prompts=10 (vs 30 in E6) due to emulated-gemm slowdown.

| metric                                     | bf16            | FP8 W8A8 (E6)   | MXFP4 W4A4 (E7) |
|---|---|---|---|
| PPL on 10-prompt probe (max_len=512)       | 14.995          | 14.978 (Δ -0.0%) | **15.336** (Δ +2.3%) |
| mean per-position accept length            | 0.838 (640 pos) | 0.090           | **0.098** (640 pos) |
| greedy 64-tok agreement vs bf16            | n/a             | 57.5%           | **19.2%** |
| mean position of first bf16/Q divergence   | n/a             | 36.6 / 64       | **~10 / 64** |

**Key finding:** MXFP4 takes a small but measurable PPL hit (+2.3% vs FP8's noise-level zero), and trajectory drift is **3.6× worse** than FP8 — first divergence at token ~10 instead of token ~36. Despite that, accept length is essentially identical to FP8 (0.098 vs 0.090), because once the trajectory forks at all, the bf16-trained draft is out of distribution regardless of how soon the fork happens. The accept-length collapse is a **trajectory-divergence** failure, not a PPL failure — both quants exhibit it as soon as the target's argmax stream begins to differ from the one the draft was trained on.

**Combined E6+E7 takeaway:** any PTQ that shifts even a small fraction of argmax decisions (~40% by token 36 for FP8, ~80% by token 10 for MXFP4) breaks accept length for a reused bf16-trained draft. This is the empirical justification for per-quant-config draft retraining (or for spec-decoding evaluation methodologies that rebase the draft's reference trajectory onto the quantized target's rollout instead of bf16's).

CSV: `results/e7_mxfp4_quality.csv`, log: `logs/e7_mxfp4.log`.

## E8 — NVFP4 FourOverSix per-tensor split (2026-05-24)

`experiments/e8_nvfp4_foursix.py` runs an offline per-tensor sensitivity scan (block-32 MSE for M=4 vs M=6 packings, decision threshold `MSE_4/MSE_6 < (6.25/4.06)² ≈ 2.37` so the marginal bit-cost is paid for in error). The scan over all 256 nn.Linear weights in SALA picks **254 FP4 (E2M1) + 2 FP6 (E2M3)** — only two attention QKV projections benefit from FP6 under this criterion. Per-tensor MX swap then applies the picks via torchao MXLinear (emulated gemm).

| metric                                      | bf16            | MXFP4 (E7)      | **NVFP4-46 (E8)** |
|---|---|---|---|
| FP4 / FP6 split                             | n/a             | 256 / 0         | **254 / 2** |
| PPL on 10-prompt probe                      | 17.96 (anchor)  | 15.34*          | **15.08** |
| mean per-position accept length (640 pos)   | 0.838           | 0.098           | **0.091** |
| greedy 64-tok agreement vs bf16             | n/a             | 19.2%           | **16.9%** |

(*E7 PPL anchor used the 20-prompt set; the 10-prompt set is a harder slice — see E9 below.)

**Takeaway:** the bit-budget-weighted criterion is so permissive that 99.2% of layers hit FP4, so E8 is effectively MXFP4 with two FP6 exceptions in attention QKV. Result: identical plateau (accept 0.091). The "FourOverSix" cost weighting in this form does not move the needle for SALA — to obtain a real FP4/FP6 mix one would need (a) a stricter threshold, or (b) a different sensitivity metric (e.g. forward-output divergence rather than weight-MSE) to surface attention as more sensitive than weight statistics suggest.

CSV: `results/e8_nvfp4_foursix.csv`, picks: `results/e8_picks.json`, log: `logs/e8_nvfp4.log`.

## E9 — drift diagnostics: hidden-only / trajectory-only / MLP-only (2026-05-24)

`experiments/e9_quant_diagnostics.py` runs four parallel ablations on free GPUs to isolate the E6/E7/E8 collapse mechanism. Each loads BF16 + a separately-quantized second copy on the same device, runs the E1 accept-length protocol with mixed sources for the rollout, the target-argmax reference, and the hidden states the draft consumes:

| diag         | rollout | target_argmax | hiddens to draft | quant scope          | accept | NLL/PPL anchor |
|---|---|---|---|---|---|---|
| bf16 (E1)    | bf16    | bf16          | bf16             | none                 | **0.838** (640 pos) | 17.96 / 17.96 |
| **D1**       | bf16    | bf16          | **FP8**          | full                 | **0.087** | 17.96 |
| **D2**       | **FP8** | **FP8**       | bf16             | full                 | **0.102** | 18.19 |
| **D3**       | FP8     | FP8           | FP8              | **MLP-only** (96/256)| **0.092** | 18.31 |
| **D4**       | MXFP4   | MXFP4         | MXFP4            | **MLP-only** (96/256)| **0.098** | 14.87 |
| E6 (full FP8)| FP8     | FP8           | FP8              | full                 | 0.090   | — |
| E7 (full MXFP4)| MXFP4 | MXFP4         | MXFP4            | full                 | 0.098   | — |

**Key finding:** every perturbation lands at **0.087–0.102** — a flat ~10× collapse from bf16's 0.838. Crucially:
- **D1** (hidden-only): with the bf16 trajectory matched and bf16 argmax targets, simply feeding the draft *FP8 hiddens* alone collapses accept length to 0.087. The bf16-trained head cannot read the FP8-target hidden distribution.
- **D2** (trajectory-only): with bf16 hiddens matched, FP8 trajectory alone collapses accept length to 0.102. Trajectory drift independently breaks it.
- **D3 / D4** (MLP-only): quantizing only the 96 MLP projections — leaving attention bf16 — collapses identically. Attention is not the protective component here.

The two failure axes (hidden distribution and argmax trajectory) are **independently sufficient** to cause the collapse; partial-precision mitigations (MLP-only) provide no protection. There is no quant-and-keep-draft path with the v2 bf16-trained head; the draft must be retrained against quantized-target traces (both rebased trajectory **and** quantized hiddens) for spec decoding to recover.

CSVs: `results/e9_d{1,2,3,4}_*.csv`, logs: `logs/e9_d{1,2,3,4}_*.log`. Free-GPU parallel run across CUDA 1/2/4/5.
