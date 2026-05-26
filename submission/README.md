# SOAR submission — Path A (GPTQ W4A16 + FP8 KV)

Minimal Path-A submission for the OpenBMB SOAR / AMD Inference 2025
competition. Seeded from the official `demo-quant.tar.gz` template and
upgraded with FP8 KV cache.

## What's in this tarball

| file | required | role |
|---|---|---|
| `prepare_env.sh` | yes | patches the base image's SALA backends (`bf16 → fp16`) and exports `SGLANG_SERVER_ARGS` |
| `prepare_model.sh` | optional | wrapper that calls `quantize_gptq.py` with `--input / --output` |
| `quantize_gptq.py` | — | RTN W4A16 quantization producing GPTQ-format weights |
| `README.md` | — | this file |

## Status: known limitations

- **Quantization is RTN, not real GPTQ.** No calibration → expect 3–8 pp
  accuracy drop vs bf16. The eval gate is ≥ 97 % of 80, i.e. ≥ 77.6.
  Acceptable for first end-to-end test; **must** be upgraded to real GPTQ
  (via `gptqmodel`) before chasing a competitive score.
- **No speculative decoding.** The trained EAGLE3 draft in
  `sala_eagle3_v2/epoch_9_step_18750` is intentionally **not** bundled
  here — our W7/W8 diagnostics (see `../results/SUMMARY.md` §E9) showed
  the bf16-trained draft collapses to ~0.09 accept length under PTQ.
  Spec-decode submission requires a draft retrained on quant hiddens.
- **`--disable-cuda-graph`** is kept on for first-submission stability;
  remove after the W4 path is verified working.

## Build the tarball

From this directory:

```bash
cd /sgl-workspace/soar_2026_w8_bench/submission
tar czf ../soar_submission_pathA_w4.tar.gz \
    prepare_env.sh prepare_model.sh quantize_gptq.py README.md
ls -lh ../soar_submission_pathA_w4.tar.gz
```

Expected size: a few KB (no model weights bundled — the platform supplies
the original SALA at `/models/MiniCPM-SALA`).

## Local smoke test (recommended before uploading)

The eval platform uses the official `soar-toolkit:latest` image. Reproduce
locally:

```bash
# 1. Pull the base image (one-time, ~mins)
docker pull ghcr.io/openbmb/soar-toolkit:latest
# or in CN: docker pull modelbest-registry.cn-beijing.cr.aliyuncs.com/public/soar-toolkit:latest

# 2. Make sure the model is at host-side ./models/MiniCPM-SALA (see official spec)

# 3. Run the container with this submission mounted
docker run --rm -it --gpus 'device=0' --shm-size=16g \
    -v "$(pwd)":/submission:ro \
    -v "$(pwd)/../sample_models/MiniCPM-SALA":/models/MiniCPM-SALA:ro \
    -v /tmp/processed_model:/processed:rw \
    -p 30000:30000 \
    ghcr.io/openbmb/soar-toolkit:latest bash

# Inside the container:
source /submission/prepare_env.sh
bash   /submission/prepare_model.sh --input /models/MiniCPM-SALA --output /processed
# (then start sglang.launch_server with the processed model — see official spec)
```

## Upgrade path (when ready to chase score)

1. **Replace RTN with calibrated GPTQ** — add `uv pip install gptqmodel` to
   `prepare_env.sh`; rewrite `quantize_gptq.py` to call
   `GPTQModel.from_pretrained(...).quantize(calibration_dataset)`.
   Pull calibration prompts from
   `SOAR-Toolkit/eval_dataset/perf_public_set.jsonl`.
2. **Remove `--disable-cuda-graph`** once W4 stability is confirmed.
3. **Try `--quantization gptq_marlin`** (Marlin kernel) instead of plain
   `gptq` for the GEMM speedup the official spec calls out.
4. **Tune `--chunked-prefill-size`** — start at 8192, sweep larger.
5. **Eventually** retrain the EAGLE3 draft on quant hiddens for Path A+B.

## Validating against the eval gates

After running the server, in another shell:

```bash
# Correctness gate (must score >= 77.6 = 97% of 80)
python3 eval_model.py \
    --api_base http://127.0.0.1:30000 \
    --model_path /processed \
    --data_path perf_public_set.jsonl \
    --concurrency 32

# Speed bench (S1 / S8 / Smax)
bash bench_serving.sh http://127.0.0.1:30000
```

Both scripts: https://github.com/OpenBMB/SOAR-Toolkit
