# MiniCPM-SALA Docker Test Bundle

Reproduce the OpenBMB sglang fork (`minicpm_sala` branch) inside its native
container `nvcr.io/nvidia/pytorch:25.01-py3` (CUDA 12.8, torch ~2.6) and
confirm whether the decode crashes we hit on our newer host stack
(torch 2.9.1 / CUDA 12.9 / flashinfer 0.6.3 / tilelang 0.1.9) disappear.

## Contents

```
sala_docker_bundle/
├── Dockerfile.minicpm_sala       # OpenBMB's pinned build (CUDA 12.8, py3.12)
├── launch_sala_sglang.sh         # README-canonical launch (dense-as-sparse, ctx=16384)
├── short_payload.json            # short prompt + 32-tok gen
├── payloads/
│   ├── long_prompt_11796tok.json # ~11.8k-token prompt + 64-tok gen (sparse regime)
│   ├── long_prompt_short_gen.json# ~11.8k-token prompt +  4-tok gen (minimal smoke)
│   └── gen_long_prompt.py        # regenerate the long prompt if needed
├── bench_client.py               # smoke test client hitting /generate
└── README.md                     # this file
```

## What we observed on the host (to verify in the container)

| variant                                            | host (torch 2.9.1 / CUDA 12.9) | expected in container |
|---|---|---|
| short prompt, default config, --disable-cuda-graph | works (~9.2 tok/s)             | works                 |
| 11.8 k prompt, 64-tok decode, canonical launch     | **crashes** (`prepare_for_decode` / `req_to_token` OOB after prefill) | should work |
| 11.8 k prompt, 4-tok decode, canonical launch      | **crashes** (same site)        | should work           |
| CUDA graph capture                                 | **crashes** (`compressed_attention` topk OOB) | should work |

Cascading CUDA memory corruption pinned to the sparse-decode kernels — likely a
flashinfer / tilelang ABI mismatch vs the fork's pinned versions.

## Steps

### 1. Build the image (pulls the fork + submodules + tilelang + fla; ~25 min)

```bash
cd sala_docker_bundle
docker build -f Dockerfile.minicpm_sala -t minicpm-sala:bench .
```

The Dockerfile clones `OpenBMB/sglang@minicpm_sala`, clones / builds
`infllmv2_cuda_impl` and `sparse_kernel` from the `3rdparty/` submodules, then
pip-installs `tilelang` and `flash-linear-attention`.

### 2. Run the container with the MiniCPM-SALA model mounted

The model used on the host is at `/sgl-workspace/SpecForge/models/MiniCPM-SALA`
(HF sha `9180fe1d`, lastMod 2026-05-07). Mount it under `/models`:

```bash
docker run --rm -it --gpus all --shm-size=16g \
    -v /sgl-workspace/SpecForge/models/MiniCPM-SALA:/models/MiniCPM-SALA:ro \
    -v "$(pwd)":/bundle \
    -p 30005:30005 \
    minicpm-sala:bench
```

Inside the container the fork lives at `/workspace/sglang` and is already
editable-installed.

### 3. Launch the server

```bash
cd /bundle
MODEL=/models/MiniCPM-SALA bash launch_sala_sglang.sh
```

`launch_sala_sglang.sh` runs `sglang.launch_server` with the README-canonical
combo: `--dense-as-sparse --disable-overlap-schedule --disable-cuda-graph
--context-length 16384 --chunked-prefill-size 8192 --attention-backend
minicpm_flashinfer --port 30005`. Look for `The server is fired up and ready to roll!`.

### 4. Run the smoke tests

In a second shell inside the container (or on the host, since the port is
forwarded):

```bash
python /bundle/bench_client.py            # short + long4 + long64
python /bundle/bench_client.py --skip-long   # just the short probe
```

A PASS on `long_4tok` is the minimum signal that the sparse-decode path works
in this container; a PASS on `long_64tok` confirms the full long-prompt decode
regime. If both PASS, the host failures were stack-ABI, not algorithmic.

### 5. Optional: enable CUDA graphs

Once decode is stable, remove `--disable-cuda-graph` from the launch script
and rerun step 3. On the host this trips a separate `compressed_attention`
topk OOB during capture (`selected index k out of range`) — check whether it
reproduces.

## Notes

* The pinned base image is `nvcr.io/nvidia/pytorch:25.01-py3`; do not bump
  CUDA / torch unless you want to re-create the same wall we hit.
* `--skip-server-warmup` is on by default in the launch script to avoid the
  synthetic-batch CUDA graph capture path (this is the trade-off that
  forces `--disable-cuda-graph` together with it on the host).
* If the build fails on the `flash-linear-attention` install, pin it to
  `0.4.2` (`pip install flash-linear-attention==0.4.2`). The fork's
  `SimpleGLAAttnBackend` was verified against that release.
* All host findings are documented in `/sgl-workspace/soar_2026_w8_bench/results/SUMMARY.md`
  under "sglang integration attempt" and "Deeper fork debug addendum".
