#!/bin/bash
# Launch SALA via OpenBMB's official sglang fork (editable-installed in mminf).
# Must preload flashinfer.comm to dodge tilelang libcudart_stub clash.
set -e
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-0}
MODEL=${MODEL:-/sgl-workspace/SpecForge/models/MiniCPM-SALA}
PORT=${PORT:-30005}
python -c "
import torch
import flashinfer.comm  # workaround: prevent tilelang stub from masking real libcudart
import runpy, sys
sys.argv = ['sglang.launch_server',
    '--model', '$MODEL',
    '--trust-remote-code',
    '--disable-radix-cache',
    '--attention-backend', 'minicpm_flashinfer',
    '--chunked-prefill-size', '8192',
    '--max-running-requests', '32',
    '--skip-server-warmup',
    '--port', '$PORT',
    '--context-length', '16384',
    '--disable-cuda-graph',
    '--disable-overlap-schedule',
    '--dense-as-sparse']
runpy.run_module('sglang.launch_server', run_name='__main__')
"
