"""bf16 NLL anchor for E6 comparison — same 20-prompt probe as e6_fp8_quality."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e6_fp8_quality import nll_probe  # type: ignore


def main():
    target_path = Path("/sgl-workspace/SpecForge/models/MiniCPM-SALA")
    prompts_path = Path("/sgl-workspace/SpecForge/SOAR-Toolkit/eval_dataset/perf_public_set.jsonl")

    device = "cuda"
    tok = AutoTokenizer.from_pretrained(str(target_path), trust_remote_code=True)
    print(f"[nll-bf16] loading target {target_path}")
    model = AutoModelForCausalLM.from_pretrained(
        str(target_path), trust_remote_code=True, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", device_map=device,
    ).eval()

    prompts: list[str] = []
    with prompts_path.open() as f:
        for line in f:
            row = json.loads(line)
            q = row.get("question") or row.get("prompt")
            if q:
                prompts.append(q)
            if len(prompts) >= 50:
                break
    texts = prompts[:20]
    nll_mean = nll_probe(model, tok, texts, device)
    ppl = math.exp(nll_mean)
    print(f"[nll-bf16] n={len(texts)} mean_nll={nll_mean:.4f} ppl={ppl:.3f}")


if __name__ == "__main__":
    main()
