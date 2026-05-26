"""Greedy-token agreement between bf16 and FP8 W8A8 SALA on identical prompts.

For each prompt, roll bf16 target greedily for 64 tokens AND roll FP8 target
greedily for 64 tokens from the same prefix. Report % of positions where the
two trajectories match (position-by-position, no rebase after divergence — once
they fork, downstream tokens carry the fork forward, which is the realistic
behavior the draft head would see).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e6_fp8_quality import apply_fp8_w8a8  # type: ignore


@torch.no_grad()
def greedy(model, ids: torch.Tensor, n: int) -> torch.Tensor:
    cur = ids
    for _ in range(n):
        logits = model(input_ids=cur, use_cache=False).logits
        nxt = logits[:, -1:].argmax(dim=-1)
        cur = torch.cat([cur, nxt], dim=1)
    return cur[:, ids.size(1):]


def main():
    target_path = Path("/sgl-workspace/SpecForge/models/MiniCPM-SALA")
    prompts_path = Path("/sgl-workspace/SpecForge/SOAR-Toolkit/eval_dataset/perf_public_set.jsonl")
    gen_n = 64
    n_prompts = 10

    device = "cuda"
    tok = AutoTokenizer.from_pretrained(str(target_path), trust_remote_code=True)

    prompts: list[str] = []
    with prompts_path.open() as f:
        for line in f:
            row = json.loads(line)
            q = row.get("question") or row.get("prompt")
            if q:
                prompts.append(q)
            if len(prompts) >= n_prompts:
                break

    print("[agree] loading bf16 target")
    bf16 = AutoModelForCausalLM.from_pretrained(
        str(target_path), trust_remote_code=True, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", device_map=device,
    ).eval()
    bf16_outs = []
    for q in prompts:
        msgs = [{"role": "user", "content": q}]
        chat = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(chat, return_tensors="pt").input_ids.to(device)
        bf16_outs.append(greedy(bf16, ids, gen_n))
    del bf16
    torch.cuda.empty_cache()

    print("[agree] loading + FP8-quantizing target")
    fp8 = AutoModelForCausalLM.from_pretrained(
        str(target_path), trust_remote_code=True, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", device_map=device,
    ).eval()
    apply_fp8_w8a8(fp8)

    fp8_outs = []
    for q in prompts:
        msgs = [{"role": "user", "content": q}]
        chat = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(chat, return_tensors="pt").input_ids.to(device)
        fp8_outs.append(greedy(fp8, ids, gen_n))

    total_match = 0
    total_n = 0
    first_divs = []
    for i, (a, b) in enumerate(zip(bf16_outs, fp8_outs)):
        a, b = a[0], b[0]
        m = (a == b).cpu().tolist()
        nm = sum(m)
        total_match += nm
        total_n += len(m)
        # First divergence position
        first_div = next((j for j, ok in enumerate(m) if not ok), len(m))
        first_divs.append(first_div)
        print(f"[agree] prompt {i}: match={nm}/{len(m)} ({nm/len(m):.2f}), first_div={first_div}")

    print(f"[agree] OVERALL match = {total_match}/{total_n} = {total_match/total_n:.3f}")
    print(f"[agree] mean first_divergence = {sum(first_divs)/len(first_divs):.2f} / {gen_n}")


if __name__ == "__main__":
    main()
