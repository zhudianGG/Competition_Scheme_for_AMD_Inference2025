"""E6: FP8 W8A8 sanity quant for SALA — quality delta vs bf16 baseline.

Loads SALA bf16, applies torchao Float8DynamicActivationFloat8WeightConfig
(per-row scales) to every nn.Linear except lm_head, and re-runs the E1
offline accept-length eval + a token-level NLL probe on the same perf set.

The comparison answer we want: how much accept-length / NLL does FP8 W8A8
cost vs the bf16 baseline (0.838 over 640 positions on v2/epoch_9).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# Reuse helpers from E1 instead of duplicating them.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from e1_offline_accept_length import (  # type: ignore
    draft_predict_at,
    load_draft,
    load_target_embeddings_into_draft,
    load_vocab_mapping,
    measure_accept_runs,
    pick_aux_layers,
    run_target_with_hooks,
)


def apply_fp8_w8a8(model: nn.Module, skip_names: tuple[str, ...] = ("lm_head",)) -> int:
    from torchao.quantization import (
        Float8DynamicActivationFloat8WeightConfig,
        PerRow,
        quantize_,
    )

    def filter_fn(mod: nn.Module, fqn: str) -> bool:
        if not isinstance(mod, nn.Linear):
            return False
        if any(skip in fqn for skip in skip_names):
            return False
        # torchao Float8 path needs both dims divisible by 16
        if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
            return False
        return True

    targets = [fqn for fqn, m in model.named_modules() if filter_fn(m, fqn)]
    quantize_(model, Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()), filter_fn=filter_fn)
    return len(targets)


@torch.no_grad()
def nll_probe(model, tok, texts: list[str], device: str, max_len: int = 512) -> float:
    """Mean per-token NLL across `texts` (truncated to max_len)."""
    total_nll = 0.0
    total_tok = 0
    for t in texts:
        ids = tok(t, return_tensors="pt", truncation=True, max_length=max_len).input_ids.to(device)
        if ids.size(1) < 2:
            continue
        logits = model(input_ids=ids, use_cache=False).logits.float()
        # NLL of token i given prefix < i: target = ids[1:], logits = logits[:-1]
        logp = torch.log_softmax(logits[:, :-1], dim=-1)
        nll = -logp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1).sum().item()
        total_nll += nll
        total_tok += ids.size(1) - 1
    return total_nll / max(total_tok, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=Path,
                    default=Path("/sgl-workspace/SpecForge/models/MiniCPM-SALA"))
    ap.add_argument("--draft", type=Path,
                    default=Path("/sgl-workspace/SpecForge/outputs/sala_eagle3_v2/epoch_9_step_18750"))
    ap.add_argument("--vocab-mapping", type=Path,
                    default=Path("/sgl-workspace/SpecForge/cache/vocab_mapping/210c72934bd4b0e45cdd3a75d8e01620.pt"))
    ap.add_argument("--prompts", type=Path,
                    default=Path("/sgl-workspace/SpecForge/SOAR-Toolkit/eval_dataset/perf_public_set.jsonl"))
    ap.add_argument("--n-prompts", type=int, default=50)
    ap.add_argument("--gen-tokens", type=int, default=64)
    ap.add_argument("--nll-n", type=int, default=20, help="num prompts to score for NLL probe")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parents[1] / "results" / "e6_fp8_quality.csv")
    args = ap.parse_args()

    device = "cuda"
    tok = AutoTokenizer.from_pretrained(str(args.target), trust_remote_code=True)

    print(f"[e6-fp8] loading target {args.target}")
    target = AutoModelForCausalLM.from_pretrained(
        str(args.target), trust_remote_code=True, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", device_map=device,
    ).eval()

    print(f"[e6-fp8] quantizing target to FP8 W8A8 (torchao Float8DynamicActivationFloat8Weight, PerRow)")
    t0 = time.time()
    n_q = apply_fp8_w8a8(target)
    torch.cuda.synchronize()
    print(f"[e6-fp8]   quantized {n_q} Linear layers in {time.time()-t0:.1f}s")
    print(f"[e6-fp8]   peak GPU mem: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

    aux_layers = pick_aux_layers(target.config.num_hidden_layers)
    print(f"[e6-fp8] aux layers = {aux_layers}")

    print(f"[e6-fp8] loading draft {args.draft}")
    draft = load_draft(args.draft, device)
    load_target_embeddings_into_draft(draft, args.target)
    draft = draft.to(device=device, dtype=torch.bfloat16)

    t2d, d2t = load_vocab_mapping(args.vocab_mapping, device)

    prompts: list[str] = []
    with args.prompts.open() as f:
        for line in f:
            row = json.loads(line)
            q = row.get("question") or row.get("prompt")
            if q:
                prompts.append(q)
            if len(prompts) >= args.n_prompts:
                break
    print(f"[e6-fp8] {len(prompts)} prompts")

    # --- Accept-length pass (matches E1 protocol exactly) ---
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    total_runs: list[int] = []

    for i, q in enumerate(prompts):
        msgs = [{"role": "user", "content": q}]
        chat = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        input_ids = tok(chat, return_tensors="pt").input_ids.to(device)
        prompt_len = input_ids.size(1)
        if prompt_len > 1024:
            continue

        cur = input_ids
        for _ in range(args.gen_tokens):
            with torch.no_grad():
                logits = target(input_ids=cur, use_cache=False).logits
            nxt = logits[:, -1:].argmax(dim=-1)
            cur = torch.cat([cur, nxt], dim=1)

        target_logits, h_cat = run_target_with_hooks(target, cur, aux_layers)
        target_argmax = target_logits.argmax(dim=-1)
        draft_pred = draft_predict_at(draft, h_cat, cur, t2d, d2t)

        start = prompt_len
        end = cur.size(1)
        runs = measure_accept_runs(draft_pred[:, start:end], target_argmax[:, start:end])
        if not runs:
            continue
        mean_run = sum(runs) / len(runs)
        total_runs.extend(runs)
        rows.append({"prompt_idx": i, "n_positions": len(runs), "mean_accept_len": mean_run})
        print(f"[e6-fp8] prompt {i}: mean_accept_len={mean_run:.3f} over {len(runs)} positions")

    overall = sum(total_runs) / len(total_runs) if total_runs else 0.0
    print(f"[e6-fp8] OVERALL FP8 accept length = {overall:.3f} (n_positions={len(total_runs)})")

    # --- NLL probe on the same prompt set ---
    nll_texts = prompts[: args.nll_n]
    nll_mean = nll_probe(target, tok, nll_texts, device)
    ppl = math.exp(nll_mean)
    print(f"[e6-fp8] NLL probe (n={len(nll_texts)}, max_len=512): mean_nll={nll_mean:.4f} ppl={ppl:.3f}")

    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["prompt_idx", "n_positions", "mean_accept_len"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
        w.writerow({"prompt_idx": "OVERALL", "n_positions": len(total_runs),
                    "mean_accept_len": f"{overall:.4f}"})
        w.writerow({"prompt_idx": "NLL_MEAN", "n_positions": len(nll_texts),
                    "mean_accept_len": f"{nll_mean:.4f}"})
        w.writerow({"prompt_idx": "PPL", "n_positions": len(nll_texts),
                    "mean_accept_len": f"{ppl:.4f}"})
    print(f"[e6-fp8] wrote {args.out}")


if __name__ == "__main__":
    main()
