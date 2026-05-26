"""E8: NVFP4 FourOverSix per-MLP/Attn split — quality vs bf16.

For each nn.Linear weight in SALA, compute two block-32 quantization errors:
  * M=6 (FP4 E2M1 with per-group-32 scale → ~6.25 bits/elem)
  * M=4 (FP4 E2M1 with sub-shared scale every 4 groups → ~4.06 bits/elem)

Pick M=4 when MSE_4 / MSE_6 < (6.25/4.06)^2 ≈ 2.37 (bit-budget weighted).
This is the FourOverSix criterion — accept slightly worse MSE if we save bits.

Apply via torchao MXLinear with mixed elem_dtype per layer:
  * picked = M=4 → fp4_e2m1
  * picked = M=6 → fp6_e2m3
Same E1-protocol accept-length + NLL + greedy-agreement vs bf16.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

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
from e6_fp8_quality import nll_probe  # type: ignore
from e4_fp4_block_scale_profile import block_mse, categorize  # type: ignore

BIT_RATIO_SQ = (6.25 / 4.0625) ** 2  # ≈ 2.37


@torch.no_grad()
def select_per_tensor(model: nn.Module, group: int = 32) -> dict[str, str]:
    """Return {linear_fqn: 'fp4' or 'fp6'} via bit-budget-weighted MSE comparison."""
    picks: dict[str, str] = {}
    cat_counts: dict[str, Counter] = defaultdict(Counter)
    for fqn, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if "lm_head" in fqn:
            continue
        if mod.in_features % group != 0 or (mod.in_features // group) % 4 != 0:
            picks[fqn] = "fp6"  # can't form 4-group blocks → keep higher precision
            cat_counts[categorize(fqn)]["fp6"] += 1
            continue
        w = mod.weight.detach().float().cpu()
        mse6, mse4 = block_mse(w, group)
        # M=4 wins iff mse4 / mse6 < (bits6/bits4)^2
        if mse6 > 0 and mse4 / mse6 < BIT_RATIO_SQ:
            picks[fqn] = "fp4"
        else:
            picks[fqn] = "fp6"
        cat_counts[categorize(fqn)][picks[fqn]] += 1
    # Summary
    print("[e8] FourOverSix per-tensor selection (bit-budget weighted):")
    total_fp4 = sum(c["fp4"] for c in cat_counts.values())
    total = sum(sum(c.values()) for c in cat_counts.values())
    print(f"  OVERALL FP4: {total_fp4}/{total} = {total_fp4/max(total,1)*100:.1f}%")
    for cat in sorted(cat_counts):
        c = cat_counts[cat]
        n = c["fp4"] + c["fp6"]
        print(f"  {cat:>8s}: FP4 {c['fp4']:3d} / FP6 {c['fp6']:3d}  ({c['fp4']/max(n,1)*100:5.1f}% FP4)")
    return picks


def apply_per_tensor_mx(model: nn.Module, picks: dict[str, str]) -> tuple[int, int]:
    """Apply MXLinear per-tensor using `picks`. Returns (n_fp4, n_fp6)."""
    from torchao.prototype.mx_formats.config import MXGemmKernelChoice, MXLinearConfig
    from torchao.prototype.mx_formats.constants import DTYPE_FP4, DTYPE_FP6_E2M3
    from torchao.prototype.mx_formats.mx_linear import MXLinear

    n_fp4 = n_fp6 = 0
    # Walk parent modules so we can do setattr
    name_to_mod = dict(model.named_modules())
    for fqn, pick in picks.items():
        parent_fqn, _, leaf = fqn.rpartition(".")
        parent = name_to_mod[parent_fqn] if parent_fqn else model
        mod = getattr(parent, leaf)
        if not isinstance(mod, nn.Linear):
            continue
        elem = DTYPE_FP4 if pick == "fp4" else DTYPE_FP6_E2M3
        cfg = MXLinearConfig(
            block_size=32, elem_dtype=elem,
            gemm_kernel_choice=MXGemmKernelChoice.EMULATED,
        )
        new = MXLinear.from_float(mod, config=cfg)
        setattr(parent, leaf, new)
        if pick == "fp4":
            n_fp4 += 1
        else:
            n_fp6 += 1
    return n_fp4, n_fp6


@torch.no_grad()
def greedy(model, ids: torch.Tensor, n: int) -> torch.Tensor:
    cur = ids
    for _ in range(n):
        logits = model(input_ids=cur, use_cache=False).logits
        nxt = logits[:, -1:].argmax(dim=-1)
        cur = torch.cat([cur, nxt], dim=1)
    return cur[:, ids.size(1):]


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
    ap.add_argument("--n-prompts", type=int, default=10)
    ap.add_argument("--gen-tokens", type=int, default=64)
    ap.add_argument("--nll-n", type=int, default=10)
    ap.add_argument("--agree-n", type=int, default=10)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parents[1] / "results" / "e8_nvfp4_foursix.csv")
    ap.add_argument("--picks-out", type=Path,
                    default=Path(__file__).resolve().parents[1] / "results" / "e8_picks.json")
    args = ap.parse_args()

    device = "cuda"
    tok = AutoTokenizer.from_pretrained(str(args.target), trust_remote_code=True)

    # --- Collect bf16 greedy anchors first ---
    prompts: list[str] = []
    with args.prompts.open() as f:
        for line in f:
            row = json.loads(line)
            q = row.get("question") or row.get("prompt")
            if q:
                prompts.append(q)
            if len(prompts) >= args.n_prompts:
                break
    print(f"[e8] {len(prompts)} prompts loaded")

    print(f"[e8] loading bf16 target for agreement anchor + sensitivity scan")
    bf16 = AutoModelForCausalLM.from_pretrained(
        str(args.target), trust_remote_code=True, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", device_map=device,
    ).eval()

    # FourOverSix per-tensor selection (uses bf16 weights, CPU side)
    print(f"[e8] running FourOverSix per-tensor sensitivity scan ...")
    t0 = time.time()
    picks = select_per_tensor(bf16, group=32)
    print(f"[e8]   scan done in {time.time()-t0:.1f}s ({len(picks)} linears)")
    args.picks_out.parent.mkdir(parents=True, exist_ok=True)
    args.picks_out.write_text(json.dumps(picks, indent=2))
    print(f"[e8]   wrote per-tensor picks to {args.picks_out}")

    # Capture bf16 rollouts before discarding model
    bf16_rollouts: list[torch.Tensor] = []
    for q in prompts[: args.agree_n]:
        msgs = [{"role": "user", "content": q}]
        chat = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(chat, return_tensors="pt").input_ids.to(device)
        bf16_rollouts.append(greedy(bf16, ids, args.gen_tokens))
    del bf16
    torch.cuda.empty_cache()
    print(f"[e8] bf16 rollouts captured")

    # --- Load + per-tensor MX-quantize ---
    print(f"[e8] loading target for quantization")
    target = AutoModelForCausalLM.from_pretrained(
        str(args.target), trust_remote_code=True, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", device_map=device,
    ).eval()
    print(f"[e8] applying per-tensor MX (FP4 / FP6, block=32, emulated gemm)")
    t0 = time.time()
    n_fp4, n_fp6 = apply_per_tensor_mx(target, picks)
    torch.cuda.synchronize()
    print(f"[e8]   {n_fp4} FP4 + {n_fp6} FP6 linears in {time.time()-t0:.1f}s")
    print(f"[e8]   peak GPU mem: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

    aux_layers = pick_aux_layers(target.config.num_hidden_layers)
    print(f"[e8] aux layers = {aux_layers}")

    print(f"[e8] loading draft {args.draft}")
    draft = load_draft(args.draft, device)
    load_target_embeddings_into_draft(draft, args.target)
    draft = draft.to(device=device, dtype=torch.bfloat16)
    t2d, d2t = load_vocab_mapping(args.vocab_mapping, device)

    # --- Accept-length pass ---
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    total_runs: list[int] = []
    nvfp4_rollouts_for_agree: list[torch.Tensor] = []

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

        if i < args.agree_n:
            nvfp4_rollouts_for_agree.append(cur[:, prompt_len:].clone())

        target_logits, h_cat = run_target_with_hooks(target, cur, aux_layers)
        target_argmax = target_logits.argmax(dim=-1)
        draft_pred = draft_predict_at(draft, h_cat, cur, t2d, d2t)

        runs = measure_accept_runs(draft_pred[:, prompt_len:cur.size(1)],
                                   target_argmax[:, prompt_len:cur.size(1)])
        if not runs:
            continue
        mean_run = sum(runs) / len(runs)
        total_runs.extend(runs)
        rows.append({"prompt_idx": i, "n_positions": len(runs), "mean_accept_len": mean_run})
        print(f"[e8] prompt {i}: mean_accept_len={mean_run:.3f} over {len(runs)} positions")

    overall = sum(total_runs) / len(total_runs) if total_runs else 0.0
    print(f"[e8] OVERALL NVFP4-FourOverSix accept length = {overall:.3f} (n_positions={len(total_runs)})")

    nll_texts = prompts[: args.nll_n]
    nll_mean = nll_probe(target, tok, nll_texts, device)
    ppl = math.exp(nll_mean)
    print(f"[e8] NLL probe (n={len(nll_texts)}, max_len=512): mean_nll={nll_mean:.4f} ppl={ppl:.3f}")

    total_match = 0
    total_n = 0
    first_divs = []
    for i, (a, b) in enumerate(zip(bf16_rollouts, nvfp4_rollouts_for_agree)):
        a, b = a[0], b[0]
        m = (a == b).cpu().tolist()
        nm = sum(m)
        total_match += nm
        total_n += len(m)
        first_div = next((j for j, ok in enumerate(m) if not ok), len(m))
        first_divs.append(first_div)
        print(f"[e8] agree prompt {i}: match={nm}/{len(m)} first_div={first_div}")
    agree_frac = total_match / max(total_n, 1)
    mean_first_div = sum(first_divs) / max(len(first_divs), 1)
    print(f"[e8] AGREE bf16 vs NVFP4-FourOverSix = {total_match}/{total_n} = {agree_frac:.3f}")
    print(f"[e8] mean first divergence = {mean_first_div:.2f} / {args.gen_tokens}")

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
        w.writerow({"prompt_idx": "AGREE_FRAC", "n_positions": total_n,
                    "mean_accept_len": f"{agree_frac:.4f}"})
        w.writerow({"prompt_idx": "MEAN_FIRST_DIV", "n_positions": len(first_divs),
                    "mean_accept_len": f"{mean_first_div:.4f}"})
        w.writerow({"prompt_idx": "N_FP4", "n_positions": n_fp4 + n_fp6,
                    "mean_accept_len": f"{n_fp4}"})
        w.writerow({"prompt_idx": "N_FP6", "n_positions": n_fp4 + n_fp6,
                    "mean_accept_len": f"{n_fp6}"})
    print(f"[e8] wrote {args.out}")


if __name__ == "__main__":
    main()
