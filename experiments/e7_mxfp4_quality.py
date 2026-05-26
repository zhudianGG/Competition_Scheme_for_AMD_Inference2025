"""E7: MXFP4 (W4A4 with E8M0 block-32 scales) quality vs bf16.

Uses torchao prototype MX linear swap with elem_dtype='fp4_e2m1', block_size=32,
emulated gemm (H100 SM 9.0 has no FP4 tensor cores — Blackwell only). Quality
measurement only; emulated path is slow.

Same protocol as E6: per-position accept length on the E1 protocol + NLL probe +
greedy-trajectory agreement against bf16 target.
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


def apply_mxfp4(model: nn.Module, skip_names: tuple[str, ...] = ("lm_head",)) -> int:
    from torchao.prototype.mx_formats.config import MXGemmKernelChoice, MXLinearConfig
    from torchao.prototype.mx_formats.constants import DTYPE_FP4
    from torchao.prototype.mx_formats.mx_linear import swap_linear_with_mx_linear

    cfg = MXLinearConfig(
        block_size=32,
        elem_dtype=DTYPE_FP4,
        gemm_kernel_choice=MXGemmKernelChoice.EMULATED,
    )

    targets = []

    def filter_fn(mod: nn.Module, fqn: str) -> bool:
        if not isinstance(mod, nn.Linear):
            return False
        if any(skip in fqn for skip in skip_names):
            return False
        # MXFP4 block size = 32 → in_features must be multiple of 32
        if mod.in_features % 32 != 0:
            return False
        targets.append(fqn)
        return True

    swap_linear_with_mx_linear(model, config=cfg, filter_fn=filter_fn)
    return len(targets)


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
    ap.add_argument("--n-prompts", type=int, default=30,
                    help="cap lower than E6 (50) because emulated MXFP4 is ~10x slower")
    ap.add_argument("--gen-tokens", type=int, default=64)
    ap.add_argument("--nll-n", type=int, default=20)
    ap.add_argument("--agree-n", type=int, default=10,
                    help="greedy-agreement prompts (loads bf16 separately, costs a forward each)")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parents[1] / "results" / "e7_mxfp4_quality.csv")
    args = ap.parse_args()

    device = "cuda"
    tok = AutoTokenizer.from_pretrained(str(args.target), trust_remote_code=True)

    # --- Greedy agreement pass: collect bf16 rollouts FIRST, then drop bf16 weights ---
    prompts: list[str] = []
    with args.prompts.open() as f:
        for line in f:
            row = json.loads(line)
            q = row.get("question") or row.get("prompt")
            if q:
                prompts.append(q)
            if len(prompts) >= args.n_prompts:
                break
    print(f"[e7-mxfp4] {len(prompts)} prompts loaded")

    print(f"[e7-mxfp4] loading bf16 target for agreement anchor")
    bf16 = AutoModelForCausalLM.from_pretrained(
        str(args.target), trust_remote_code=True, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", device_map=device,
    ).eval()
    bf16_rollouts: list[torch.Tensor] = []
    for q in prompts[: args.agree_n]:
        msgs = [{"role": "user", "content": q}]
        chat = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(chat, return_tensors="pt").input_ids.to(device)
        bf16_rollouts.append(greedy(bf16, ids, args.gen_tokens))
    del bf16
    torch.cuda.empty_cache()
    print(f"[e7-mxfp4] bf16 rollouts captured ({args.agree_n} prompts)")

    # --- Load + MXFP4-quantize target ---
    print(f"[e7-mxfp4] loading target {args.target}")
    target = AutoModelForCausalLM.from_pretrained(
        str(args.target), trust_remote_code=True, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", device_map=device,
    ).eval()
    print(f"[e7-mxfp4] swapping Linear → MX (fp4_e2m1, block=32, emulated gemm)")
    t0 = time.time()
    n_q = apply_mxfp4(target)
    torch.cuda.synchronize()
    print(f"[e7-mxfp4]   swapped {n_q} Linear layers in {time.time()-t0:.1f}s")
    print(f"[e7-mxfp4]   peak GPU mem: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

    aux_layers = pick_aux_layers(target.config.num_hidden_layers)
    print(f"[e7-mxfp4] aux layers = {aux_layers}")

    print(f"[e7-mxfp4] loading draft {args.draft}")
    draft = load_draft(args.draft, device)
    load_target_embeddings_into_draft(draft, args.target)
    draft = draft.to(device=device, dtype=torch.bfloat16)
    t2d, d2t = load_vocab_mapping(args.vocab_mapping, device)

    # --- Accept-length pass ---
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    total_runs: list[int] = []
    fp4_rollouts_for_agree: list[torch.Tensor] = []

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
            fp4_rollouts_for_agree.append(cur[:, prompt_len:].clone())

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
        print(f"[e7-mxfp4] prompt {i}: mean_accept_len={mean_run:.3f} over {len(runs)} positions")

    overall = sum(total_runs) / len(total_runs) if total_runs else 0.0
    print(f"[e7-mxfp4] OVERALL MXFP4 accept length = {overall:.3f} (n_positions={len(total_runs)})")

    # --- NLL probe ---
    nll_texts = prompts[: args.nll_n]
    nll_mean = nll_probe(target, tok, nll_texts, device)
    ppl = math.exp(nll_mean)
    print(f"[e7-mxfp4] NLL probe (n={len(nll_texts)}, max_len=512): mean_nll={nll_mean:.4f} ppl={ppl:.3f}")

    # --- Greedy agreement vs bf16 ---
    total_match = 0
    total_n = 0
    first_divs = []
    for i, (a, b) in enumerate(zip(bf16_rollouts, fp4_rollouts_for_agree)):
        a, b = a[0], b[0]
        m = (a == b).cpu().tolist()
        nm = sum(m)
        total_match += nm
        total_n += len(m)
        first_div = next((j for j, ok in enumerate(m) if not ok), len(m))
        first_divs.append(first_div)
        print(f"[e7-mxfp4] agree prompt {i}: match={nm}/{len(m)} first_div={first_div}")
    agree_frac = total_match / max(total_n, 1)
    mean_first_div = sum(first_divs) / max(len(first_divs), 1)
    print(f"[e7-mxfp4] AGREE bf16 vs MXFP4 = {total_match}/{total_n} = {agree_frac:.3f}")
    print(f"[e7-mxfp4] mean first divergence = {mean_first_div:.2f} / {args.gen_tokens}")

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
    print(f"[e7-mxfp4] wrote {args.out}")


if __name__ == "__main__":
    main()
