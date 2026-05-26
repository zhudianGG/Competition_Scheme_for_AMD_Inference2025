"""E9 — quant diagnostics: isolate hidden-drift vs trajectory-drift, test MLP-only quant.

Modes:
  --mode hidden_only   :  rollout + argmax from BF16 model, hiddens from QUANT model
                          (does the draft tolerate quant hiddens if trajectory matches?)
  --mode traj_only     :  rollout + argmax from QUANT model, hiddens from BF16 model
                          (does the draft tolerate quant trajectory if hiddens match?)
  --mode mlp_only      :  quantize only MLP (gate/up/down) projections, keep attention BF16,
                          run full E6 protocol on the resulting hybrid model.

  --quant fp8 | mxfp4
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


def quantize_fp8(model: nn.Module, mlp_only: bool = False) -> int:
    from torchao.quantization import (
        Float8DynamicActivationFloat8WeightConfig,
        PerRow,
        quantize_,
    )
    MLP_TAGS = ("gate_proj", "up_proj", "down_proj")

    def filter_fn(mod, fqn):
        if not isinstance(mod, nn.Linear):
            return False
        if "lm_head" in fqn:
            return False
        if mlp_only and not any(t in fqn for t in MLP_TAGS):
            return False
        if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
            return False
        return True

    targets = [f for f, m in model.named_modules() if filter_fn(m, f)]
    quantize_(model, Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()),
              filter_fn=filter_fn)
    return len(targets)


def quantize_mxfp4(model: nn.Module, mlp_only: bool = False) -> int:
    from torchao.prototype.mx_formats.config import MXGemmKernelChoice, MXLinearConfig
    from torchao.prototype.mx_formats.constants import DTYPE_FP4
    from torchao.prototype.mx_formats.mx_linear import swap_linear_with_mx_linear
    MLP_TAGS = ("gate_proj", "up_proj", "down_proj")

    cfg = MXLinearConfig(block_size=32, elem_dtype=DTYPE_FP4,
                         gemm_kernel_choice=MXGemmKernelChoice.EMULATED)
    targets = []

    def filter_fn(mod, fqn):
        if not isinstance(mod, nn.Linear):
            return False
        if "lm_head" in fqn:
            return False
        if mlp_only and not any(t in fqn for t in MLP_TAGS):
            return False
        if mod.in_features % 32 != 0:
            return False
        targets.append(fqn)
        return True

    swap_linear_with_mx_linear(model, config=cfg, filter_fn=filter_fn)
    return len(targets)


@torch.no_grad()
def greedy(model, ids, n):
    cur = ids
    for _ in range(n):
        logits = model(input_ids=cur, use_cache=False).logits
        nxt = logits[:, -1:].argmax(dim=-1)
        cur = torch.cat([cur, nxt], dim=1)
    return cur


def load_model(path, device):
    return AutoModelForCausalLM.from_pretrained(
        str(path), trust_remote_code=True, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", device_map=device,
    ).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["hidden_only", "traj_only", "mlp_only"], required=True)
    ap.add_argument("--quant", choices=["fp8", "mxfp4"], required=True)
    ap.add_argument("--tag", required=True, help="output suffix, e.g. 'd1_fp8'")
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
    ap.add_argument("--out-dir", type=Path,
                    default=Path("/sgl-workspace/soar_2026_w8_bench/results"))
    args = ap.parse_args()

    device = "cuda"
    tok = AutoTokenizer.from_pretrained(str(args.target), trust_remote_code=True)

    prompts: list[str] = []
    with args.prompts.open() as f:
        for line in f:
            row = json.loads(line)
            q = row.get("question") or row.get("prompt")
            if q:
                prompts.append(q)
            if len(prompts) >= args.n_prompts:
                break

    quant_fn = quantize_fp8 if args.quant == "fp8" else quantize_mxfp4

    if args.mode == "mlp_only":
        # Single model: quantize only MLP projections
        print(f"[e9-{args.tag}] loading + MLP-only {args.quant} quant")
        m = load_model(args.target, device)
        n_q = quant_fn(m, mlp_only=True)
        print(f"[e9-{args.tag}]   quantized {n_q} MLP layers; peak {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
        target_for_traj = m
        target_for_hidden = m
        target_for_argmax = m
    else:
        # Need both BF16 and quant. Load BF16 first.
        print(f"[e9-{args.tag}] loading BF16 anchor")
        bf16 = load_model(args.target, device)
        print(f"[e9-{args.tag}] loading + {args.quant} quantizing second copy")
        # Load second copy on same device → may OOM. Move bf16 to CPU first.
        bf16_cpu = bf16  # keep ref
        torch.cuda.empty_cache()
        qm = load_model(args.target, device)
        n_q = quant_fn(qm, mlp_only=False)
        print(f"[e9-{args.tag}]   quantized {n_q} layers; peak {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
        # Push bf16 to GPU now (both should fit on 80 GB H100: ~14 GB each)
        # Already on GPU. Continue.
        if args.mode == "hidden_only":
            # rollout, argmax from bf16; hiddens from quant
            target_for_traj = bf16
            target_for_argmax = bf16
            target_for_hidden = qm
        else:  # traj_only
            # rollout, argmax from quant; hiddens from bf16
            target_for_traj = qm
            target_for_argmax = qm
            target_for_hidden = bf16

    aux_layers = pick_aux_layers(target_for_traj.config.num_hidden_layers)
    print(f"[e9-{args.tag}] aux layers = {aux_layers}")

    print(f"[e9-{args.tag}] loading draft {args.draft}")
    draft = load_draft(args.draft, device)
    load_target_embeddings_into_draft(draft, args.target)
    draft = draft.to(device=device, dtype=torch.bfloat16)
    t2d, d2t = load_vocab_mapping(args.vocab_mapping, device)

    rows = []
    total_runs: list[int] = []

    for i, q in enumerate(prompts):
        msgs = [{"role": "user", "content": q}]
        chat = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        input_ids = tok(chat, return_tensors="pt").input_ids.to(device)
        prompt_len = input_ids.size(1)
        if prompt_len > 1024:
            continue

        cur = greedy(target_for_traj, input_ids, args.gen_tokens)

        # target_argmax + hiddens: from possibly DIFFERENT models on the same `cur`
        if target_for_hidden is target_for_argmax:
            tlogits, hcat = run_target_with_hooks(target_for_argmax, cur, aux_layers)
            targmax = tlogits.argmax(dim=-1)
        else:
            tlogits, _ = run_target_with_hooks(target_for_argmax, cur, aux_layers)
            targmax = tlogits.argmax(dim=-1)
            _, hcat = run_target_with_hooks(target_for_hidden, cur, aux_layers)

        draft_pred = draft_predict_at(draft, hcat, cur, t2d, d2t)
        runs = measure_accept_runs(draft_pred[:, prompt_len:cur.size(1)],
                                   targmax[:, prompt_len:cur.size(1)])
        if not runs:
            continue
        mean_run = sum(runs) / len(runs)
        total_runs.extend(runs)
        rows.append({"prompt_idx": i, "n_positions": len(runs), "mean_accept_len": mean_run})
        print(f"[e9-{args.tag}] prompt {i}: mean_accept_len={mean_run:.3f} over {len(runs)} positions")

    overall = sum(total_runs) / len(total_runs) if total_runs else 0.0
    print(f"[e9-{args.tag}] OVERALL accept length = {overall:.3f} (n_positions={len(total_runs)})")

    nll_texts = prompts[: args.nll_n]
    nll_target = target_for_argmax  # NLL of the argmax-producing model
    nll_mean = nll_probe(nll_target, tok, nll_texts, device)
    ppl = math.exp(nll_mean)
    print(f"[e9-{args.tag}] NLL probe (n={len(nll_texts)}): nll={nll_mean:.4f} ppl={ppl:.3f}")

    out_path = args.out_dir / f"e9_{args.tag}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
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
    print(f"[e9-{args.tag}] wrote {out_path}")


if __name__ == "__main__":
    main()
