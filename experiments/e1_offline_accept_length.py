"""E1-offline: accept-length measurement for SALA + trained EAGLE3 head.

We cannot run sglang spec decoding on SALA (not in registry), so this measures
the EAGLE3 head's per-position top-1 match rate against the target's argmax,
using teacher-forced target hidden states. The reported metric is the average
'longest accepted run' starting at each position over a target greedy rollout.

Run target on (prompt + N target-generated tokens), capture aux hidden states
at the same 3 layers the head was trained on. For each generation position t,
project (h_t low/mid/high) through the draft head, take top-1 in the draft
vocab, map back to target vocab via d2t, compare to target_argmax[t+1].
Accept length at t = number of consecutive matches starting at t.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_draft(ckpt_dir: Path, device: str):
    sys.path.insert(0, str(Path("/sgl-workspace/SpecForge")))
    from specforge.modeling.draft.llama3_eagle import LlamaForCausalLMEagle3
    from transformers.models.llama.configuration_llama import LlamaConfig

    cfg = LlamaConfig.from_pretrained(ckpt_dir)
    cfg.draft_vocab_size = json.load(open(ckpt_dir / "config.json"))["draft_vocab_size"]
    draft = LlamaForCausalLMEagle3(cfg, attention_backend="sdpa")
    sd = {}
    from safetensors import safe_open
    with safe_open(str(ckpt_dir / "model.safetensors"), framework="pt") as f:
        for k in f.keys():
            sd[k] = f.get_tensor(k)
    missing, unexpected = draft.load_state_dict(sd, strict=False)
    if missing:
        print(f"[draft] missing keys: {missing[:5]}{'...' if len(missing)>5 else ''}")
    if unexpected:
        print(f"[draft] unexpected keys: {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")
    draft = draft.to(device=device, dtype=torch.bfloat16).eval()
    return draft


def load_target_embeddings_into_draft(draft, target_path: Path):
    draft.load_embedding(str(target_path), embedding_key="model.embed_tokens.weight")
    print(f"[draft] loaded embed_tokens from {target_path}")


def load_vocab_mapping(path: Path, device: str):
    vm = torch.load(path, map_location=device)
    return vm["t2d"].to(device), vm["d2t"].to(device)


def pick_aux_layers(num_layers: int) -> list[int]:
    return [1, num_layers // 2 - 1, num_layers - 4]


@torch.no_grad()
def run_target_with_hooks(model, input_ids, aux_layers):
    captured: dict[int, torch.Tensor] = {}
    handles = []
    layers = model.model.layers

    def hook_for(idx):
        def hook(_module, _input, output):
            captured[idx] = output[0] if isinstance(output, tuple) else output
        return hook

    for idx in aux_layers:
        handles.append(layers[idx].register_forward_hook(hook_for(idx)))
    try:
        out = model(input_ids=input_ids, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    h_cat = torch.cat([captured[i] for i in aux_layers], dim=-1)
    return out.logits, h_cat


@torch.no_grad()
def draft_predict_at(draft, h_cat, prev_input_ids, t2d, d2t):
    # h_cat: (1, T, 3*hidden). prev_input_ids: (1, T) — the input that produced h_cat.
    inputs_embeds = draft.embed_tokens(prev_input_ids)
    hidden = draft(hidden_states=h_cat, inputs_embeds=inputs_embeds, ttt_length=1)
    logits = draft.lm_head(hidden)  # (1, T, draft_vocab)
    draft_top1 = logits.argmax(dim=-1)  # (1, T)
    # Map draft ids to target ids. d2t is an offset table: target_id = draft_id + d2t[draft_id].
    target_pred = draft_top1 + d2t[draft_top1]
    return target_pred  # (1, T) — prediction for position t+1 given hidden at t


def measure_accept_runs(draft_pred: torch.Tensor, target_argmax: torch.Tensor) -> list[int]:
    # target_logits[t] is the logit for position t+1 (the token AFTER seeing context up to t).
    # So target_argmax[t] = target's greedy choice for the token that comes next.
    # draft_pred[t] = draft's prediction for the same "next token" given h_t.
    # Both refer to the same position → compare element-wise at index t.
    n = draft_pred.size(1)
    match = (draft_pred[0, :n] == target_argmax[0, :n]).cpu().tolist()
    runs = []
    for s in range(n):
        k = 0
        while s + k < len(match) and match[s + k]:
            k += 1
        runs.append(k)
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=Path,
                    default=Path("/sgl-workspace/SpecForge/models/MiniCPM-SALA"))
    ap.add_argument("--draft", type=Path,
                    default=Path("/sgl-workspace/SpecForge/outputs/sala_eagle3_v1/epoch_0_step_1875"))
    ap.add_argument("--vocab-mapping", type=Path,
                    default=Path("/sgl-workspace/SpecForge/cache/vocab_mapping/210c72934bd4b0e45cdd3a75d8e01620.pt"))
    ap.add_argument("--prompts", type=Path,
                    default=Path("/sgl-workspace/SpecForge/SOAR-Toolkit/eval_dataset/perf_public_set.jsonl"))
    ap.add_argument("--n-prompts", type=int, default=50)
    ap.add_argument("--gen-tokens", type=int, default=64,
                    help="how many target-greedy tokens to roll forward per prompt")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parents[1] / "results" / "e1_offline_accept.csv")
    args = ap.parse_args()

    device = "cuda"
    tok = AutoTokenizer.from_pretrained(str(args.target), trust_remote_code=True)
    print(f"[e1-offline] loading target {args.target}")
    target = AutoModelForCausalLM.from_pretrained(
        str(args.target), trust_remote_code=True, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", device_map=device,
    ).eval()
    aux_layers = pick_aux_layers(target.config.num_hidden_layers)
    print(f"[e1-offline] aux layers = {aux_layers}")

    print(f"[e1-offline] loading draft {args.draft}")
    draft = load_draft(args.draft, device)
    load_target_embeddings_into_draft(draft, args.target)
    draft = draft.to(device=device, dtype=torch.bfloat16)

    print(f"[e1-offline] loading vocab mapping {args.vocab_mapping}")
    t2d, d2t = load_vocab_mapping(args.vocab_mapping, device)

    prompts = []
    with args.prompts.open() as f:
        for line in f:
            row = json.loads(line)
            q = row.get("question") or row.get("prompt")
            if q:
                prompts.append(q)
            if len(prompts) >= args.n_prompts:
                break
    print(f"[e1-offline] {len(prompts)} prompts")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    total_runs: list[int] = []

    for i, q in enumerate(prompts):
        msgs = [{"role": "user", "content": q}]
        chat = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        input_ids = tok(chat, return_tensors="pt").input_ids.to(device)
        prompt_len = input_ids.size(1)
        if prompt_len > 1024:
            continue  # keep this offline pass cheap

        # Greedy rollout on target for N tokens
        cur = input_ids
        for _ in range(args.gen_tokens):
            with torch.no_grad():
                logits = target(input_ids=cur, use_cache=False).logits
            nxt = logits[:, -1:].argmax(dim=-1)
            cur = torch.cat([cur, nxt], dim=1)

        # One full pass with hooks over the rolled context
        target_logits, h_cat = run_target_with_hooks(target, cur, aux_layers)
        target_argmax = target_logits.argmax(dim=-1)
        draft_pred = draft_predict_at(draft, h_cat, cur, t2d, d2t)

        # Restrict to the generated region (skip the prompt — we want spec decoding regime)
        start = prompt_len
        end = cur.size(1)
        runs = measure_accept_runs(draft_pred[:, start:end], target_argmax[:, start:end])
        if not runs:
            continue
        mean_run = sum(runs) / len(runs)
        total_runs.extend(runs)
        rows.append({"prompt_idx": i, "n_positions": len(runs), "mean_accept_len": mean_run})
        print(f"[e1-offline] prompt {i}: mean_accept_len={mean_run:.2f} over {len(runs)} positions")

    overall = sum(total_runs) / len(total_runs) if total_runs else 0.0
    print(f"[e1-offline] OVERALL mean accept length = {overall:.3f} (n_positions={len(total_runs)})")

    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["prompt_idx", "n_positions", "mean_accept_len"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
        w.writerow({"prompt_idx": "OVERALL", "n_positions": len(total_runs),
                    "mean_accept_len": f"{overall:.4f}"})
    print(f"[e1-offline] wrote {args.out}")


if __name__ == "__main__":
    main()
