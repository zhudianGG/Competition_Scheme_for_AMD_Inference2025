"""E2: HF-backend hand-rolled spec decoding for MiniCPM-SALA + trained EAGLE3 head.

Two target forwards per round: a "seed" forward on cur_ids to capture aux hidden
states for draft seeding, then a "verify" forward on cur_ids + draft_K to check
acceptance. Reuses prior-round verify outputs where the tokens still match (i.e.,
positions 0..cur_len+accept-1), so the seed forward only needs to compute the
bonus position's hidden state — but since SALA's lightning/sparse caches don't
expose proper crop(), we recompute the full seed forward each round.

Per round:
  1. Seed target forward on cur_ids → cached_h3, cached_logits.
  2. Draft proposes K tokens with full-prefix causal SDPA seeded by cached_h3.
  3. Verify target forward on cur_ids + draft_K → v_logits.
  4. Accept the longest prefix where draft matches target greedy; add bonus token.
  5. Concatenate accepted + bonus into cur for the next round.

Reports mean accept length per round, spec tps, and tps vs vanilla greedy.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
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
    draft.load_state_dict(sd, strict=False)
    draft = draft.to(device=device, dtype=torch.bfloat16).eval()
    return draft


def load_vocab_mapping(path: Path, device: str):
    vm = torch.load(path, map_location=device)
    return vm["t2d"].to(device), vm["d2t"].to(device)


def pick_aux_layers(num_layers: int) -> list[int]:
    return [1, num_layers // 2 - 1, num_layers - 4]


class HiddenCapture:
    """Captures aux-layer hidden states via forward hooks. Reusable across forwards."""

    def __init__(self, model, aux_layers):
        self.captured: dict[int, torch.Tensor] = {}
        self.handles = []
        self.aux_layers = aux_layers
        layers = model.model.layers
        for idx in aux_layers:
            self.handles.append(layers[idx].register_forward_hook(self._hook_for(idx)))

    def _hook_for(self, idx):
        def hook(_module, _input, output):
            self.captured[idx] = output[0] if isinstance(output, tuple) else output
        return hook

    def h_cat(self):
        return torch.cat([self.captured[i] for i in self.aux_layers], dim=-1)

    def close(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def draft_propose(draft, h_prefix_3, prefix_ids, d2t, k: int):
    """Autoregressively propose k draft tokens with standard causal SDPA."""
    h_prefix = draft.project_hidden_states(h_prefix_3)  # (1, T, hidden)
    seq_ids = prefix_ids
    seq_hidden = h_prefix
    draft_ids_t = []
    for _ in range(k):
        inputs_embeds = draft.embed_tokens(seq_ids).to(seq_hidden.dtype)
        T = seq_ids.size(1)
        position_ids = torch.arange(T, device=seq_hidden.device).unsqueeze(0)
        out_h = draft.midlayer(
            input_emb=inputs_embeds,
            hidden_states=seq_hidden,
            cache_hidden=None,
            attention_mask=None,
            position_ids=position_ids,
            past_key_values=None,
            output_attentions=False,
            use_cache=False,
        )
        last_h = out_h[:, -1:, :]
        d_id = draft.compute_logits(last_h).argmax(dim=-1)  # (1, 1) draft-vocab
        t_id = d_id + d2t[d_id]  # target-vocab
        draft_ids_t.append(t_id)
        seq_ids = torch.cat([seq_ids, t_id], dim=1)
        seq_hidden = torch.cat([seq_hidden, last_h], dim=1)
    return torch.cat(draft_ids_t, dim=1)  # (1, K)


@torch.no_grad()
def run_spec(target, draft, tok, prompt, aux_layers, d2t, k, max_new_tokens):
    msgs = [{"role": "user", "content": prompt}]
    chat = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    cur = tok(chat, return_tensors="pt").input_ids.to(target.device)
    prompt_len = cur.size(1)

    cap = HiddenCapture(target, aux_layers)
    try:
        t0 = time.time()
        accept_lens = []
        while cur.size(1) - prompt_len < max_new_tokens:
            cur_len = cur.size(1)
            # Seed forward: capture aux hidden for draft
            _ = target(input_ids=cur, use_cache=False)
            seed_h3 = cap.h_cat()

            # Draft proposes K tokens
            draft_t_seq = draft_propose(draft, seed_h3, cur, d2t, k)  # (1, K)

            # Verify forward: target on cur + drafts
            verify_ids = torch.cat([cur, draft_t_seq], dim=1)
            out = target(input_ids=verify_ids, use_cache=False)
            v_logits = out.logits

            # Predictions for each draft position
            pred_for_draft = v_logits[:, cur_len - 1 : cur_len - 1 + k, :].argmax(dim=-1)
            match = (pred_for_draft == draft_t_seq).cpu()[0].tolist()
            accept = 0
            for m in match:
                if m:
                    accept += 1
                else:
                    break
            bonus_pos = cur_len - 1 + accept
            bonus = v_logits[:, bonus_pos : bonus_pos + 1, :].argmax(dim=-1)
            accepted = draft_t_seq[:, :accept]
            cur = torch.cat([cur, accepted, bonus], dim=1)

            accept_lens.append(accept)
            if (bonus == tok.eos_token_id).any():
                break
        elapsed = time.time() - t0
    finally:
        cap.close()
    return cur.size(1) - prompt_len, accept_lens, elapsed


@torch.no_grad()
def run_greedy(target, tok, prompt, max_new_tokens):
    msgs = [{"role": "user", "content": prompt}]
    chat = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    cur = tok(chat, return_tensors="pt").input_ids.to(target.device)
    prompt_len = cur.size(1)
    t0 = time.time()
    for _ in range(max_new_tokens):
        logits = target(input_ids=cur, use_cache=False).logits
        nxt = logits[:, -1:].argmax(dim=-1)
        cur = torch.cat([cur, nxt], dim=1)
        if (nxt == tok.eos_token_id).any():
            break
    return cur.size(1) - prompt_len, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=Path, default=Path("/sgl-workspace/SpecForge/models/MiniCPM-SALA"))
    ap.add_argument("--draft", type=Path, default=Path("/sgl-workspace/SpecForge/outputs/sala_eagle3_v2/epoch_9_step_18750"))
    ap.add_argument("--vocab-mapping", type=Path, default=Path("/sgl-workspace/SpecForge/cache/vocab_mapping/48548f943af8f668b23ab888755979d8.pt"))
    ap.add_argument("--prompts", type=Path, default=Path("/sgl-workspace/SpecForge/SOAR-Toolkit/eval_dataset/perf_public_set.jsonl"))
    ap.add_argument("--n-prompts", type=int, default=10)
    ap.add_argument("--gen-tokens", type=int, default=64)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--skip-greedy", action="store_true")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "results" / "e2_hf_spec_decode.csv")
    args = ap.parse_args()

    device = "cuda"
    print(f"[e2] loading target {args.target}")
    tok = AutoTokenizer.from_pretrained(str(args.target), trust_remote_code=True)
    target = AutoModelForCausalLM.from_pretrained(
        str(args.target), trust_remote_code=True, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", device_map=device,
    ).eval()
    aux_layers = pick_aux_layers(target.config.num_hidden_layers)
    print(f"[e2] aux layers = {aux_layers}")

    print(f"[e2] loading draft {args.draft}")
    draft = load_draft(args.draft, device)
    draft.load_embedding(str(args.target), embedding_key="model.embed_tokens.weight")
    draft = draft.to(device=device, dtype=torch.bfloat16)

    print(f"[e2] loading vocab mapping {args.vocab_mapping}")
    _, d2t = load_vocab_mapping(args.vocab_mapping, device)

    prompts = []
    with args.prompts.open() as f:
        for line in f:
            row = json.loads(line)
            q = row.get("question") or row.get("prompt")
            if q:
                prompts.append(q)
            if len(prompts) >= args.n_prompts:
                break
    print(f"[e2] {len(prompts)} prompts, K={args.k}, gen_tokens={args.gen_tokens}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    all_accept_lens = []
    spec_total_tokens = spec_total_time = 0.0
    greedy_total_tokens = greedy_total_time = 0.0

    for i, q in enumerate(prompts):
        try:
            n_spec, accept_lens, t_spec = run_spec(target, draft, tok, q, aux_layers, d2t, args.k, args.gen_tokens)
        except Exception as e:
            print(f"[e2] prompt {i} spec failed: {e}")
            continue
        spec_total_tokens += n_spec
        spec_total_time += t_spec
        all_accept_lens.extend(accept_lens)
        mean_l = sum(accept_lens) / len(accept_lens) if accept_lens else 0.0

        if not args.skip_greedy:
            n_greedy, t_greedy = run_greedy(target, tok, q, args.gen_tokens)
            greedy_total_tokens += n_greedy
            greedy_total_time += t_greedy
            speedup = (n_spec / t_spec) / (n_greedy / t_greedy) if t_greedy > 0 else 0.0
            print(f"[e2] prompt {i}: rounds={len(accept_lens)} mean_accept={mean_l:.2f} "
                  f"spec={n_spec}tok/{t_spec:.2f}s greedy={n_greedy}tok/{t_greedy:.2f}s speedup={speedup:.2f}x")
        else:
            print(f"[e2] prompt {i}: rounds={len(accept_lens)} mean_accept={mean_l:.2f} "
                  f"spec={n_spec}tok/{t_spec:.2f}s")

        rows.append({
            "prompt_idx": i, "n_rounds": len(accept_lens),
            "mean_accept_len": f"{mean_l:.3f}",
            "spec_tokens": n_spec, "spec_time_s": f"{t_spec:.3f}",
        })

    overall = sum(all_accept_lens) / len(all_accept_lens) if all_accept_lens else 0.0
    spec_tps = spec_total_tokens / spec_total_time if spec_total_time > 0 else 0.0
    greedy_tps = greedy_total_tokens / greedy_total_time if greedy_total_time > 0 else 0.0
    print(f"\n[e2] OVERALL mean accept length per round = {overall:.3f}  (n_rounds={len(all_accept_lens)}, K={args.k})")
    print(f"[e2] spec throughput: {spec_tps:.2f} tok/s ({spec_total_tokens} tok / {spec_total_time:.2f} s)")
    if greedy_tps > 0:
        print(f"[e2] greedy throughput: {greedy_tps:.2f} tok/s ({greedy_total_tokens} tok / {greedy_total_time:.2f} s)")
        print(f"[e2] wall-clock speedup: {spec_tps / greedy_tps:.2f}x")

    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["prompt_idx", "n_rounds", "mean_accept_len", "spec_tokens", "spec_time_s"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
        w.writerow({"prompt_idx": "OVERALL", "n_rounds": len(all_accept_lens),
                    "mean_accept_len": f"{overall:.4f}",
                    "spec_tokens": spec_total_tokens, "spec_time_s": f"{spec_total_time:.3f}"})
    print(f"[e2] wrote {args.out}")


if __name__ == "__main__":
    main()
