"""E5 — GLA tree-verify state-fork correctness.

Tests claim #5: in MiniCPM-SALA's hybrid GLA architecture, tree-verify must
fork the linear-attention (mamba/conv) state per draft-tree branch, otherwise
sibling branches share state and verification produces wrong tokens.

This script runs the same fixed prompt three ways and compares token sequences:
  (a) baseline      : no spec decoding (single-token verify)
  (b) tree+fork     : EAGLE3 tree verify, state-fork ON  (default in current sglang)
  (c) tree-no-fork  : EAGLE3 tree verify, state-fork OFF (requires patch — see below)

For (c) we set SD_HC_NO_GLA_FORK=1 in the server env. The sglang backend at
  python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py
must read this env var and, when set, take a degraded code path that skips the
sibling-aware intermediate cache logic. See state_fork_switch.patch for the
suggested change.

Asserts: tokens(a) == tokens(b). On (c) we expect divergence — that's the
positive signal that the state-fork is what's making (b) correct.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib import server  # noqa: E402

TARGET = "/sgl-workspace/SpecForge/models/MiniCPM-SALA"
EAGLE3 = "/sgl-workspace/SpecForge/models/MiniCPM-SALA-EAGLE3"

PROMPT = (
    "Write a short story (about 200 words) about a robot who discovers it loves gardening. "
    "Begin with: 'On the morning of the seventh day,'"
)
SEED = 0xDEADBEEF
MAX_NEW = 96


def gen(api_base: str) -> list[int]:
    """Send a single deterministic completion request, return generated token ids."""
    r = requests.post(
        f"{api_base}/v1/completions",
        json={
            "model": TARGET,
            "prompt": PROMPT,
            "temperature": 0,
            "max_tokens": MAX_NEW,
            "seed": SEED,
            "logprobs": 0,
        },
        timeout=600,
    )
    r.raise_for_status()
    obj = r.json()
    # Token-level retrieval via /v1/completions response: choices[0]["logprobs"]["tokens"] when logprobs is set
    choices = obj["choices"][0]
    lp = choices.get("logprobs") or {}
    if "tokens" in lp:
        return lp["tokens"]
    # Fallback: tokenize the returned text — adequate for byte-level comparison.
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TARGET, trust_remote_code=True)
    return tok.encode(choices["text"], add_special_tokens=False)


def first_diff(a: list, b: list) -> int:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b)) if len(a) != len(b) else -1


SPEC_EAGLE3 = [
    "--speculative-algorithm", "EAGLE3",
    "--speculative-draft-model-path", EAGLE3,
    "--speculative-num-steps", "4",
    "--speculative-num-draft-tokens", "8",
    "--speculative-eagle-topk", "2",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-no-fork", action="store_true",
                    help="skip the (c) tree-no-fork run (in case patch isn't applied)")
    args = ap.parse_args()

    results: dict[str, list] = {}

    print("\n[e5] === (a) baseline (no spec) ===")
    with server.sglang_server(TARGET, [], log_path=ROOT / "results" / "e5_a.log") as api:
        results["a"] = gen(api)
    print(f"[e5]   {len(results['a'])} tokens")

    print("\n[e5] === (b) tree-verify + state-fork ON ===")
    with server.sglang_server(TARGET, SPEC_EAGLE3,
                               log_path=ROOT / "results" / "e5_b.log") as api:
        results["b"] = gen(api)
    print(f"[e5]   {len(results['b'])} tokens")

    if not args.skip_no_fork:
        print("\n[e5] === (c) tree-verify + state-fork OFF (SD_HC_NO_GLA_FORK=1) ===")
        with server.sglang_server(TARGET, SPEC_EAGLE3,
                                   env={"SD_HC_NO_GLA_FORK": "1"},
                                   log_path=ROOT / "results" / "e5_c.log") as api:
            results["c"] = gen(api)
        print(f"[e5]   {len(results['c'])} tokens")

    out = ROOT / "results" / "e5_sequences.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"[e5] wrote {out}")

    diff_ab = first_diff(results["a"], results["b"])
    print(f"\n[e5] (a) vs (b) diverge at idx: {diff_ab}  ({'IDENTICAL' if diff_ab == -1 else 'DIFFERENT'})")
    if diff_ab != -1:
        print("[e5] FAIL: tree-verify with state-fork produced different output than baseline")
        sys.exit(1)

    if "c" in results:
        diff_ac = first_diff(results["a"], results["c"])
        print(f"[e5] (a) vs (c) diverge at idx: {diff_ac}  "
              f"({'NO DIVERGENCE — patch not active?' if diff_ac == -1 else 'expected divergence'})")
        if diff_ac == -1:
            print("[e5] WARN: state-fork switch may not be wired up; (c) was identical to baseline")

    print("[e5] PASS")


if __name__ == "__main__":
    main()
