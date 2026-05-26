"""E4 — FourOverSix block-scale profile.

Tests claim #4: in MiniCPM-SALA, ~40-43% of blocks pick M=4 (vs M=6) under the
FourOverSix MSE comparison, with MLP weights skewing toward M=4 more than
attention QKV. This is fully offline — no server, no GPU strictly required.

Algorithm per block (per output channel groups of 32 elements):
  - M=6: classic E4M3 scale per group of 32 (1 byte scale)
  - M=4: 4 groups share a sub-scale; pick the variant with lower MSE.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open


def categorize(name: str) -> str:
    n = name.lower()
    if "q_proj" in n or "k_proj" in n or "v_proj" in n or "qkv" in n:
        return "qkv"
    if "o_proj" in n:
        return "o_proj"
    if "gate_proj" in n or "gate" in n:
        return "gate"
    if "up_proj" in n or "up" == n.split(".")[-2] if "." in n else False:
        return "up"
    if "down_proj" in n or "down" in n:
        return "down"
    if "lm_head" in n or "embed" in n:
        return "embed"
    return "other"


def fp4_e2m1_quant(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Per-group FP4 (E2M1) quantization: representable values are {0, 0.5, 1, 1.5, 2, 3, 4, 6}*sign."""
    REPR = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=x.device)
    s = scale.clamp_min(1e-12)
    xn = x / s
    sign = torch.sign(xn)
    mag = xn.abs()
    # Find closest of REPR.
    diffs = (mag.unsqueeze(-1) - REPR).abs()
    idx = diffs.argmin(dim=-1)
    q = REPR[idx] * sign
    return q * s


def block_mse(weight: torch.Tensor, group: int) -> tuple[float, float]:
    """Returns (mse_m6, mse_m4) for the per-row 32-wide grouping."""
    rows, cols = weight.shape
    assert cols % group == 0, f"cols {cols} not divisible by group {group}"
    w = weight.reshape(rows, cols // group, group)
    absmax = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)

    # M=6: each group has its own scale.
    scale6 = absmax / 6.0
    qw6 = fp4_e2m1_quant(w, scale6)
    mse6 = (qw6 - w).pow(2).mean().item()

    # M=4: 4 groups share a sub-scale. Aggregate every 4 groups.
    n_groups = cols // group
    if n_groups % 4 == 0:
        w4 = w.reshape(rows, n_groups // 4, 4, group)
        absmax4 = w4.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
        scale4 = absmax4 / 4.0
        qw4 = fp4_e2m1_quant(w4, scale4)
        mse4 = (qw4 - w4).pow(2).mean().item()
    else:
        mse4 = float("inf")
    return mse6, mse4


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", type=Path,
                    default=Path("/sgl-workspace/SpecForge/models/MiniCPM-SALA"))
    ap.add_argument("--group", type=int, default=32)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parents[1] / "results" / "e4_block_scale.csv")
    ap.add_argument("--limit", type=int, default=None, help="cap number of tensors (for debug)")
    args = ap.parse_args()

    shards = sorted(args.model_path.glob("model-*.safetensors"))
    if not shards:
        sys.exit(f"no shards under {args.model_path}")
    print(f"[e4] scanning {len(shards)} shards under {args.model_path}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    seen = 0
    for shard in shards:
        with safe_open(str(shard), framework="pt") as f:
            for name in f.keys():
                if not name.endswith(".weight"):
                    continue
                # Skip norms / 1D / embeddings (we still profile lm_head if 2D).
                if any(tag in name for tag in ("layernorm", "_norm.", "rotary")):
                    continue
                w = f.get_tensor(name).float()
                if w.dim() != 2 or w.shape[1] % args.group != 0:
                    continue
                if w.shape[1] // args.group < 4:
                    continue  # need >=4 groups for M=4 sub-grouping
                mse6, mse4 = block_mse(w, args.group)
                picked = "M=4" if mse4 < mse6 else "M=6"
                rows.append({
                    "name": name,
                    "category": categorize(name),
                    "shape": tuple(w.shape),
                    "mse_m6": mse6,
                    "mse_m4": mse4,
                    "picked": picked,
                })
                seen += 1
                if seen % 10 == 0:
                    print(f"[e4]   {seen} tensors processed (last {name} → {picked})")
                if args.limit and seen >= args.limit:
                    break
        if args.limit and seen >= args.limit:
            break

    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "category", "shape",
                                          "mse_m6", "mse_m4", "picked"])
        w.writeheader()
        for r in rows:
            r["shape"] = "x".join(str(v) for v in r["shape"])
            w.writerow(r)
    print(f"[e4] wrote {len(rows)} rows to {args.out}")

    # Quick summary by category.
    by_cat: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r["picked"])
    print("[e4] M=4 selection rate by category:")
    for cat, picks in sorted(by_cat.items()):
        rate = sum(1 for p in picks if p == "M=4") / len(picks)
        print(f"  {cat:>8s}: {rate*100:5.1f}% (n={len(picks)})")


if __name__ == "__main__":
    main()
