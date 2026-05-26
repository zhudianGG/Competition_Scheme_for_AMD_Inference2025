"""E1 — Medusa vs EAGLE3 accept-rate sweep.

Tests claim #1: EAGLE autoregressive draft's accept rate ceiling is meaningfully
higher than Medusa's parallel heads. For each (draft_kind, K) we spin up an
sglang server pointed at MiniCPM-SALA + the corresponding draft head, replay
500 perf_public_set samples, and record accept length + e2e throughput.

Requires:
  /sgl-workspace/SpecForge/models/MiniCPM-SALA            (target)
  /sgl-workspace/SpecForge/models/MiniCPM-SALA-Medusa     (Medusa head)   [if --include medusa]
  /sgl-workspace/SpecForge/models/MiniCPM-SALA-EAGLE3     (EAGLE3 head)   [if --include eagle3]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib import runner, server  # noqa: E402

TARGET = "/sgl-workspace/SpecForge/models/MiniCPM-SALA"
MEDUSA = "/sgl-workspace/SpecForge/models/MiniCPM-SALA-Medusa"
EAGLE3 = "/sgl-workspace/SpecForge/models/MiniCPM-SALA-EAGLE3"
DATASET = "/sgl-workspace/SpecForge/SOAR-Toolkit/eval_dataset/perf_public_set.jsonl"


def server_args(kind: str, k: int) -> list[str]:
    if kind == "none":
        return []
    if kind == "medusa":
        return [
            "--speculative-algorithm", "MEDUSA",
            "--speculative-draft-model-path", MEDUSA,
            "--speculative-num-steps", str(k),
            "--speculative-num-draft-tokens", str(k + 1),
        ]
    if kind == "eagle3":
        return [
            "--speculative-algorithm", "EAGLE3",
            "--speculative-draft-model-path", EAGLE3,
            "--speculative-num-steps", str(k),
            "--speculative-num-draft-tokens", str(k + 1),
            "--speculative-eagle-topk", "1",
        ]
    raise ValueError(kind)


def load_prompts(path: Path, limit: int) -> list[str]:
    out: list[str] = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            q = obj.get("question") or obj.get("input") or obj.get("prompt")
            if q:
                out.append(q)
            if len(out) >= limit:
                break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-questions", type=int, default=500)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--kinds", nargs="+", default=["none", "medusa", "eagle3"])
    ap.add_argument("--quick", action="store_true", help="K=1,3 only; 50 samples")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "e1_accept_rate.csv")
    args = ap.parse_args()

    if args.quick:
        args.ks = [1, 3]
        args.num_questions = 50

    prompts = load_prompts(Path(DATASET), args.num_questions)
    print(f"[e1] {len(prompts)} prompts, kinds={args.kinds}, ks={args.ks}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "draft_kind", "k", "n_ok", "duration_s", "throughput_tok_s",
            "ttft_p50", "ttft_p95", "e2e_p50", "e2e_p95", "accept_length",
        ])

        for kind in args.kinds:
            ks = [0] if kind == "none" else args.ks
            for k in ks:
                extra = server_args(kind, k)
                log = ROOT / "results" / f"e1_{kind}_k{k}.log"
                print(f"\n[e1] === kind={kind} k={k} ===")
                try:
                    with server.sglang_server(TARGET, extra, log_path=log) as api_base:
                        stats = runner.run(api_base, TARGET, prompts, concurrency=1,
                                           max_new_tokens=args.max_new_tokens)
                except Exception as e:
                    print(f"[e1] kind={kind} k={k} FAILED: {e}")
                    w.writerow([kind, k, 0, 0, 0, 0, 0, 0, 0, ""])
                    continue
                w.writerow([
                    kind, k, stats.ok, f"{stats.duration:.2f}",
                    f"{stats.throughput_tok_s:.2f}",
                    f"{stats.ttft_p50:.4f}", f"{stats.ttft_p95:.4f}",
                    f"{stats.e2e_p50:.4f}", f"{stats.e2e_p95:.4f}",
                    f"{stats.avg_accept_length:.3f}" if stats.avg_accept_length else "",
                ])
                f.flush()
                print(f"[e1] kind={kind} k={k} accept_len={stats.avg_accept_length} "
                      f"tok/s={stats.throughput_tok_s:.1f}")
    print(f"[e1] wrote {args.out}")


if __name__ == "__main__":
    main()
