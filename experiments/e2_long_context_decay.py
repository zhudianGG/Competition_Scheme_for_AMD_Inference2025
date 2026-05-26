"""E2 — long-context accept-rate decay.

Tests claim #2: accept rate decays as prompt length grows. Generates needle-
in-haystack prompts at {512, 1k, 2k, 4k, 8k, 16k} tokens (100 each), then for
each (length, draft_kind) pair runs them through the corresponding sglang server.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib import runner, server, workload  # noqa: E402

TARGET = "/sgl-workspace/SpecForge/models/MiniCPM-SALA"
MEDUSA = "/sgl-workspace/SpecForge/models/MiniCPM-SALA-Medusa"
EAGLE3 = "/sgl-workspace/SpecForge/models/MiniCPM-SALA-EAGLE3"

# (kind, K) — fixed per plan: medusa K=2, eagle3 K=4
DRAFTS = [
    ("medusa", 2, MEDUSA),
    ("eagle3", 4, EAGLE3),
]


def server_args(kind: str, k: int, head: str) -> list[str]:
    if kind == "medusa":
        return [
            "--speculative-algorithm", "MEDUSA",
            "--speculative-draft-model-path", head,
            "--speculative-num-steps", str(k),
            "--speculative-num-draft-tokens", str(k + 1),
        ]
    return [
        "--speculative-algorithm", "EAGLE3",
        "--speculative-draft-model-path", head,
        "--speculative-num-steps", str(k),
        "--speculative-num-draft-tokens", str(k + 1),
        "--speculative-eagle-topk", "1",
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", type=int, nargs="+",
                    default=[512, 1024, 2048, 4096, 8192, 16384])
    ap.add_argument("--n-per-length", type=int, default=100)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "e2_long_context.csv")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data" / "e2")
    ap.add_argument("--quick", action="store_true", help="20 prompts/len, lengths up to 4k")
    args = ap.parse_args()

    if args.quick:
        args.lengths = [512, 2048, 4096]
        args.n_per_length = 20

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TARGET, trust_remote_code=True)
    print(f"[e2] generating workloads under {args.data_dir}")
    for L in args.lengths:
        out = args.data_dir / f"needle_{L}.jsonl"
        if out.exists() and sum(1 for _ in out.open()) >= args.n_per_length:
            print(f"[e2]   reusing {out}")
            continue
        workload.gen_long_context_set(L, args.n_per_length, tok, out)
        print(f"[e2]   wrote {out}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "draft_kind", "k", "prompt_len", "n_ok", "throughput_tok_s",
            "ttft_p50", "e2e_p50", "accept_length",
        ])
        for kind, k, head in DRAFTS:
            extra = server_args(kind, k, head)
            log = ROOT / "results" / f"e2_{kind}.log"
            print(f"\n[e2] === kind={kind} k={k} ===")
            try:
                with server.sglang_server(TARGET, extra, log_path=log) as api_base:
                    for L in args.lengths:
                        prompts = [
                            row["prompt"]
                            for row in workload.load_jsonl(
                                args.data_dir / f"needle_{L}.jsonl",
                                limit=args.n_per_length,
                            )
                        ]
                        stats = runner.run(api_base, TARGET, prompts, concurrency=1,
                                           max_new_tokens=64)
                        w.writerow([
                            kind, k, L, stats.ok,
                            f"{stats.throughput_tok_s:.2f}",
                            f"{stats.ttft_p50:.4f}", f"{stats.e2e_p50:.4f}",
                            f"{stats.avg_accept_length:.3f}" if stats.avg_accept_length else "",
                        ])
                        f.flush()
                        print(f"[e2]   L={L} accept_len={stats.avg_accept_length}")
            except Exception as e:
                print(f"[e2] kind={kind} FAILED: {e}")
                for L in args.lengths:
                    w.writerow([kind, k, L, 0, 0, 0, 0, ""])
    print(f"[e2] wrote {args.out}")


if __name__ == "__main__":
    main()
