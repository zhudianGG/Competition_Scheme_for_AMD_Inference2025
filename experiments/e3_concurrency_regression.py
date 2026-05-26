"""E3 — concurrency break-even.

Tests claim #3: speculative decoding's gain shrinks with concurrency and goes
negative past some break-even. Calls `python -m sglang.bench_serving` directly
(rather than the SOAR wrapper) so we can pick arbitrary concurrencies.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib import server  # noqa: E402

TARGET = "/sgl-workspace/SpecForge/models/MiniCPM-SALA"
MEDUSA = "/sgl-workspace/SpecForge/models/MiniCPM-SALA-Medusa"
EAGLE3 = "/sgl-workspace/SpecForge/models/MiniCPM-SALA-EAGLE3"
SHAREGPT = "/sgl-workspace/SpecForge/SOAR-Toolkit/eval_dataset/perf_public_set.jsonl"

DRAFTS = {
    "none":   [],
    "medusa": ["--speculative-algorithm", "MEDUSA",
               "--speculative-draft-model-path", MEDUSA,
               "--speculative-num-steps", "2",
               "--speculative-num-draft-tokens", "3"],
    "eagle3": ["--speculative-algorithm", "EAGLE3",
               "--speculative-draft-model-path", EAGLE3,
               "--speculative-num-steps", "4",
               "--speculative-num-draft-tokens", "5",
               "--speculative-eagle-topk", "1"],
}

_RE_THROUGHPUT = re.compile(r"Output token throughput \(tok/s\):\s+([\d.]+)")
_RE_DUR = re.compile(r"Benchmark duration \(s\):\s+([\d.]+)")
_RE_TTFT_P50 = re.compile(r"Median TTFT\s*\(ms\):\s+([\d.]+)")
_RE_TTFT_P95 = re.compile(r"P95 TTFT\s*\(ms\):\s+([\d.]+)")


def to_custom_dataset(src: Path, dst: Path, limit: int) -> int:
    n = 0
    with src.open() as fin, dst.open("w") as fout:
        for line in fin:
            if not line.strip() or n >= limit:
                break
            obj = json.loads(line)
            q = obj.get("question") or obj.get("input") or obj.get("prompt")
            if not q:
                continue
            fout.write(json.dumps({
                "conversations": [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": "placeholder"},
                ]
            }) + "\n")
            n += 1
    return n


def parse(out: str) -> dict[str, float]:
    def grab(rx: re.Pattern[str]) -> float:
        m = rx.search(out)
        return float(m.group(1)) if m else 0.0
    return {
        "throughput_tok_s": grab(_RE_THROUGHPUT),
        "duration_s": grab(_RE_DUR),
        "ttft_p50_ms": grab(_RE_TTFT_P50),
        "ttft_p95_ms": grab(_RE_TTFT_P95),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrencies", type=int, nargs="+",
                    default=[1, 2, 4, 8, 16, 32, 64])
    ap.add_argument("--kinds", nargs="+", default=["none", "medusa", "eagle3"])
    ap.add_argument("--num-prompts", type=int, default=200)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "e3_concurrency.csv")
    ap.add_argument("--quick", action="store_true", help="conc {1,4,16}, 50 prompts")
    args = ap.parse_args()

    if args.quick:
        args.concurrencies = [1, 4, 16]
        args.num_prompts = 50

    args.out.parent.mkdir(parents=True, exist_ok=True)
    data = ROOT / "data" / "e3_custom.jsonl"
    data.parent.mkdir(parents=True, exist_ok=True)
    n = to_custom_dataset(Path(SHAREGPT), data, args.num_prompts)
    print(f"[e3] prepared {n} prompts at {data}")

    with args.out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["draft_kind", "concurrency", "throughput_tok_s",
                    "duration_s", "ttft_p50_ms", "ttft_p95_ms"])
        for kind in args.kinds:
            extra = DRAFTS[kind]
            log = ROOT / "results" / f"e3_{kind}.log"
            print(f"\n[e3] === kind={kind} ===")
            try:
                with server.sglang_server(TARGET, extra, log_path=log) as api_base:
                    host = api_base.replace("http://", "").split(":")[0]
                    port = api_base.rsplit(":", 1)[1]
                    for c in args.concurrencies:
                        cmd = [
                            sys.executable, "-m", "sglang.bench_serving",
                            "--backend", "sglang",
                            "--host", host, "--port", port,
                            "--dataset-name", "custom",
                            "--dataset-path", str(data),
                            "--num-prompts", str(n),
                            "--max-concurrency", str(c),
                            "--flush-cache",
                        ]
                        print(f"[e3]   concurrency={c}")
                        cp = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
                        stats = parse(cp.stdout + cp.stderr)
                        w.writerow([kind, c,
                                    f"{stats['throughput_tok_s']:.2f}",
                                    f"{stats['duration_s']:.2f}",
                                    f"{stats['ttft_p50_ms']:.2f}",
                                    f"{stats['ttft_p95_ms']:.2f}"])
                        f.flush()
            except Exception as e:
                print(f"[e3] kind={kind} FAILED: {e}")
                for c in args.concurrencies:
                    w.writerow([kind, c, 0, 0, 0, 0])
    print(f"[e3] wrote {args.out}")


if __name__ == "__main__":
    main()
