"""E3: Sglang EAGLE3 reference benchmark (gpt-oss-20b + baseten EAGLE3 head).

Calibration: confirms that sglang's stock EAGLE3 spec decoding delivers expected
speedups on the same prompt set we use for SALA, validating the comparison
context. SALA can't be run through sglang (model not in registry), so this is
the apples-to-oranges control: same prompts, same decoding regime, different
model + draft head + serving stack.

Reports per-prompt and overall: tokens generated, wall time, tps, and spec
accept length (rounds and mean accept per round) from sglang's metrics.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import requests


def gen_one(base_url: str, prompt: str, max_new_tokens: int, eagle: bool):
    payload = {
        "text": prompt,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": max_new_tokens,
        },
        "return_logprob": False,
    }
    t0 = time.time()
    r = requests.post(f"{base_url}/generate", json=payload, timeout=300)
    elapsed = time.time() - t0
    r.raise_for_status()
    data = r.json()
    meta = data.get("meta_info", {})
    return {
        "text": data.get("text", ""),
        "elapsed": elapsed,
        "completion_tokens": meta.get("completion_tokens", 0),
        "spec_accept_length": meta.get("spec_verify_ct"),
        "meta": meta,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:30005")
    ap.add_argument("--prompts", type=Path,
                    default=Path("/sgl-workspace/SpecForge/SOAR-Toolkit/eval_dataset/perf_public_set.jsonl"))
    ap.add_argument("--n-prompts", type=int, default=10)
    ap.add_argument("--gen-tokens", type=int, default=64)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parents[1] / "results" / "e3_sglang_eagle3.csv")
    args = ap.parse_args()

    # wait for server
    for _ in range(60):
        try:
            r = requests.get(f"{args.base_url}/health", timeout=2)
            if r.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(2)
    else:
        raise RuntimeError(f"sglang server at {args.base_url} not responding")

    prompts = []
    with args.prompts.open() as f:
        for line in f:
            row = json.loads(line)
            q = row.get("question") or row.get("prompt")
            if q:
                prompts.append(q)
            if len(prompts) >= args.n_prompts:
                break
    print(f"[e3] {len(prompts)} prompts, gen_tokens={args.gen_tokens}")

    # warmup
    print("[e3] warmup")
    gen_one(args.base_url, prompts[0], args.gen_tokens, eagle=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    total_tokens = total_time = 0.0
    all_accept = []
    for i, q in enumerate(prompts):
        res = gen_one(args.base_url, q, args.gen_tokens, eagle=True)
        n = res["completion_tokens"]
        t = res["elapsed"]
        total_tokens += n
        total_time += t
        accept = res["spec_accept_length"]
        if accept is not None:
            all_accept.append(accept)
        tps = n / t if t > 0 else 0.0
        print(f"[e3] prompt {i}: gen={n}tok in {t:.2f}s ({tps:.1f}tok/s) spec_accept={accept}")
        rows.append({
            "prompt_idx": i, "tokens": n, "time_s": f"{t:.3f}",
            "tps": f"{tps:.2f}", "spec_accept_length": accept,
        })

    overall_tps = total_tokens / total_time if total_time > 0 else 0.0
    mean_accept = sum(all_accept) / len(all_accept) if all_accept else None
    print(f"\n[e3] OVERALL: {total_tokens} tok / {total_time:.2f} s = {overall_tps:.2f} tok/s")
    if mean_accept is not None:
        print(f"[e3] OVERALL mean spec accept length = {mean_accept:.3f} (n={len(all_accept)})")

    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["prompt_idx", "tokens", "time_s", "tps", "spec_accept_length"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
        w.writerow({
            "prompt_idx": "OVERALL", "tokens": total_tokens,
            "time_s": f"{total_time:.3f}", "tps": f"{overall_tps:.3f}",
            "spec_accept_length": f"{mean_accept:.4f}" if mean_accept is not None else "",
        })
    print(f"[e3] wrote {args.out}")


if __name__ == "__main__":
    main()
