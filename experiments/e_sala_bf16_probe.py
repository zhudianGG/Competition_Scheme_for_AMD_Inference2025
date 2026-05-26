"""Quick bf16 SALA throughput probe via OpenBMB fork sglang.

Sequential single-request probe (fork is unstable under concurrency for short prompts).
"""
from __future__ import annotations

import asyncio
import csv
import time
from pathlib import Path

import aiohttp

API = "http://127.0.0.1:30005"
MAX_NEW = 256
RESULTS = Path("/sgl-workspace/soar_2026_w8_bench/results/sala_bf16/bf16_probe.csv")

PROMPTS = [
    "Explain quantum entanglement at a high-school level.",
    "Write a short poem about a thunderstorm in 8 lines.",
    "Describe the photosynthesis cycle in 5 bullet points.",
    "What are three risks of LLM fine-tuning on user data?",
    "Outline a 1-day Tokyo itinerary focused on history.",
    "Summarize the plot of Hamlet in three sentences.",
    "List five differences between TCP and UDP.",
    "Translate to French: 'The early bird catches the worm.'",
]


async def one(session, prompt, idx):
    payload = {
        "model": "minicpm-sala",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": MAX_NEW,
        "stream": False,
    }
    t0 = time.perf_counter()
    async with session.post(f"{API}/v1/chat/completions", json=payload,
                             timeout=aiohttp.ClientTimeout(total=600)) as resp:
        body = await resp.json()
    t1 = time.perf_counter()
    usage = body.get("usage") or {}
    return {
        "idx": idx,
        "duration_s": round(t1 - t0, 3),
        "completion_tokens": usage.get("completion_tokens", 0),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "tps": round(usage.get("completion_tokens", 0) / (t1 - t0), 2),
    }


async def main():
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    async with aiohttp.ClientSession() as session:
        # warm up
        await one(session, PROMPTS[0], -1)
        rows = []
        for i, p in enumerate(PROMPTS):
            r = await one(session, p, i)
            print(r)
            rows.append(r)
    total_tok = sum(r["completion_tokens"] for r in rows)
    total_dur = sum(r["duration_s"] for r in rows)
    avg_tps = round(total_tok / total_dur, 2) if total_dur > 0 else 0.0
    summary = {
        "concurrency": 1,
        "n_requests": len(rows),
        "total_out_tokens": total_tok,
        "total_duration_s": round(total_dur, 3),
        "avg_tps": avg_tps,
    }
    print("SUMMARY:", summary)
    with RESULTS.open("w") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {RESULTS}")


if __name__ == "__main__":
    asyncio.run(main())
