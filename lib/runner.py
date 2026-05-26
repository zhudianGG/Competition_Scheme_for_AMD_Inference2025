"""Async aiohttp pressure-test runner. Returns per-request ITL/TPOT/TTFT stats."""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Sequence

import aiohttp


@dataclass
class RequestResult:
    idx: int
    ok: bool
    ttft: float = 0.0
    e2e: float = 0.0
    out_tokens: int = 0
    spec_verify_ct: int = 0
    error: str | None = None


@dataclass
class RunStats:
    total: int
    ok: int
    duration: float
    throughput_tok_s: float
    ttft_p50: float
    ttft_p95: float
    e2e_p50: float
    e2e_p95: float
    avg_accept_length: float | None
    per_request: list[RequestResult] = field(default_factory=list)


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round(p * (len(xs) - 1)))))
    return xs[k]


async def _one(session: aiohttp.ClientSession, api_base: str, model: str, prompt: str,
               idx: int, max_new_tokens: int, sem: asyncio.Semaphore) -> RequestResult:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_new_tokens,
        "stream": True,
    }
    async with sem:
        start = time.perf_counter()
        first = None
        out_tokens = 0
        spec = 0
        try:
            async with session.post(f"{api_base}/v1/chat/completions", json=payload,
                                     timeout=aiohttp.ClientTimeout(total=600)) as resp:
                async for raw in resp.content:
                    if not raw:
                        continue
                    line = raw.decode("utf-8", errors="ignore").strip()
                    if not line.startswith("data: "):
                        continue
                    body = line[6:]
                    if body == "[DONE]":
                        break
                    if first is None:
                        first = time.perf_counter()
                    try:
                        obj = json.loads(body)
                    except Exception:
                        continue
                    usage = obj.get("usage") or {}
                    if usage:
                        out_tokens = usage.get("completion_tokens", out_tokens)
                        spec = usage.get("spec_verify_ct", spec)
            end = time.perf_counter()
            return RequestResult(idx, True, (first or end) - start, end - start, out_tokens, spec)
        except Exception as e:
            return RequestResult(idx, False, 0.0, time.perf_counter() - start, 0, 0, str(e))


async def run_async(api_base: str, model: str, prompts: Sequence[str],
                    concurrency: int, max_new_tokens: int = 256) -> RunStats:
    sem = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(limit=concurrency * 2)
    t0 = time.perf_counter()
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            _one(session, api_base, model, p, i, max_new_tokens, sem)
            for i, p in enumerate(prompts)
        ]
        results = await asyncio.gather(*tasks)
    dur = time.perf_counter() - t0

    ok = [r for r in results if r.ok]
    total_out = sum(r.out_tokens for r in ok)
    total_verify = sum(r.spec_verify_ct for r in ok)
    accept = (total_out / total_verify) if total_verify > 0 else None
    return RunStats(
        total=len(results), ok=len(ok), duration=dur,
        throughput_tok_s=total_out / dur if dur > 0 else 0.0,
        ttft_p50=_pct([r.ttft for r in ok], 0.5),
        ttft_p95=_pct([r.ttft for r in ok], 0.95),
        e2e_p50=_pct([r.e2e for r in ok], 0.5),
        e2e_p95=_pct([r.e2e for r in ok], 0.95),
        avg_accept_length=accept,
        per_request=results,
    )


def run(api_base: str, model: str, prompts: Sequence[str], concurrency: int,
        max_new_tokens: int = 256) -> RunStats:
    return asyncio.run(run_async(api_base, model, prompts, concurrency, max_new_tokens))
