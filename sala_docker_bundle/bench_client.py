"""Smoke test client for the OpenBMB MiniCPM-SALA sglang fork.

Runs three probes against a server on $SALA_PORT (default 30005), hitting the
native sglang `/generate` endpoint (text + sampling_params schema):

  short  : "Hello, my name is" + 32-token generation     (basic decode loop)
  long4  : ~11.8k-token prompt + 4-token generation       (minimal long-prompt smoke)
  long64 : ~11.8k-token prompt + 64-token generation      (full sparse decode regime)

Reports per-probe latency and prints the first 200 chars of each response.
Exits non-zero on any failure (HTTP error or empty completion).
"""
import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
PORT = int(os.environ.get("SALA_PORT", 30005))
URL = f"http://127.0.0.1:{PORT}/generate"


def post(payload: dict, timeout: float):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(URL, data=data,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read())
    return body, time.time() - t0


def probe(name: str, payload: dict, timeout: float) -> bool:
    print(f"\n--- {name} (timeout {timeout}s) ---")
    try:
        body, dt = post(payload, timeout)
    except Exception as e:
        print(f"  FAIL ({type(e).__name__}): {e}")
        return False
    out = body.get("text", "") if isinstance(body, dict) else ""
    if not out and isinstance(body, list) and body:
        out = body[0].get("text", "")
    meta = body.get("meta_info", {}) if isinstance(body, dict) else {}
    ok = bool(out)
    print(f"  {'OK' if ok else 'FAIL'} in {dt:.2f}s — out_chars={len(out)}, "
          f"completion_tokens={meta.get('completion_tokens', '?')}")
    print(f"  preview: {str(out)[:200].rstrip()!r}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-long", action="store_true",
                    help="skip the 11.8k-token probes during initial bring-up")
    args = ap.parse_args()

    short = json.loads((HERE / "short_payload.json").read_text())
    long_full = json.loads((HERE / "payloads/long_prompt_11796tok.json").read_text())
    long_short = json.loads((HERE / "payloads/long_prompt_short_gen.json").read_text())

    results = []
    results.append(("short_32tok", probe("short prompt + 32 tok", short, 60)))
    if not args.skip_long:
        results.append(("long_4tok", probe("11.8k prompt + 4 tok", long_short, 120)))
        results.append(("long_64tok", probe("11.8k prompt + 64 tok", long_full, 600)))

    print("\n=== summary ===")
    for name, ok in results:
        print(f"  {name:15s}: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if all(ok for _, ok in results) else 1)


if __name__ == "__main__":
    main()
