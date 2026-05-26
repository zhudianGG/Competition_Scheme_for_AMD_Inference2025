"""Metrics extraction from SGLang server log and /get_server_info."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import requests


_AVG_ACCEPT_RE = re.compile(r"avg accept length:\s*([\d.]+)", re.IGNORECASE)
_VERIFY_MS_RE = re.compile(r"verify[_ ]ms[_ ]avg[:=]\s*([\d.]+)", re.IGNORECASE)


def parse_log_accept_length(log_path: Path) -> float | None:
    """Returns the last avg accept length printed, or None if missing."""
    if not log_path.exists():
        return None
    last: float | None = None
    for line in log_path.read_text().splitlines():
        m = _AVG_ACCEPT_RE.search(line)
        if m:
            last = float(m.group(1))
    return last


def from_response_meta(meta: dict[str, Any]) -> tuple[int, int]:
    """(completion_tokens, spec_verify_ct) — accept length = comp / verify."""
    return meta.get("completion_tokens", 0), meta.get("spec_verify_ct", 0)


def server_info(api_base: str) -> dict[str, Any]:
    r = requests.get(f"{api_base}/get_server_info", timeout=5)
    r.raise_for_status()
    return r.json()
