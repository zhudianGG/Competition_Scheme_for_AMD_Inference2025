"""SGLang server lifecycle wrapper.

Spawns `python -m sglang.launch_server` with caller-supplied args, waits for the
/get_server_info endpoint to respond, yields the API base, and tears down cleanly.
"""
from __future__ import annotations

import contextlib
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterator, Sequence

import requests


DEFAULT_MODEL = "/sgl-workspace/SpecForge/models/MiniCPM-SALA"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_ready(api_base: str, timeout: float = 600.0) -> None:
    deadline = time.time() + timeout
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            r = requests.get(f"{api_base}/get_server_info", timeout=2)
            if r.status_code == 200:
                return
        except Exception as e:
            last_err = e
        time.sleep(2.0)
    raise RuntimeError(f"sglang server at {api_base} did not become ready: {last_err}")


@contextlib.contextmanager
def sglang_server(
    model_path: str = DEFAULT_MODEL,
    extra_args: Sequence[str] = (),
    log_path: Path | None = None,
    env: dict[str, str] | None = None,
    port: int | None = None,
    host: str = "127.0.0.1",
) -> Iterator[str]:
    port = port or _free_port()
    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", model_path,
        "--host", host, "--port", str(port),
        "--trust-remote-code",
        *extra_args,
    ]
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    log_fp = open(log_path, "w", buffering=1) if log_path else subprocess.DEVNULL
    print(f"[server] launching: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=log_fp if log_path else subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=merged_env,
        preexec_fn=os.setsid,
    )
    api_base = f"http://{host}:{port}"
    try:
        _wait_ready(api_base)
        yield api_base
    finally:
        print(f"[server] terminating pid={proc.pid}")
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGINT)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        if log_path:
            log_fp.close()
