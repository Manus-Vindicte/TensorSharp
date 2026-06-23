"""Shared helpers for engine runners: subprocess + HTTP timing utilities."""
from __future__ import annotations

import json
import subprocess
import time
import urllib.error
import urllib.request
from typing import Optional


def run_cmd(cmd: list[str], timeout_s: float, env: Optional[dict] = None
            ) -> tuple[int, str, str, float]:
    """Run a command, capturing stdout/stderr and wall time (ms)."""
    t0 = time.monotonic()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout_s, env=env)
        return proc.returncode, proc.stdout, proc.stderr, (time.monotonic() - t0) * 1000
    except subprocess.TimeoutExpired as ex:
        elapsed = (time.monotonic() - t0) * 1000
        out = ex.stdout or ""
        err = (ex.stderr or "") + f"\nTIMEOUT after {timeout_s}s"
        if isinstance(out, bytes):
            out = out.decode("utf-8", "replace")
        if isinstance(err, bytes):
            err = err.decode("utf-8", "replace")
        return 124, out, err, elapsed


def tail(text: str, n: int = 30) -> str:
    return "\n".join(text.splitlines()[-n:])


def http_post_json(url: str, body: dict, headers: Optional[dict] = None,
                   timeout_s: float = 900) -> dict:
    data = json.dumps(body).encode("utf-8")
    hdrs = {"Content-Type": "application/json"}
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(url, data=data, headers=hdrs, method="POST")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def http_post_stream(url: str, body: dict, headers: Optional[dict] = None,
                     timeout_s: float = 900):
    """POST and yield (recv_time, line) for each non-empty SSE 'data:' line."""
    data = json.dumps(body).encode("utf-8")
    hdrs = {"Content-Type": "application/json"}
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(url, data=data, headers=hdrs, method="POST")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line or not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                break
            yield time.monotonic(), payload


def server_alive(base_url: str, timeout_s: float = 5) -> bool:
    for path in ("/health", "/v1/models", "/"):
        try:
            with urllib.request.urlopen(base_url.rstrip("/") + path, timeout=timeout_s) as resp:
                if resp.status < 500:
                    return True
        except urllib.error.HTTPError as ex:
            if ex.code < 500:
                return True
        except Exception:
            continue
    return False
