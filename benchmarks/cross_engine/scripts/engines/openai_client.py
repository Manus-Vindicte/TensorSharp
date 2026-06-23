"""
OpenAI-compatible chat client with client-side throughput timing.

Used by both the vLLM runner and the llama.cpp server runner. Streams the
completion to observe time-to-first-token (a prefill-latency proxy) and derives
prefill / decode tokens-per-second from the server-reported usage counts.
"""
from __future__ import annotations

import json
import time
from typing import Optional

from engines.common import http_post_stream


def chat_completion_timed(base_url: str, api_key: str, model: str,
                          messages: list, max_tokens: int,
                          tools: Optional[list] = None,
                          timeout_s: float = 900) -> dict:
    """Run a streamed chat completion and return timing + usage metrics.

    Returns a dict with: ttft_ms, e2e_ms, prompt_tokens, completion_tokens,
    prefill_tps, decode_tps, content, tool_calls (list), error (str or "").
    """
    url = base_url.rstrip("/") + "/v1/chat/completions"
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    headers = {"Authorization": f"Bearer {api_key or 'EMPTY'}"}

    out = {"ttft_ms": 0.0, "e2e_ms": 0.0, "prompt_tokens": 0, "completion_tokens": 0,
           "prefill_tps": 0.0, "decode_tps": 0.0, "content": "", "tool_calls": [], "error": ""}
    content_parts: list[str] = []
    tool_calls: list = []
    t0 = time.monotonic()
    first_token_t: Optional[float] = None
    usage = None

    try:
        for recv_t, payload in http_post_stream(url, body, headers, timeout_s):
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {}) or {}
                piece = delta.get("content")
                if piece:
                    if first_token_t is None:
                        first_token_t = recv_t
                    content_parts.append(piece)
                if delta.get("tool_calls"):
                    if first_token_t is None:
                        first_token_t = recv_t
                    tool_calls.extend(delta["tool_calls"])
    except Exception as ex:
        out["error"] = f"request failed: {ex}"
        return out

    e2e = (time.monotonic() - t0) * 1000
    out["e2e_ms"] = e2e
    out["content"] = "".join(content_parts)
    out["tool_calls"] = tool_calls
    if usage:
        out["prompt_tokens"] = int(usage.get("prompt_tokens", 0))
        out["completion_tokens"] = int(usage.get("completion_tokens", 0))

    if first_token_t is not None:
        ttft = (first_token_t - t0) * 1000
        out["ttft_ms"] = ttft
        if out["prompt_tokens"] and ttft > 0:
            out["prefill_tps"] = out["prompt_tokens"] / (ttft / 1000.0)
        decode_ms = e2e - ttft
        decode_toks = max(out["completion_tokens"] - 1, 0)
        if decode_ms > 0 and decode_toks > 0:
            out["decode_tps"] = decode_toks / (decode_ms / 1000.0)
    return out
