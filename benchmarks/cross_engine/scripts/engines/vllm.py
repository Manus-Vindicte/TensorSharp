"""
vLLM engine runner (OpenAI-compatible server).

vLLM loads the Hugging Face model (configured per model as hf_id) rather than
the GGUF file, so it is compared as the "reference server" — same prompts and
scenarios, not the same on-disk weights. One server process serves one model;
GPU vs CPU is fixed at server launch (vLLM --device). Point base_url at an
already-running server, or set engines.vllm.launch=true to auto-start one per
(model, backend) from launch_cmd.

Covers synthetic throughput, single-turn text, image, multi-turn, and
function-call. Audio/video are out of scope for this runner.
"""
from __future__ import annotations

import base64
import json
import shlex
import subprocess
import time
from pathlib import Path

from bench_spec import BenchResult, Config, ModelSpec, TaskSpec, PROMPTS_DIR, SCENARIOS_DIR
from engines.common import server_alive
from engines.openai_client import chat_completion_timed
from engines.llamacpp import _to_openai_tools  # reuse tool-schema conversion

# Track auto-launched servers so they are reused across cells and cleaned up once.
_LAUNCHED: dict[tuple[str, str], subprocess.Popen] = {}


def _base_url(cfg: Config, backend: str) -> str:
    ec = cfg.engine_cfg("vllm")
    return ec.get(f"base_url_{backend}") or ec.get("base_url", "http://localhost:8000")


def ensure_server(cfg: Config, backend: str, model: ModelSpec) -> tuple[str, str]:
    """Return (base_url, error). Optionally auto-launch a server."""
    ec = cfg.engine_cfg("vllm")
    base_url = _base_url(cfg, backend)
    if server_alive(base_url):
        return base_url, ""
    if not ec.get("launch", False):
        return base_url, f"no vLLM server at {base_url} (set engines.vllm.launch=true to auto-start)"

    device = ec.get("backends", {}).get(backend, backend)
    key = (model.vllm_served_name, device)
    if key in _LAUNCHED and _LAUNCHED[key].poll() is None:
        return base_url, ""
    cmd_tmpl = ec.get("launch_cmd", "")
    if not cmd_tmpl:
        return base_url, "engines.vllm.launch_cmd not set"
    cmd = cmd_tmpl.format(hf_id=model.hf_id, served_name=model.vllm_served_name, device=device)
    proc = subprocess.Popen(shlex.split(cmd), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    _LAUNCHED[key] = proc
    deadline = time.monotonic() + float(ec.get("startup_timeout_s", 600))
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return base_url, f"vLLM server exited during startup (rc={proc.returncode})"
        if server_alive(base_url):
            return base_url, ""
        time.sleep(3)
    return base_url, "vLLM server did not become ready before timeout"


def shutdown_all():
    for proc in _LAUNCHED.values():
        if proc.poll() is None:
            proc.terminate()
    for proc in _LAUNCHED.values():
        try:
            proc.wait(timeout=20)
        except Exception:
            proc.kill()
    _LAUNCHED.clear()


def _image_messages(prompt: str, image_path: Path) -> list:
    b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return [{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
    ]}]


def _synthetic_prompt(n_tokens: int) -> str:
    # ~1 token per word proxy; vLLM reports the true prompt_tokens in usage.
    return "Lorem ipsum dolor sit amet " * max(1, n_tokens // 5)


def run(cfg: Config, backend: str, model: ModelSpec, task: TaskSpec) -> BenchResult:
    res = BenchResult(engine="vllm", backend=backend, model=model.short_id, task=task.short_id)
    base_url, err = ensure_server(cfg, backend, model)
    if err:
        res.skipped = True
        res.error = err
        return res
    api_key = cfg.engine_cfg("vllm").get("api_key", "EMPTY")
    served = model.vllm_served_name
    res.cmd = f"POST {base_url}/v1/chat/completions model={served} task={task.short_id}"

    if task.kind == "synthetic":
        if task.tg and not task.pp:
            messages = [{"role": "user", "content": "Write a long detailed story without stopping."}]
            max_tokens = task.tg
        else:
            messages = [{"role": "user", "content": _synthetic_prompt(task.pp or 32)}]
            max_tokens = max(task.tg, 1)
        r = chat_completion_timed(base_url, api_key, served, messages, max_tokens=max_tokens)
    elif task.kind == "multi_turn":
        return _multi_turn(base_url, api_key, served, task, res)
    elif task.kind == "function_call":
        tools = _to_openai_tools(json.loads((SCENARIOS_DIR / task.tools_file).read_text()))
        prompt = (PROMPTS_DIR / task.prompt_file).read_text().strip()
        r = chat_completion_timed(base_url, api_key, served,
                                  [{"role": "user", "content": prompt}],
                                  max_tokens=task.max_tokens, tools=tools)
        res.tool_call_ok = (not r["error"]) and len(r["tool_calls"]) > 0
    elif task.kind == "image":
        prompt = (PROMPTS_DIR / task.prompt_file).read_text().strip()
        r = chat_completion_timed(base_url, api_key, served,
                                  _image_messages(prompt, cfg.media_path("image")),
                                  max_tokens=task.max_tokens)
    else:  # text
        prompt = (PROMPTS_DIR / task.prompt_file).read_text().strip()
        r = chat_completion_timed(base_url, api_key, served,
                                  [{"role": "user", "content": prompt}],
                                  max_tokens=task.max_tokens)

    if r["error"]:
        res.error = r["error"]
        return res
    res.ttft_ms = r["ttft_ms"]
    res.e2e_ms = r["e2e_ms"]
    res.prefill_tokens = r["prompt_tokens"]
    res.decode_tokens = r["completion_tokens"]
    if task.pp:
        res.prefill_tps = r["prefill_tps"]
    if task.kind != "synthetic" or task.tg:
        res.decode_tps = r["decode_tps"]
    res.output_text = r["content"][:300]
    res.ok = r["completion_tokens"] > 0 or r["prefill_tps"] > 0 or len(r["tool_calls"]) > 0
    if not res.ok:
        res.error = "vLLM returned no tokens"
    return res


def _multi_turn(base_url, api_key, served, task, res) -> BenchResult:
    turns = [json.loads(l) for l in (SCENARIOS_DIR / task.scenario_file).read_text().splitlines() if l.strip()]
    messages: list = []
    tps_list, prompt_toks, completion_toks, e2e = [], 0, 0, 0.0
    for t in turns:
        messages.append({"role": "user", "content": t.get("user") or t.get("content", "")})
        r = chat_completion_timed(base_url, api_key, served, messages,
                                  max_tokens=t.get("max_tokens", task.max_tokens))
        if r["error"]:
            res.error = r["error"]
            return res
        messages.append({"role": "assistant", "content": r["content"]})
        if r["decode_tps"] > 0:
            tps_list.append(r["decode_tps"])
        prompt_toks += r["prompt_tokens"]
        completion_toks += r["completion_tokens"]
        e2e += r["e2e_ms"]
    res.turns = len(turns)
    res.prefill_tokens = prompt_toks
    res.decode_tokens = completion_toks
    res.e2e_ms = e2e
    if tps_list:
        res.decode_tps = sum(tps_list) / len(tps_list)
    res.ok = completion_toks > 0
    if not res.ok:
        res.error = "vLLM returned no tokens"
    return res
