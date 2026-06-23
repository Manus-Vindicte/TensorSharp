"""
llama.cpp engine runner.

CLI tools for synthetic / text / image / audio (llama-bench, llama-cli,
llama-mtmd-cli); the OpenAI-compatible llama-server for multi-turn and
function-call (when engines.llamacpp.server_url is configured). GPU vs CPU is
selected with -ngl (999 = all layers on GPU, 0 = CPU).
"""
from __future__ import annotations

import base64
import json
import re
import shutil
from pathlib import Path

from bench_spec import BenchResult, Config, ModelSpec, TaskSpec, PROMPTS_DIR, SCENARIOS_DIR
from engines.common import run_cmd, tail
from engines.openai_client import chat_completion_timed

LLAMA_BENCH_ROW_RE = re.compile(
    r"^\|.*?\|\s*[\d.]+\s*GiB\s*\|.*?\|.*?\|.*?\|\s*(?P<test>\w+)\s*\|\s*(?P<tps>[0-9.]+)\s*")
PERF_PREFILL_RE = re.compile(
    r"prompt eval time\s*=\s*([0-9.]+)\s*ms\s*/\s*(\d+)\s*tokens.*?([0-9.]+)\s*tokens per second")
PERF_DECODE_RE = re.compile(
    r"eval time\s*=\s*([0-9.]+)\s*ms\s*/\s*(\d+)\s*runs.*?([0-9.]+)\s*tokens per second")


def _bin(cfg: Config, name: str) -> str:
    bin_dir = cfg.engine_cfg("llamacpp").get("bin_dir", "")
    return str(Path(bin_dir) / name) if bin_dir else name


def _bin_available(path: str) -> bool:
    return Path(path).exists() or shutil.which(path) is not None


def _ngl(cfg: Config, backend: str) -> str:
    return str(cfg.engine_cfg("llamacpp").get("backends", {}).get(backend, 0))


def _parse_bench(out: str) -> dict[str, float]:
    metrics = {}
    for line in out.splitlines():
        m = LLAMA_BENCH_ROW_RE.match(line)
        if m:
            metrics[m.group("test")] = float(m.group("tps"))
    return metrics


def run(cfg: Config, backend: str, model: ModelSpec, task: TaskSpec) -> BenchResult:
    res = BenchResult(engine="llamacpp", backend=backend, model=model.short_id, task=task.short_id)
    ngl = _ngl(cfg, backend)

    if task.kind in ("multi_turn", "function_call"):
        return _run_server(cfg, backend, model, task, res)

    if task.kind == "synthetic":
        bench_bin = _bin(cfg, "llama-bench")
        if not _bin_available(bench_bin):
            res.skipped = True
            res.error = f"llama-bench not found ({bench_bin})"
            return res
        cmd = [bench_bin, "-m", str(model.gguf),
               "-p", str(task.pp or 0), "-n", str(task.tg or 0),
               "-ngl", ngl, "-r", "3", "-o", "md"]
        res.cmd = " ".join(cmd)
        rc, out, err, wall = run_cmd(cmd, timeout_s=1800)
        res.total_wall_ms = wall
        res.raw_tail = tail((out or "") + "\n" + (err or ""), 20)
        if rc != 0:
            res.error = f"exit {rc}"
            return res
        metrics = _parse_bench(out)
        if not metrics:
            res.error = "could not parse llama-bench output"
            return res
        if task.pp and f"pp{task.pp}" in metrics:
            res.prefill_tokens = task.pp
            res.prefill_tps = metrics[f"pp{task.pp}"]
            res.prefill_ms = (task.pp / res.prefill_tps) * 1000 if res.prefill_tps else 0
        if task.tg and f"tg{task.tg}" in metrics:
            res.decode_tokens = task.tg
            res.decode_tps = metrics[f"tg{task.tg}"]
            res.decode_ms = (task.tg / res.decode_tps) * 1000 if res.decode_tps else 0
        res.ok = res.prefill_tps > 0 or res.decode_tps > 0
        if not res.ok:
            res.error = "no matching pp/tg row"
        return res

    prompt_path = PROMPTS_DIR / task.prompt_file
    if task.kind == "text":
        cmd = [_bin(cfg, "llama-cli"), "-m", str(model.gguf), "-f", str(prompt_path),
               "-n", str(task.max_tokens), "--temp", "0", "-st",
               "--no-warmup", "--no-display-prompt", "-ngl", ngl, "--jinja"]
    elif task.kind == "image":
        cmd = [_bin(cfg, "llama-mtmd-cli"), "-m", str(model.gguf), "--mmproj", str(model.mmproj),
               "--image", str(cfg.media_path("image")), "-p", prompt_path.read_text(),
               "-n", str(task.max_tokens), "--temp", "0", "-ngl", ngl, "--jinja", "--no-warmup"]
    elif task.kind == "audio":
        cmd = [_bin(cfg, "llama-mtmd-cli"), "-m", str(model.gguf), "--mmproj", str(model.mmproj),
               "--audio", str(cfg.media_path("audio")), "-p", prompt_path.read_text(),
               "-n", str(task.max_tokens), "--temp", "0", "-ngl", ngl, "--jinja", "--no-warmup"]
    else:
        res.error = f"unsupported task kind {task.kind} for llama.cpp"
        return res

    if not _bin_available(cmd[0]):
        res.skipped = True
        res.error = f"{Path(cmd[0]).name} not found ({cmd[0]})"
        return res
    res.cmd = " ".join(cmd)
    rc, out, err, wall = run_cmd(cmd, timeout_s=1800)
    res.total_wall_ms = wall
    combined = (out or "") + "\n" + (err or "")
    res.raw_tail = tail(combined, 40)
    if rc != 0:
        res.error = f"exit {rc}"
        return res

    pe = PERF_PREFILL_RE.search(combined)
    de = PERF_DECODE_RE.search(combined)
    if pe:
        res.prefill_ms, res.prefill_tokens, res.prefill_tps = float(pe.group(1)), int(pe.group(2)), float(pe.group(3))
    if de:
        res.decode_ms, res.decode_tokens, res.decode_tps = float(de.group(1)), int(de.group(2)), float(de.group(3))
    if res.prefill_ms == 0 and res.decode_ms == 0:
        res.error = "could not parse llama-cli timings"
        return res
    res.ok = True
    return res


def _run_server(cfg: Config, backend: str, model: ModelSpec, task: TaskSpec, res: BenchResult) -> BenchResult:
    server_url = cfg.engine_cfg("llamacpp").get("server_url")
    if not server_url:
        res.skipped = True
        res.error = "needs llama-server (set engines.llamacpp.server_url)"
        return res
    served = cfg.engine_cfg("llamacpp").get("server_model_name") or Path(str(model.gguf)).stem
    res.cmd = f"POST {server_url}/v1/chat/completions model={served} task={task.short_id}"

    if task.kind == "multi_turn":
        return _multi_turn(server_url, "EMPTY", served, task, res)
    return _function_call(cfg, server_url, "EMPTY", served, model, task, res)


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
        res.error = "server returned no tokens"
    return res


def _function_call(cfg, base_url, api_key, served, model, task, res) -> BenchResult:
    tools = json.loads((SCENARIOS_DIR / task.tools_file).read_text())
    oai_tools = _to_openai_tools(tools)
    prompt = (PROMPTS_DIR / task.prompt_file).read_text().strip()
    r = chat_completion_timed(base_url, api_key, served,
                              [{"role": "user", "content": prompt}],
                              max_tokens=task.max_tokens, tools=oai_tools)
    if r["error"]:
        res.error = r["error"]
        return res
    res.ttft_ms = r["ttft_ms"]
    res.e2e_ms = r["e2e_ms"]
    res.prefill_tokens = r["prompt_tokens"]
    res.prefill_tps = r["prefill_tps"]
    res.decode_tokens = r["completion_tokens"]
    res.decode_tps = r["decode_tps"]
    res.tool_call_ok = len(r["tool_calls"]) > 0
    res.output_text = r["content"][:300]
    res.ok = True
    return res


def _to_openai_tools(tools: list) -> list:
    """Convert the repo's TensorSharp tool schema to OpenAI tools format."""
    out = []
    for t in tools:
        props = {}
        for pname, p in (t.get("Parameters") or {}).items():
            entry = {"type": (p.get("Type") or "string").lower(),
                     "description": p.get("Description", "")}
            if p.get("Enum"):
                entry["enum"] = p["Enum"]
            props[pname] = entry
        out.append({"type": "function", "function": {
            "name": t["Name"],
            "description": t.get("Description", ""),
            "parameters": {"type": "object", "properties": props,
                           "required": t.get("Required", [])},
        }})
    return out
