"""
TensorSharp engine runner.

Drives the TensorSharp.Cli binary across every task kind:
synthetic (--benchmark), single-turn text, image/audio/video, multi-turn
(--multi-turn-jsonl), function-call (--tools), and text diffusion
(--diffusion-steps). Timings are parsed from the CLI's own info-level logs.
"""
from __future__ import annotations

import re
import statistics
from pathlib import Path

from bench_spec import BenchResult, Config, ModelSpec, TaskSpec, SCENARIOS_DIR, PROMPTS_DIR
from engines.common import run_cmd, tail

# CLI log line patterns (work with or without the "cli.inference " prefix).
BENCH_SUMMARY_RE = re.compile(
    r"benchmark summary:\s*bestPrefillMs=(?P<pms>[0-9.]+)\s*bestPrefillTps=(?P<pps>[0-9.]+)\s*"
    r"bestDecodeMs=(?P<dms>[0-9.]+)\s*bestDecodeTps=(?P<dps>[0-9.]+)")
PREFILL_RE = re.compile(r"prefill complete:\s*tokens=(?P<tok>\d+)\s*ms=(?P<ms>[0-9.]+)\s*tokensPerSec=(?P<tps>[0-9.]+)")
DECODE_RE = re.compile(r"decode complete:\s*tokens=(?P<tok>\d+)\s*ms=(?P<ms>[0-9.]+)\s*tokensPerSec=(?P<tps>[0-9.]+)")
LOAD_RE = re.compile(r"Loaded model .*? elapsedMs=(?P<load_ms>[0-9.]+)")
MT_TURN_RE = re.compile(r"multi-turn content chars=\d+ tokens=(?P<tok>\d+) decodeMs=(?P<ms>[0-9.]+) tokPerSec=(?P<tps>[0-9.]+)")
MT_PREFILL_RE = re.compile(r"multi-turn prompt tokens=(?P<tok>\d+)")


def _backend_id(cfg: Config, backend: str) -> str:
    return cfg.engine_cfg("tensorsharp").get("backends", {}).get(backend, "cpu")


def run(cfg: Config, backend: str, model: ModelSpec, task: TaskSpec) -> BenchResult:
    res = BenchResult(engine="tensorsharp", backend=backend, model=model.short_id, task=task.short_id)
    bin_path = cfg.engine_cfg("tensorsharp").get("bin")
    if not bin_path:
        res.skipped = True
        res.error = "tensorsharp.bin not configured"
        return res
    if not Path(bin_path).exists():
        res.skipped = True
        res.error = f"TensorSharp.Cli not built at {bin_path} (dotnet build -c Release)"
        return res
    ts_backend = _backend_id(cfg, backend)
    base = [str(bin_path), "--model", str(model.gguf), "--backend", ts_backend,
            "--log-level", "info", "--log-file", "off"]

    if task.kind == "synthetic":
        cmd = base + ["--benchmark",
                      "--bench-prefill", str(task.pp or 32),
                      "--bench-decode", str(task.tg or 1),
                      "--bench-runs", "3"]
    elif task.kind == "multi_turn":
        cmd = base + ["--multi-turn-jsonl", str(SCENARIOS_DIR / task.scenario_file),
                      "--max-tokens", str(task.max_tokens), "--temperature", "0"]
    elif task.kind == "diffusion":
        cmd = base + ["--input", str(PROMPTS_DIR / task.prompt_file),
                      "--max-tokens", str(task.max_tokens),
                      "--diffusion-steps", str(model.diffusion_steps),
                      "--diffusion-seed", "0"]
    else:
        cmd = base + ["--input", str(PROMPTS_DIR / task.prompt_file),
                      "--max-tokens", str(task.max_tokens),
                      "--temperature", "0", "--warmup-runs", "1"]
        if task.kind == "function_call":
            cmd += ["--tools", str(SCENARIOS_DIR / task.tools_file)]
        elif task.kind == "image":
            cmd += ["--image", str(cfg.media_path("image"))]
        elif task.kind == "audio":
            cmd += ["--audio", str(cfg.media_path("audio"))]
        elif task.kind == "video":
            cmd += ["--video", str(cfg.media_path("video"))]
        if task.kind in ("image", "audio", "video") and model.mmproj is not None:
            cmd += ["--mmproj", str(model.mmproj)]

    res.cmd = " ".join(cmd)
    rc, out, err, wall = run_cmd(cmd, timeout_s=1800)
    res.total_wall_ms = wall
    combined = (err or "") + "\n" + (out or "")
    res.raw_tail = tail(combined, 30)

    if rc != 0:
        res.error = f"exit {rc}"
        return res

    m = LOAD_RE.search(combined)
    if m:
        res.model_load_ms = float(m.group("load_ms"))

    if task.kind == "synthetic":
        m = BENCH_SUMMARY_RE.search(combined)
        if not m:
            res.error = "could not parse benchmark summary"
            return res
        if task.pp:
            res.prefill_tokens = task.pp
            res.prefill_ms = float(m.group("pms"))
            res.prefill_tps = float(m.group("pps"))
        if task.tg:
            res.decode_tokens = task.tg
            res.decode_ms = float(m.group("dms"))
            res.decode_tps = float(m.group("dps"))
        res.ok = True
        return res

    if task.kind == "multi_turn":
        tps = [float(x.group("tps")) for x in MT_TURN_RE.finditer(combined)]
        toks = [int(x.group("tok")) for x in MT_TURN_RE.finditer(combined)]
        mss = [float(x.group("ms")) for x in MT_TURN_RE.finditer(combined)]
        prefill_toks = [int(x.group("tok")) for x in MT_PREFILL_RE.finditer(combined)]
        if not tps:
            res.error = "could not parse multi-turn timings"
            return res
        res.turns = len(tps)
        res.decode_tokens = sum(toks)
        res.decode_ms = sum(mss)
        res.decode_tps = statistics.mean(tps)
        res.prefill_tokens = sum(prefill_toks)
        res.ok = True
        return res

    if task.kind == "diffusion":
        res.diffusion_steps = model.diffusion_steps
        # diffusion uses the same decode-complete log line for generated-token timing
        m = DECODE_RE.search(combined)
        if m:
            res.decode_tokens = int(m.group("tok"))
            res.decode_ms = float(m.group("ms"))
            res.decode_tps = float(m.group("tps"))
        mp = PREFILL_RE.search(combined)
        if mp:
            res.prefill_tokens = int(mp.group("tok"))
            res.prefill_ms = float(mp.group("ms"))
            res.prefill_tps = float(mp.group("tps"))
        res.ok = res.decode_ms > 0 or res.prefill_ms > 0 or "diffusion" in combined.lower()
        if not res.ok:
            res.error = "could not parse diffusion timings"
        return res

    # text / image / audio / video / function_call
    mp = PREFILL_RE.search(combined)
    if mp:
        res.prefill_tokens = int(mp.group("tok"))
        res.prefill_ms = float(mp.group("ms"))
        res.prefill_tps = float(mp.group("tps"))
    md = DECODE_RE.search(combined)
    if md:
        res.decode_tokens = int(md.group("tok"))
        res.decode_ms = float(md.group("ms"))
        res.decode_tps = float(md.group("tps"))
    if task.kind == "function_call":
        res.tool_call_ok = ("--- Tool Calls ---" in out) or ("Function:" in out)
    if res.prefill_ms == 0 and res.decode_ms == 0:
        res.error = "could not parse prefill/decode metrics"
        return res
    res.ok = True
    return res
