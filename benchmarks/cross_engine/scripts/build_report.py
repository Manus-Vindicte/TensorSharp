#!/usr/bin/env python3
"""
Aggregate cross-engine benchmark JSON results into a markdown report.

Reads results/<engine>__<backend>__<model>__<task>.json and emits a comparison
report: per-model throughput tables (engine x backend columns, prefill/decode),
plus a scenario-coverage table (multi-turn, function-call, diffusion).

Usage:
    python3 build_report.py [--results DIR] [--out FILE]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from bench_spec import TASKS, DEFAULT_RESULTS_DIR, load_config

ENGINE_ORDER = ["tensorsharp", "llamacpp", "vllm"]
ENGINE_LABEL = {"tensorsharp": "TensorSharp", "llamacpp": "llama.cpp", "vllm": "vLLM"}
BACKEND_ORDER = ["gpu", "cpu"]

THROUGHPUT_TASKS = ["pp512", "tg128", "pp2048", "short_text", "long_text", "image", "audio", "video", "diffusion"]
SCENARIO_TASKS = ["multi_turn", "function_call"]
PREFILL_ONLY = {"pp512", "pp2048"}
DECODE_ONLY = {"tg128", "diffusion"}


def load_all(results_dir: Path) -> dict:
    data: dict = {}
    for f in sorted(results_dir.glob("*.json")):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        try:
            engine, backend, model, task = f.stem.split("__")
        except ValueError:
            continue
        data.setdefault(model, {}).setdefault(task, {}).setdefault(engine, {})[backend] = d
    return data


def fmt(v) -> str:
    if v is None or not isinstance(v, (int, float)) or v <= 0:
        return "—"
    return f"{v:.1f}"


def cell(d: dict | None, metric: str) -> str:
    if d is None:
        return "n/a"
    if d.get("skipped"):
        return "skip"
    if not d.get("ok"):
        return "fail"
    return fmt(d.get(metric, 0.0))


def column_pairs(present_engines, present_backends):
    pairs = []
    for e in ENGINE_ORDER:
        if e not in present_engines:
            continue
        for b in BACKEND_ORDER:
            if b in present_backends.get(e, set()):
                pairs.append((e, b))
    return pairs


def throughput_table(model_data: dict, pairs) -> str:
    header = ["Task"]
    for e, b in pairs:
        header.append(f"{ENGINE_LABEL[e]} {b} prefill")
        header.append(f"{ENGINE_LABEL[e]} {b} decode")
    rows = ["| " + " | ".join(header) + " |",
            "|------|" + "|".join(["----:"] * (len(pairs) * 2)) + "|"]
    for t in THROUGHPUT_TASKS:
        if t not in model_data:
            continue
        cells = [TASKS[t].short_id]
        for e, b in pairs:
            d = model_data.get(t, {}).get(e, {}).get(b)
            pref = "—" if t in DECODE_ONLY else cell(d, "prefill_tps")
            dec = "—" if t in PREFILL_ONLY else cell(d, "decode_tps")
            cells.extend([pref, dec])
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def scenario_table(model_data: dict, pairs) -> str:
    if not any(t in model_data for t in SCENARIO_TASKS):
        return ""
    header = ["Scenario"]
    for e, b in pairs:
        header.append(f"{ENGINE_LABEL[e]} {b}")
    rows = ["| " + " | ".join(header) + " |",
            "|------|" + "|".join([":--:"] * len(pairs)) + "|"]
    for t in SCENARIO_TASKS:
        if t not in model_data:
            continue
        cells = [TASKS[t].short_id]
        for e, b in pairs:
            d = model_data.get(t, {}).get(e, {}).get(b)
            if d is None:
                cells.append("n/a")
            elif d.get("skipped"):
                cells.append("skip")
            elif not d.get("ok"):
                cells.append("fail")
            elif t == "function_call":
                tc = d.get("tool_call_ok")
                tag = "✅ tool" if tc else ("⚠️ no-tool" if tc is False else "ok")
                cells.append(f"{tag} ({fmt(d.get('decode_tps'))} t/s)")
            else:
                cells.append(f"{d.get('turns', 0)} turns ({fmt(d.get('decode_tps'))} t/s)")
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(DEFAULT_RESULTS_DIR))
    ap.add_argument("--out", default=str(SCRIPT_DIR.parents[0] / "REPORT.md"))
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(Path(args.config) if args.config else None)
    results_dir = Path(args.results)
    data = load_all(results_dir)

    out: list[str] = []
    out.append("# Cross-engine inference benchmark report\n")
    out.append("Comparison of **TensorSharp**, **llama.cpp** and **vLLM** across text, "
               "multimodal (image / audio / video), multi-turn and function-call scenarios "
               "on GPU and CPU backends.\n")
    out.append("All numbers are tokens / second (higher is better). `—` = metric does not "
               "apply, `n/a` = cell not in this run, `skip` = engine/backend unavailable, "
               "`fail` = errored at runtime.\n")
    out.append("Generated by `benchmarks/cross_engine/scripts/build_report.py` from the raw "
               "per-cell JSON in `benchmarks/cross_engine/results/`. TensorSharp and llama.cpp "
               "run the same GGUF file; vLLM runs the Hugging Face model (see config `hf_id`), "
               "so it is a cross-stack reference rather than a same-weights comparison.\n")

    if not data:
        out.append("\n_No results found. Run `run_benchmark.py` first._\n")
        Path(args.out).write_text("\n".join(out))
        print(f"Wrote {args.out} (no results)")
        return 0

    for mid in cfg.models:
        if mid not in data:
            continue
        model_data = data[mid]
        present_engines = set()
        present_backends: dict[str, set] = {}
        for t, ed in model_data.items():
            for e, bd in ed.items():
                present_engines.add(e)
                present_backends.setdefault(e, set()).update(bd.keys())
        pairs = column_pairs(present_engines, present_backends)
        if not pairs:
            continue

        out.append(f"\n## {cfg.models[mid].display}\n")
        out.append(f"Model id: `{mid}` · family `{cfg.models[mid].family}`\n")
        out.append("### Throughput\n")
        out.append(throughput_table(model_data, pairs))
        out.append("")
        st = scenario_table(model_data, pairs)
        if st:
            out.append("### Conversational & agentic scenarios\n")
            out.append("Multi-turn shows turns completed + mean per-turn decode speed; "
                       "function-call shows whether the model emitted a tool call.\n")
            out.append(st)
            out.append("")

    out.append("\n## Reproducing\n")
    out.append("```bash")
    out.append("cd benchmarks/cross_engine")
    out.append("cp config.example.json config.json   # edit paths / endpoints")
    out.append("python3 scripts/run_benchmark.py     # add --quick for a smoke pass")
    out.append("python3 scripts/build_report.py")
    out.append("```")
    out.append("")

    Path(args.out).write_text("\n".join(out))
    print(f"Wrote {args.out} ({Path(args.out).stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
