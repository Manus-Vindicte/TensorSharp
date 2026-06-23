#!/usr/bin/env python3
"""
Cross-engine inference benchmark driver.

Runs the same (model, task) scenarios across TensorSharp, llama.cpp and vLLM on
GPU and CPU backends, then writes one JSON file per
engine x backend x model x task cell into the results directory. Use
build_report.py to aggregate the JSON into a markdown comparison report.

Examples:
    # everything the config + installed engines can run
    python3 run_benchmark.py

    # a fast smoke pass on one model
    python3 run_benchmark.py --quick --models gemma4 --tasks short_text,tg128

    # only TensorSharp vs llama.cpp on GPU, text + tools
    python3 run_benchmark.py --engines tensorsharp,llamacpp --backends gpu \
        --tasks short_text,long_text,function_call

See README.md for configuration and prerequisites.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import json
import time

from bench_spec import (BACKENDS, TASKS, QUICK_OVERRIDES, DEFAULT_RESULTS_DIR,
                        BenchResult, load_config, applies)
from engines import tensorsharp as ts_engine
from engines import llamacpp as llama_engine
from engines import vllm as vllm_engine

ENGINE_RUNNERS = {
    "tensorsharp": ts_engine.run,
    "llamacpp": llama_engine.run,
    "vllm": vllm_engine.run,
}

ALL_ENGINES = ["tensorsharp", "llamacpp", "vllm"]
ALL_TASKS = list(TASKS.keys())


def main() -> int:
    ap = argparse.ArgumentParser(description="Cross-engine inference benchmark driver")
    ap.add_argument("--engines", default=",".join(ALL_ENGINES),
                    help="comma-separated: tensorsharp,llamacpp,vllm")
    ap.add_argument("--backends", default=",".join(BACKENDS),
                    help="comma-separated: gpu,cpu")
    ap.add_argument("--models", default="gemma4,qwen36,diffusiongemma",
                    help="comma-separated model ids from config.json")
    ap.add_argument("--tasks", default=",".join(ALL_TASKS),
                    help="comma-separated task ids")
    ap.add_argument("--config", default=None, help="path to config.json")
    ap.add_argument("--results", default=str(DEFAULT_RESULTS_DIR))
    ap.add_argument("--quick", action="store_true", help="smaller token counts (smoke)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip a cell if its result JSON already exists")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan (with applicability) and exit")
    args = ap.parse_args()

    cfg = load_config(Path(args.config) if args.config else None)

    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    model_ids = [m.strip() for m in args.models.split(",") if m.strip()]
    task_ids = [t.strip() for t in args.tasks.split(",") if t.strip()]

    unknown_models = [m for m in model_ids if m not in cfg.models]
    if unknown_models:
        print(f"error: models not in config: {unknown_models}", file=sys.stderr)
        print(f"available: {list(cfg.models)}", file=sys.stderr)
        return 2

    if args.quick:
        for tid, overrides in QUICK_OVERRIDES.items():
            for k, v in overrides.items():
                setattr(TASKS[tid], k, v)

    results_dir = Path(args.results)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Build the plan, recording skip reasons for non-applicable cells.
    plan = []
    skips = []
    for engine in engines:
        for backend in backends:
            for mid in model_ids:
                model = cfg.models[mid]
                for tid in task_ids:
                    task = TASKS[tid]
                    ok, why = applies(engine, backend, model, task)
                    if ok:
                        plan.append((engine, backend, model, task))
                    else:
                        skips.append((engine, backend, mid, tid, why))

    print("# cross-engine benchmark plan")
    print(f"engines : {engines}")
    print(f"backends: {backends}")
    print(f"models  : {model_ids}")
    print(f"tasks   : {task_ids}")
    print(f"out dir : {results_dir}")
    print(f"runnable cells: {len(plan)}   skipped cells: {len(skips)}\n")

    if args.dry_run:
        for engine, backend, model, task in plan:
            print(f"  RUN  {engine:11s} {backend:3s} {model.short_id:14s} {task.short_id}")
        for engine, backend, mid, tid, why in skips:
            print(f"  skip {engine:11s} {backend:3s} {mid:14s} {tid:13s} -> {why}")
        return 0

    ok_count = fail_count = 0
    try:
        for i, (engine, backend, model, task) in enumerate(plan, 1):
            out_file = results_dir / f"{engine}__{backend}__{model.short_id}__{task.short_id}.json"
            if args.skip_existing and out_file.exists():
                print(f"[{i:3d}/{len(plan)}] {engine:11s} {backend:3s} {model.short_id:14s} {task.short_id:13s} cached")
                continue
            print(f"[{i:3d}/{len(plan)}] {engine:11s} {backend:3s} {model.short_id:14s} {task.short_id:13s} ...", flush=True)
            t0 = time.monotonic()
            try:
                result = ENGINE_RUNNERS[engine](cfg, backend, model, task)
            except Exception as ex:
                result = BenchResult(engine=engine, backend=backend, model=model.short_id,
                                     task=task.short_id, error=f"runner exception: {ex}")
            wall = time.monotonic() - t0
            out_file.write_text(json.dumps(asdict(result), indent=2, default=str))
            if result.skipped:
                status = "SKIP"
            elif result.ok:
                status = "OK  "
                ok_count += 1
            else:
                status = "FAIL"
                fail_count += 1
            extra = ""
            if result.tool_call_ok is not None:
                extra += f" tool_call={result.tool_call_ok}"
            if result.turns:
                extra += f" turns={result.turns}"
            print(f"          -> {status} prefill={result.prefill_tps:7.1f} t/s "
                  f"decode={result.decode_tps:6.1f} t/s wall={wall:5.1f}s "
                  f"{result.error}{extra}", flush=True)
    finally:
        vllm_engine.shutdown_all()

    print(f"\n# done: {ok_count} ok, {fail_count} failed, "
          f"{len(plan) - ok_count - fail_count} skipped/cached")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
