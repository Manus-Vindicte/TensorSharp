# Cross-engine inference benchmark automation

Reproducible automation that benchmarks **TensorSharp** against **llama.cpp** and
**vLLM** on the same prompts and scenarios, then produces a markdown comparison
report.

It sweeps a four-dimensional matrix:

| Dimension | Values |
|---|---|
| **Engine** | `tensorsharp`, `llamacpp`, `vllm` |
| **Backend** | `gpu`, `cpu` |
| **Model** | `gemma4` (Gemma 4 E4B), `qwen36` (Qwen3.6 35B-A3B MoE), `diffusiongemma` (DiffusionGemma text-diffusion) |
| **Task / scenario** | synthetic prefill/decode, single-turn text, image, audio, video, multi-turn chat, function/tool calling, text diffusion |

Each `engine × backend × model × task` cell is run independently, timed, and
written to its own JSON file, so partial runs and re-runs are cheap
(`--skip-existing`). Cells that cannot run (e.g. video on vLLM, tools on
DiffusionGemma, an engine that isn't installed) are auto-skipped with a recorded
reason instead of failing the matrix.

> This is a *generalized, machine-configurable* harness. It is separate from the
> captured Apple-Silicon snapshot in [`../inference_matrix/`](../inference_matrix/)
> (TensorSharp vs llama.cpp vs Ollama on Gemma 4 only), which is left untouched.

## Layout

```
cross_engine/
├── config.example.json      # copy to config.json and edit paths/endpoints
├── prompts/                 # single-turn + multimodal prompt text
├── scenarios/               # multi_turn.jsonl, tools_weather.json
├── scripts/
│   ├── bench_spec.py        # config loader + model/task/result model + applicability
│   ├── run_benchmark.py     # orchestrator (the entry point)
│   ├── build_report.py      # JSON results -> markdown report
│   └── engines/
│       ├── common.py        # subprocess + HTTP helpers
│       ├── openai_client.py # streamed OpenAI chat timing (shared)
│       ├── tensorsharp.py   # TensorSharp.Cli runner (all task kinds)
│       ├── llamacpp.py      # llama-bench / llama-cli / llama-mtmd-cli / llama-server
│       └── vllm.py          # vLLM OpenAI-compatible server runner
└── results/                 # per-cell JSON (git-ignored)
```

## Prerequisites

The harness only needs Python 3.9+ (standard library only — no pip packages).
Install whichever engines you want to compare; missing ones are skipped.

- **TensorSharp** — build the CLI in Release:
  `dotnet build TensorSharp.Cli/TensorSharp.Cli.csproj -c Release`.
  The build output is `TensorSharp.Cli/bin/Release/TensorSharp.Cli` (the project
  sets `AppendTargetFrameworkToOutputPath=false`, so there is no `net10.0`
  subfolder). GPU backend defaults to `ggml_cuda` (NVIDIA) — change to
  `ggml_metal` on macOS in `config.json`. CPU backend defaults to native `ggml_cpu`.
- **llama.cpp** — `llama-bench`, `llama-cli`, `llama-mtmd-cli` on `PATH` (or set
  `engines.llamacpp.bin_dir`). For multi-turn / function-call, run an
  OpenAI-compatible `llama-server` and set `engines.llamacpp.server_url`.
- **vLLM** — an OpenAI-compatible server (`vllm serve <hf_id>`). Point
  `engines.vllm.base_url` at it, or set `engines.vllm.launch=true` to let the
  harness start one per `(model, backend)` from `launch_cmd`.

Model files: GGUF files for TensorSharp/llama.cpp under `model_dir`; the matching
Hugging Face repo id (`hf_id`) for vLLM. Media files (`apple.png`,
`obama_first_45_secs.mp3`, `concert.mp4`) under `data_dir` for image/audio/video.

## Configure

```bash
cd benchmarks/cross_engine
cp config.example.json config.json
$EDITOR config.json     # set repo_root, model_dir, data_dir, engine paths/endpoints
```

Path strings support `{repo_root}`, `{model_dir}` and `{data_dir}` placeholders.
`config.json` is git-ignored so machine-specific paths never get committed.

## Run

```bash
# Full matrix the config + installed engines can run
python3 scripts/run_benchmark.py

# See the plan (which cells run, which are skipped and why) without running
python3 scripts/run_benchmark.py --dry-run

# Fast smoke pass
python3 scripts/run_benchmark.py --quick --models gemma4 --tasks short_text,tg128,function_call

# Narrow the sweep
python3 scripts/run_benchmark.py --engines tensorsharp,vllm --backends gpu \
    --models gemma4,qwen36 --tasks long_text,image,multi_turn

# Resume a partial run
python3 scripts/run_benchmark.py --skip-existing
```

Then build the report:

```bash
python3 scripts/build_report.py            # writes REPORT.md
```

### Driver flags

| Flag | Meaning |
|---|---|
| `--engines` | subset of `tensorsharp,llamacpp,vllm` |
| `--backends` | subset of `gpu,cpu` |
| `--models` | subset of the model ids in `config.json` |
| `--tasks` | subset of task ids (below) |
| `--quick` | smaller token counts for a smoke pass |
| `--skip-existing` | skip cells whose result JSON already exists |
| `--dry-run` | print the applicability plan and exit |
| `--config` / `--results` | override config path / output dir |

### Tasks

| Task | Kind | Measures |
|---|---|---|
| `pp512`, `pp2048` | synthetic | prefill (prompt-processing) throughput |
| `tg128` | synthetic | decode (token-generation) throughput |
| `short_text`, `long_text` | single-turn text | real prefill + decode throughput |
| `image`, `audio`, `video` | multimodal | multimodal prefill + decode throughput |
| `multi_turn` | conversational | 3-turn chat with KV-cache reuse, mean per-turn decode |
| `function_call` | agentic | tool/function call emitted? + decode throughput |
| `diffusion` | text diffusion | DiffusionGemma block-denoising throughput |

## How timing is measured

- **TensorSharp** — the CLI's own info-level timers (`--benchmark` for synthetic;
  `prefill complete` / `decode complete` log lines for real tasks;
  `--multi-turn-jsonl` per-turn lines; `--tools` for function-call;
  `--diffusion-steps` for diffusion). Model load and warm-up are excluded.
- **llama.cpp** — `llama-bench -r 3` for synthetic; `llama_perf_context_print`
  prompt-eval / eval timers from `llama-cli` / `llama-mtmd-cli`; `llama-server`
  via the shared OpenAI client for multi-turn / function-call.
- **vLLM** — streamed OpenAI chat completion: client-observed time-to-first-token
  is the prefill-latency proxy and `usage` token counts drive prefill/decode
  tokens-per-second.

GPU vs CPU is engine-specific: TensorSharp `--backend` id, llama.cpp `-ngl`
(999 = GPU, 0 = CPU), vLLM `--device` fixed at server launch.

## Applicability / skips

Cells are skipped (not failed) when they don't make sense:

- audio/video only run on models that declare those modalities (`gemma4`);
- `function_call` only on tool-capable models (`gemma4`, `qwen36`);
- `diffusion` only on `diffusiongemma`, which in turn runs *only* the diffusion
  task (TensorSharp-only — llama.cpp/vLLM have no text-diffusion path);
- vLLM has no audio/video path in this harness and needs an `hf_id` mapping;
- llama.cpp has no video CLI; multi-turn/function-call need `llama-server`.

Every skip records its reason in the cell JSON and the report (`skip`).

## CI

`.github/workflows/cross-engine-benchmark.yml` runs this on a self-hosted GPU
runner via `workflow_dispatch` (choose engines/backends/models/tasks, or run the
`--quick` smoke profile), uploads the JSON results + `REPORT.md` as artifacts,
and renders the report into the job summary.
