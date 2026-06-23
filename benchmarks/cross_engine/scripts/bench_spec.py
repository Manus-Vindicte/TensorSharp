#!/usr/bin/env python3
"""
Shared data model + configuration loading for the cross-engine benchmark.

Defines the model / task / result dataclasses, the task registry, and the
config loader that resolves machine-specific paths from config.json. Engine
runners (engines/*.py) and the orchestrator (run_benchmark.py) import from here.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]            # benchmarks/cross_engine
PROMPTS_DIR = ROOT / "prompts"
SCENARIOS_DIR = ROOT / "scenarios"
DEFAULT_RESULTS_DIR = ROOT / "results"

# Backends the matrix sweeps. Each engine maps these labels to its own concrete
# backend selector (TensorSharp backend id, llama.cpp -ngl value, vLLM device).
BACKENDS = ["gpu", "cpu"]


# ---------------------------------------------------------------------------
# Model registry entry (built from config.json)
# ---------------------------------------------------------------------------
@dataclass
class ModelSpec:
    short_id: str
    display: str
    family: str
    gguf: Optional[Path]
    mmproj: Optional[Path]
    hf_id: Optional[str]
    vllm_served_name: Optional[str]
    modalities: list[str]
    supports_tools: bool
    supports_thinking: bool
    diffusion: bool
    diffusion_steps: int = 48

    def supports(self, modality: str) -> bool:
        return modality in self.modalities


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------
@dataclass
class TaskSpec:
    short_id: str
    kind: str   # synthetic | text | image | audio | video | multi_turn | function_call | diffusion
    description: str
    pp: int = 0
    tg: int = 0
    prompt_file: Optional[str] = None      # under prompts/
    scenario_file: Optional[str] = None    # under scenarios/
    tools_file: Optional[str] = None       # under scenarios/
    max_tokens: int = 64


TASKS: dict[str, TaskSpec] = {
    # --- synthetic throughput (prompt-processing / token-generation) ---
    "pp512": TaskSpec("pp512", "synthetic", "Synthetic prefill, 512 tokens", pp=512, tg=0),
    "tg128": TaskSpec("tg128", "synthetic", "Synthetic decode, 128 tokens after 32-token prefill", pp=32, tg=128),
    "pp2048": TaskSpec("pp2048", "synthetic", "Synthetic prefill, 2048 tokens (long context)", pp=2048, tg=0),
    # --- single-turn text ---
    "short_text": TaskSpec("short_text", "text", "Single-turn short prompt -> 64 generated",
                           prompt_file="short_text.txt", max_tokens=64),
    "long_text": TaskSpec("long_text", "text", "Single-turn long prompt (~1k tokens) -> 64 generated",
                          prompt_file="long_text.txt", max_tokens=64),
    # --- multimodal ---
    "image": TaskSpec("image", "image", "Image + question -> 64 generated",
                      prompt_file="image_question.txt", max_tokens=64),
    "audio": TaskSpec("audio", "audio", "Audio clip + question -> 64 generated",
                      prompt_file="audio_question.txt", max_tokens=64),
    "video": TaskSpec("video", "video", "Video clip + question -> 64 generated",
                      prompt_file="video_question.txt", max_tokens=64),
    # --- conversational / agentic ---
    "multi_turn": TaskSpec("multi_turn", "multi_turn", "Multi-turn chat with KV-cache reuse (3 turns)",
                           scenario_file="multi_turn.jsonl", max_tokens=64),
    "function_call": TaskSpec("function_call", "function_call", "Single-turn tool / function call",
                              prompt_file="function_call.txt", tools_file="tools_weather.json",
                              max_tokens=128),
    # --- text diffusion (DiffusionGemma) ---
    "diffusion": TaskSpec("diffusion", "diffusion", "DiffusionGemma block denoising -> 128 tokens",
                          prompt_file="short_text.txt", max_tokens=128),
}

# Smaller token counts for a fast smoke run.
QUICK_OVERRIDES = {
    "pp512": dict(pp=128),
    "tg128": dict(tg=32),
    "pp2048": dict(pp=512),
    "short_text": dict(max_tokens=16),
    "long_text": dict(max_tokens=16),
    "image": dict(max_tokens=16),
    "audio": dict(max_tokens=16),
    "video": dict(max_tokens=16),
    "multi_turn": dict(max_tokens=16),
    "function_call": dict(max_tokens=32),
    "diffusion": dict(max_tokens=32),
}


# ---------------------------------------------------------------------------
# Result record (one per engine x backend x model x task cell)
# ---------------------------------------------------------------------------
@dataclass
class BenchResult:
    engine: str
    backend: str          # "gpu" | "cpu"
    model: str
    task: str
    ok: bool = False
    skipped: bool = False
    error: str = ""
    # throughput
    prefill_tokens: int = 0
    prefill_ms: float = 0.0
    prefill_tps: float = 0.0
    decode_tokens: int = 0
    decode_ms: float = 0.0
    decode_tps: float = 0.0
    # latency / scenario level
    ttft_ms: float = 0.0          # time to first token (client-observed, where available)
    e2e_ms: float = 0.0           # end-to-end generation latency
    total_wall_ms: float = 0.0
    model_load_ms: float = 0.0
    # scenario-specific
    turns: int = 0
    tool_call_ok: Optional[bool] = None
    diffusion_steps: int = 0
    output_text: str = ""
    cmd: str = ""
    raw_tail: str = ""

    def cell_id(self) -> str:
        return f"{self.engine}__{self.backend}__{self.model}__{self.task}"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class Config:
    repo_root: Path
    model_dir: Path
    data_dir: Path
    raw: dict
    models: dict[str, ModelSpec]

    def engine_cfg(self, engine: str) -> dict:
        return self.raw.get("engines", {}).get(engine, {})

    def media_path(self, modality: str) -> Optional[Path]:
        p = self.raw.get("media", {}).get(modality)
        return Path(p) if p else None


def _interp(value, vars_: dict[str, str]):
    if isinstance(value, str):
        for k, v in vars_.items():
            value = value.replace("{" + k + "}", v)
        return value
    if isinstance(value, dict):
        return {k: _interp(v, vars_) for k, v in value.items()}
    if isinstance(value, list):
        return [_interp(v, vars_) for v in value]
    return value


def load_config(path: Optional[Path] = None) -> Config:
    """Load config.json (falling back to config.example.json) and resolve paths."""
    if path is None:
        cand = ROOT / "config.json"
        path = cand if cand.exists() else ROOT / "config.example.json"
    raw = json.loads(Path(path).read_text())

    repo_root = raw.get("repo_root", str(ROOT.parents[1]))
    model_dir = raw.get("model_dir", "")
    # data_dir may reference {repo_root}
    data_dir = _interp(raw.get("data_dir", "{repo_root}/data"), {"repo_root": repo_root})

    vars_ = {"repo_root": repo_root, "model_dir": model_dir, "data_dir": data_dir}
    raw = _interp(raw, vars_)

    models: dict[str, ModelSpec] = {}
    for short_id, m in raw.get("models", {}).items():
        models[short_id] = ModelSpec(
            short_id=short_id,
            display=m.get("display", short_id),
            family=m.get("family", short_id),
            gguf=Path(m["gguf"]) if m.get("gguf") else None,
            mmproj=Path(m["mmproj"]) if m.get("mmproj") else None,
            hf_id=m.get("hf_id"),
            vllm_served_name=m.get("vllm_served_name"),
            modalities=list(m.get("modalities", ["text"])),
            supports_tools=bool(m.get("supports_tools", False)),
            supports_thinking=bool(m.get("supports_thinking", False)),
            diffusion=bool(m.get("diffusion", False)),
            diffusion_steps=int(m.get("diffusion_steps", 48)),
        )

    return Config(
        repo_root=Path(repo_root),
        model_dir=Path(model_dir),
        data_dir=Path(data_dir),
        raw=raw,
        models=models,
    )


# ---------------------------------------------------------------------------
# Applicability: which engine x backend x model x task cells make sense
# ---------------------------------------------------------------------------
def applies(engine: str, backend: str, model: ModelSpec, task: TaskSpec) -> tuple[bool, str]:
    # Modality gating against the model's declared capabilities.
    if task.kind in ("image", "audio", "video") and not model.supports(task.kind):
        return False, f"{model.short_id} has no {task.kind} support"
    if task.kind == "function_call" and not model.supports_tools:
        return False, f"{model.short_id} has no tool-calling support"
    if task.kind == "diffusion" and not model.diffusion:
        return False, f"{model.short_id} is not a diffusion model"
    if task.kind != "diffusion" and model.diffusion:
        return False, f"{model.short_id} only runs the diffusion task"

    # Engine capabilities.
    if engine == "tensorsharp":
        return True, ""

    if engine == "llamacpp":
        if task.kind == "video":
            return False, "llama.cpp has no video CLI path"
        if task.kind == "audio" and model.mmproj is None:
            return False, "no audio projector for llama-mtmd-cli"
        if model.diffusion:
            return False, "llama.cpp has no text-diffusion path"
        return True, ""

    if engine == "vllm":
        if model.hf_id is None or model.vllm_served_name is None:
            return False, f"{model.short_id} has no vLLM (HF) mapping"
        if task.kind in ("audio", "video"):
            return False, "vLLM path here covers text/image/tools only"
        if model.diffusion:
            return False, "vLLM has no text-diffusion path"
        return True, ""

    return False, f"unknown engine {engine}"
