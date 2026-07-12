#!/usr/bin/env python3
"""Real MLX VLM/audio smoke matrix for disposable PR CI.

Each candidate runs in an isolated subprocess with its own HF_HOME. The parent
removes that cache immediately after the candidate finishes, so the workflow can
try several model families without accumulating model weights on disk.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path


GiB = 1024 ** 3


@dataclass(frozen=True)
class Candidate:
    family: str
    repo: str
    modality: str = "image"
    shard: str = "small"
    required: bool = False


class MultimodalExecutionError(RuntimeError):
    def __init__(self, stage: str, original: BaseException):
        super().__init__(f"{stage}: {original}")
        self.stage = stage


CANDIDATES = [
    Candidate("idefics3-smolvlm", "mlx-community/SmolVLM-256M-Instruct-4bit", required=True),
    Candidate("qwen2-vl", "mlx-community/Qwen2-VL-2B-Instruct-4bit", required=True),
    Candidate("qwen2-vl", "mlx-community/Qwen2-VL-2B-Instruct-4bit", modality="video", required=True),
    Candidate("qwen3-vl", "unsloth/Qwen3-VL-2B-Instruct", required=True),
    Candidate("qwen3-vl", "unsloth/Qwen3-VL-2B-Instruct", modality="video", required=True),
    Candidate("qwen3.5-vl", "unsloth/Qwen3.5-0.8B"),
    Candidate("qwen3.5-vl", "unsloth/Qwen3.5-0.8B", modality="video"),
    Candidate("lfm2-vl", "mlx-community/LFM2.5-VL-1.6B-4bit", shard="small"),
    Candidate("jina-vlm", "jinaai/jina-vlm-mlx", shard="small"),
    Candidate("internvl-chat", "mlx-community/InternVL3-1B-4bit", shard="small"),
    Candidate("fastvlm", "mlx-community/FastVLM-0.5B-bf16", shard="small"),
    Candidate("gemma4", "unsloth/gemma-4-E2B-it-UD-MLX-4bit", shard="medium", required=True),
    Candidate("gemma4", "unsloth/gemma-4-E2B-it-UD-MLX-4bit", modality="video", shard="medium", required=True),
    Candidate("minicpm-v4.6", "mlx-community/MiniCPM-V-4.6-mxfp4", shard="medium"),
    Candidate("glm4v", "mlx-community/GLM-4.6V-Flash-mxfp4", shard="medium"),
    Candidate("deepseek-vl-v2", "mlx-community/deepseek-vl2-tiny-4bit", shard="medium"),
    Candidate("phi3-v", "mlx-community/Phi-3.5-vision-instruct-4bit", shard="medium"),
    Candidate("granite-vision", "mlx-community/granite-vision-3.2-2b-4bit", shard="medium"),
    Candidate("qwen2.5-vl", "mlx-community/Qwen2.5-VL-3B-Instruct-4bit", shard="medium"),
    Candidate("qwen2.5-vl", "mlx-community/Qwen2.5-VL-3B-Instruct-4bit", modality="video", shard="medium"),
    Candidate("gemma3", "mlx-community/gemma-3-4b-it-4bit", shard="medium"),
    Candidate("llava", "mlx-community/llava-1.5-7b-4bit", shard="large-a"),
    Candidate("llava-next", "mlx-community/llava-v1.6-mistral-7b-4bit", shard="large-a"),
    Candidate("llava-bunny", "mlx-community/Bunny-Llama-3-8B-V-4bit", shard="large-a"),
    Candidate("gemma3n", "mlx-community/gemma-3n-E2B-it-4bit", shard="large-a"),
    Candidate("idefics2", "mlx-community/idefics2-8b-4bit", shard="large-a"),
    Candidate("molmo", "mlx-community/Molmo-7B-D-0924-4bit", shard="large-a"),
    Candidate("aya-vision", "mlx-community/aya-vision-8b-4bit", shard="large-b"),
    Candidate("mllama", "mlx-community/Llama-3.2-11B-Vision-Instruct-4bit", shard="large-b"),
    Candidate("zaya1-vl", "OsaurusAI/ZAYA1-VL-8B-MXFP4", shard="large-b"),
    Candidate("youtu-vl", "tencent/Youtu-VL-4B-Instruct", shard="large-b"),
    Candidate("pixtral", "mlx-community/pixtral-12b-4bit", shard="large-b"),
    Candidate("molmo2", "mlx-community/Molmo2-8B-4bit", shard="large-b"),
    Candidate("kimi-vl", "mlx-community/Kimi-VL-A3B-Thinking-4bit", shard="large-b"),
    Candidate("deepseek-vl-v2-small", "mlx-community/deepseek-vl2-small-4bit", shard="large-b"),
    Candidate("mistral3-vl", "mlx-community/Mistral-Small-3.1-24B-Instruct-2503-4bit", shard="large-b"),
    Candidate("qwen3.6-moe", "unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit", shard="moe"),
    Candidate("qwen3-vl-moe", "mlx-community/Qwen3-VL-30B-A3B-Instruct-3bit", shard="moe"),
    Candidate("llama4", "mlx-community/Llama-4-Scout-17B-16E-Instruct-4bit", shard="moe"),
    Candidate("step3p7", "mlx-community/Step-3.7-Flash-4bit", shard="moe"),
    Candidate("ernie4.5-moe-vl", "mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit", shard="moe"),
    Candidate("phi4mm", "Ferox-AI/Phi-4-multimodal-instruct-mlx-4bit", modality="audio", shard="audio"),
    Candidate("gemma3n", "mlx-community/gemma-3n-E2B-it-4bit", modality="audio", shard="audio"),
    Candidate("gemma4", "unsloth/gemma-4-E2B-it-UD-MLX-4bit", modality="audio", shard="audio"),
    Candidate("minicpmo", "mlx-community/MiniCPM-o-4_5-4bit", modality="audio", shard="audio"),
    Candidate("qwen2-audio", "mlx-community/Qwen2-Audio-7B-Instruct-4bit", modality="audio", shard="audio"),
    Candidate("qwen2.5-omni", "giangndm/qwen2.5-omni-3b-mlx-4bit", modality="audio", shard="audio"),
    Candidate("nemotron-h-nano-omni", "mlx-community/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-mxfp4", modality="audio", shard="audio"),
    Candidate("qwen3-omni", "abnormalmapstudio/Qwen3-Omni-30B-A3B-Instruct-mxfp4-mlx", modality="audio", shard="audio"),
]


SHARD_POLICIES = {
    "small": (8.0, 2.0, False),
    "medium": (12.0, 1.7, False),
    "large-a": (18.0, 1.35, True),
    "large-b": (26.0, 1.25, True),
    "moe": (40.0, 1.15, True),
    "audio": (40.0, 1.2, True),
}


def _model_key(candidate: Candidate) -> str:
    return re.sub(r"[^a-z0-9]+", "-", candidate.family.lower()).strip("-")


def _model_groups() -> dict[str, list[Candidate]]:
    groups: dict[str, list[Candidate]] = {}
    keys: dict[str, str] = {}
    for candidate in CANDIDATES:
        key = _model_key(candidate)
        existing_repo = keys.setdefault(key, candidate.repo)
        if existing_repo != candidate.repo:
            raise ValueError(
                f"Model key {key!r} maps to both {existing_repo!r} and "
                f"{candidate.repo!r}."
            )
        groups.setdefault(candidate.repo, []).append(candidate)
    return groups


def _model_matrix(model_keys: str | None = None) -> dict:
    selected = {
        key.strip()
        for key in str(model_keys or "").split(",")
        if key.strip()
    }
    include = []
    for repo, candidates in _model_groups().items():
        if selected and _model_key(candidates[0]) not in selected:
            continue
        policies = [SHARD_POLICIES[candidate.shard] for candidate in candidates]
        modalities = list(dict.fromkeys(candidate.modality for candidate in candidates))
        include.append(
            {
                "model_key": _model_key(candidates[0]),
                "family": candidates[0].family,
                "repo": repo,
                "modalities": "+".join(modalities),
                "max_model_gb": f"{max(policy[0] for policy in policies):.1f}",
                "disk_multiplier": f"{max(policy[1] for policy in policies):.2f}",
                "large_models": str(any(policy[2] for policy in policies)).lower(),
                "min_real_passes": "1" if any(candidate.required for candidate in candidates) else "0",
                "total_budget_s": str(300 + 600 * len(candidates)),
                "timeout_minutes": 15 + 10 * len(candidates),
            }
        )
    found = {model["model_key"] for model in include}
    unknown = sorted(selected - found)
    if unknown:
        raise ValueError(f"Unknown model keys: {unknown}")
    return {"include": include}


def _print_group(title: str, body: str) -> None:
    print(f"::group::{title}")
    print(body.rstrip())
    print("::endgroup::")


def _stage(name: str) -> None:
    print(f"STAGE:{name}", flush=True)


def _last_stage(output: str) -> str:
    matches = re.findall(r"^STAGE:([^\r\n]+)", output, flags=re.MULTILINE)
    return matches[-1].strip() if matches else "unknown"


def _candidate_size_bytes(repo: str) -> int | None:
    from huggingface_hub import HfApi

    info = HfApi().model_info(repo, files_metadata=True)
    sizes = [getattr(sibling, "size", None) or 0 for sibling in info.siblings]
    total = sum(sizes)
    return total if total > 0 else None


def _free_gib(path: Path) -> float:
    return shutil.disk_usage(path).free / GiB


def _is_resource_failure(returncode: int | None, output: str) -> bool:
    if returncode in {-9, 137, 247}:
        return True
    lowered = output.lower()
    return any(
        marker in lowered
        for marker in (
            "out of memory",
            "cannot allocate memory",
            "unable to allocate",
            "[malloc]",
            "no space left on device",
            "killed: 9",
            "signal 9",
            "metal command buffer",
            "gpu timeout error",
            "kiogpucommandbuffercallbackerrortimeout",
            "causing prior/excessive gpu errors",
            "resource exhausted",
        )
    )


def _is_network_failure(output: str) -> bool:
    lowered = output.lower()
    return any(
        marker in lowered
        for marker in (
            "read operation timed out",
            "connect timeout",
            "connection error",
            "connection reset",
            "temporary failure in name resolution",
            "failed to establish a new connection",
            "max retries exceeded",
            "http 429",
            "http 5",
        )
    )


def _missing_dependency(output: str) -> str | None:
    match = re.search(
        r"(?:ModuleNotFoundError|ImportError): No module named ['\"]([^'\"]+)",
        output,
    )
    if match is None or match.group(1).startswith("mlx_vlm.models."):
        return None
    return match.group(1)


def _is_explicitly_unsupported(output: str) -> bool:
    lowered = output.lower()
    return any(
        marker in lowered
        for marker in (
            "model type ",
            "no module named 'mlx_vlm.models.",
            " parameters not in model",
            "unrecognized processing class",
            "contains custom code which must be executed",
            "installed mlx-lm / mlx-vlm rejects",
            "cannot load mlx ",
        )
    )


def _classify_dual_path_failure(multimodal_error: str, text_error: str) -> tuple[str, str]:
    combined = f"{multimodal_error}\n{text_error}"
    if _is_resource_failure(None, combined):
        return "skipped-resource", "model could not run within runner resources"
    if _is_network_failure(combined):
        return "skipped-network", "model failed because network access was unavailable"
    dependency = _missing_dependency(combined)
    if dependency is not None:
        return "unsupported-dependency", f"release environment is missing {dependency!r}"
    if _is_explicitly_unsupported(combined):
        return "unsupported-yet", "installed MLX runtime explicitly rejects this model"
    return "failed-load", "both text and multimodal loading failed unexpectedly"


def _as_text(part) -> str:
    if part is None:
        return ""
    if isinstance(part, bytes):
        return part.decode("utf-8", errors="replace")
    return str(part)


def _combined_output(*parts) -> str:
    return "\n".join(text for text in (_as_text(part) for part in parts) if text)


def _failure_reason(returncode: int | None, output: str) -> str:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    detail = lines[-1] if lines else ""
    if detail:
        return f"returncode={returncode}: {detail[:240]}"
    return f"returncode={returncode}"


def _run_child(
    candidate: Candidate,
    timeout_s: int,
    hf_home: Path | None = None,
) -> dict:
    owns_hf_home = hf_home is None
    if hf_home is None:
        hf_home = Path(tempfile.mkdtemp(prefix="unsloth_mlx_real_hf_"))
    else:
        hf_home.mkdir(parents=True, exist_ok=True)
    payload = base64.b64encode(json.dumps(asdict(candidate)).encode()).decode()
    env = os.environ.copy()
    env.update(
        {
            "HF_HOME": str(hf_home),
            "HF_HUB_OFFLINE": "1",
            "HF_HUB_DISABLE_XET": "1",
            "HF_HUB_ENABLE_HF_TRANSFER": "0",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
        }
    )
    started = time.monotonic()
    try:
        completed = subprocess.run(
            [sys.executable, __file__, "--child", payload],
            cwd=Path(__file__).resolve().parents[2],
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout_s,
        )
        elapsed = time.monotonic() - started
        output = _combined_output(completed.stdout, completed.stderr)
        _print_group(f"{candidate.family}: child output", output)
        match = re.search(r"RESULT_JSON:(\{.*\})", output)
        if completed.returncode == 0 and match:
            result = json.loads(match.group(1))
            result.setdefault("status", "passed")
            result["elapsed_s"] = round(elapsed, 1)
            return result
        if _is_resource_failure(completed.returncode, output):
            return {
                "family": candidate.family,
                "repo": candidate.repo,
                "modality": candidate.modality,
                "status": "skipped-resource",
                "reason": f"resource failure, returncode={completed.returncode}",
                "elapsed_s": round(elapsed, 1),
            }
        return {
            "family": candidate.family,
            "repo": candidate.repo,
            "modality": candidate.modality,
            "status": "failed",
            "reason": _failure_reason(completed.returncode, output),
            "elapsed_s": round(elapsed, 1),
        }
    except subprocess.TimeoutExpired as exc:
        output = _combined_output(exc.stdout, exc.stderr)
        _print_group(f"{candidate.family}: timeout output", output)
        stage = _last_stage(output)
        if _is_network_failure(output):
            status = "skipped-network"
        elif _is_resource_failure(None, output):
            status = "skipped-resource"
        else:
            status = "skipped-timeout"
        return {
            "family": candidate.family,
            "repo": candidate.repo,
            "modality": candidate.modality,
            "status": status,
            "reason": f"timeout after {timeout_s}s during {stage}",
            "last_stage": stage,
            "elapsed_s": timeout_s,
        }
    finally:
        if owns_hf_home:
            shutil.rmtree(hf_home, ignore_errors=True)


def _download_candidate(repo: str, hf_home: Path, timeout_s: int) -> dict:
    env = os.environ.copy()
    env.pop("HF_HUB_OFFLINE", None)
    env.update(
        {
            "HF_HOME": str(hf_home),
            "HF_HUB_DISABLE_XET": "1",
            "HF_HUB_ENABLE_HF_TRANSFER": "0",
            "HF_HUB_DOWNLOAD_TIMEOUT": "120",
            "PYTHONUNBUFFERED": "1",
        }
    )
    started = time.monotonic()
    try:
        completed = subprocess.run(
            [
                sys.executable,
                __file__,
                "--download",
                repo,
                "--download-cache",
                str(hf_home / "hub"),
            ],
            cwd=Path(__file__).resolve().parents[2],
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        output = _combined_output(exc.stdout, exc.stderr)
        _print_group(f"{repo}: download timeout", output)
        return {
            "status": "skipped-network",
            "reason": f"download timeout after {timeout_s}s",
            "download_elapsed_s": timeout_s,
        }

    elapsed = round(time.monotonic() - started, 1)
    output = _combined_output(completed.stdout, completed.stderr)
    _print_group(f"{repo}: download output", output)
    if completed.returncode == 0:
        return {"status": "downloaded", "download_elapsed_s": elapsed}
    status = "skipped-network" if _is_network_failure(output) else "skipped-download"
    return {
        "status": status,
        "reason": _failure_reason(completed.returncode, output),
        "download_elapsed_s": elapsed,
    }


def _child_prompt(processor, config, modality: str):
    from mlx_vlm import prompt_utils

    model_type = getattr(config, "model_type", None)
    if isinstance(config, dict):
        model_type = config.get("model_type")
    model_type = str(model_type or "").lower()

    if modality == "audio":
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio"},
                    {"type": "text", "text": "Describe this short tone in a detailed paragraph of about 128 tokens."},
                ],
            }
        ]
        try:
            return prompt_utils.apply_chat_template(
                processor,
                config,
                messages,
                add_generation_prompt=True,
                num_audios=1,
            )
        except Exception:
            audio_token = (
                getattr(processor, "audio_token", None)
                or getattr(getattr(processor, "tokenizer", None), "audio_token", None)
                or "<audio>"
            )
            return f"{audio_token}\nDescribe this short tone in a detailed paragraph of about 128 tokens."

    if modality == "video":
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": "dummy.mp4", "max_pixels": 224 * 224, "fps": 1},
                    {"type": "text", "text": "Describe this short video in a detailed paragraph of about 128 tokens."},
                ],
            }
        ]
        if hasattr(processor, "apply_chat_template"):
            try:
                return processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass
        try:
            return prompt_utils.apply_chat_template(
                processor,
                config,
                messages,
                add_generation_prompt=True,
                video=True,
            )
        except Exception:
            video_token = (
                getattr(processor, "video_token", None)
                or getattr(getattr(processor, "tokenizer", None), "video_token", None)
                or "<video>"
            )
            return f"{video_token}\nDescribe this short video in a detailed paragraph of about 128 tokens."

    if "deepseek_vl" in model_type:
        return "<image>\nDescribe this image in a detailed paragraph of about 128 tokens."

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe this image in a detailed paragraph of about 128 tokens."},
            ],
        }
    ]
    try:
        return prompt_utils.apply_chat_template(
            processor,
            config,
            messages,
            add_generation_prompt=True,
            num_images=1,
        )
    except Exception:
        image_token = (
            getattr(processor, "image_token", None)
            or getattr(getattr(processor, "tokenizer", None), "image_token", None)
            or "<image>"
        )
        if "llava" in model_type:
            return f"USER: {image_token}\nDescribe this image in a detailed paragraph of about 128 tokens.\nASSISTANT:"
        return f"{image_token}\nDescribe this image in a detailed paragraph of about 128 tokens."


def _text_prompt(tokenizer) -> str:
    return "Write a detailed paragraph of about 128 tokens about validating model inference."


def _short_exception(exc: BaseException) -> str:
    lines = traceback.format_exception_only(type(exc), exc)
    return " ".join(line.strip() for line in lines if line.strip())[:300]


def _markdown_cell(value) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("|", "\\|")
    if len(text) > 220:
        text = text[:217].rstrip() + "..."
    return text


def _peak_memory_gb(mx) -> float:
    if hasattr(mx.metal, "get_peak_memory"):
        return round(float(mx.metal.get_peak_memory() or 0.0) / GiB, 3)
    return 0.0


def _common_summary(candidate: Candidate, processor, mx, **extra) -> dict:
    studio_shape = bool(extra.pop("studio_shape", False))
    summary = {
        "family": candidate.family,
        "repo": candidate.repo,
        "modality": candidate.modality,
        "shard": candidate.shard,
        "studio_shape": studio_shape,
        "processor": f"{processor.__class__.__module__}.{processor.__class__.__name__}",
        "image_processor": (
            None
            if getattr(processor, "image_processor", None) is None
            else f"{processor.image_processor.__class__.__module__}.{processor.image_processor.__class__.__name__}"
        ),
        "feature_extractor": (
            None
            if getattr(processor, "feature_extractor", None) is None
            else f"{processor.feature_extractor.__class__.__module__}.{processor.feature_extractor.__class__.__name__}"
        ),
        "video_processor": (
            None
            if getattr(processor, "video_processor", None) is None
            else f"{processor.video_processor.__class__.__module__}.{processor.video_processor.__class__.__name__}"
        ),
        "peak_memory_gb": _peak_memory_gb(mx),
        "requested_max_tokens": int(os.environ.get("UNSLOTH_MLX_REAL_MAX_TOKENS", "128")),
    }
    summary.update(extra)
    return summary


def _shape_list(value) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return [int(dim) for dim in shape]


def _input_keys(inputs) -> list[str]:
    if hasattr(inputs, "keys"):
        return sorted(str(key) for key in inputs.keys())
    return []


def _tone_array():
    import numpy as np

    sample_rate = 16000
    t = np.arange(sample_rate // 2, dtype=np.float32) / sample_rate
    audio = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    return audio, sample_rate


def _video_frames():
    import numpy as np

    height = width = 96
    frames = np.stack(
        [
            np.full((height, width, 3), (80, 120, 210), dtype=np.uint8),
            np.full((height, width, 3), (210, 120, 80), dtype=np.uint8),
        ],
        axis=0,
    ).transpose(0, 3, 1, 2)
    return np.ascontiguousarray(frames)


def _studio_user_text(candidate: Candidate) -> str:
    if candidate.modality == "audio":
        return "Please transcribe this short tone in a detailed paragraph of about 128 tokens."
    if candidate.modality == "video":
        return "Describe this short video in a detailed paragraph of about 128 tokens."
    if candidate.family == "qwen3-vl":
        return "Describe this image in detail. " + ("token " * 1200)
    return "Describe this image in a detailed paragraph of about 128 tokens."


def _studio_prompt(processor, config, candidate: Candidate, user_text: str) -> str:
    if candidate.modality == "text":
        messages = [{"role": "user", "content": user_text}]
    else:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": candidate.modality},
                    {"type": "text", "text": user_text},
                ],
            }
        ]
    chat_target = processor
    if (
        getattr(processor, "apply_chat_template", None) is None
        or getattr(processor, "chat_template", None) is None
    ):
        chat_target = getattr(processor, "tokenizer", processor)
    if hasattr(chat_target, "apply_chat_template"):
        try:
            return chat_target.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass
    if candidate.modality == "text":
        return user_text
    return _child_prompt(processor, config, candidate.modality)


def _first_processor_success(attempts: list[tuple[str, object]]):
    errors = []
    for label, call in attempts:
        try:
            inputs = call()
        except Exception as exc:
            errors.append(f"{label}: {_short_exception(exc)}")
            continue
        if not hasattr(inputs, "keys"):
            errors.append(f"{label}: returned {type(inputs).__name__}, expected mapping")
            continue
        return inputs, label
    raise RuntimeError("all Studio-shaped processor calls failed; " + "; ".join(errors[:5]))


def _build_studio_video_inputs(candidate: Candidate, processor, config):
    prompt = _studio_prompt(
        processor,
        config,
        candidate,
        _studio_user_text(candidate),
    )
    frames = _video_frames()
    attempts = [
        (
            "processor(text,videos)",
            lambda: processor(
                text=[prompt],
                videos=[frames],
                padding=True,
                return_tensors="pt",
            ),
        ),
        (
            "processor(prompt,videos)",
            lambda: processor(
                prompt,
                videos=[frames],
                return_tensors="pt",
            ),
        ),
    ]
    inputs, processor_call = _first_processor_success(attempts)
    return inputs, {"processor_call": processor_call, "video_frames": len(frames)}


def _to_mlx(value, mx):
    if isinstance(value, mx.array):
        return value
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return mx.array(value.detach().cpu().numpy())
    except ImportError:
        pass
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return mx.array(value)
        if isinstance(value, (list, tuple)) and value and not isinstance(
            value[0], (str, bytes, dict)
        ):
            return mx.array(np.asarray(value))
    except (TypeError, ValueError):
        pass
    return value


def _run_studio_shape_generation(
    candidate: Candidate,
    model,
    processor,
    mx,
    tmp_dir: Path,
) -> dict:
    from PIL import Image
    from mlx_vlm import generate as vlm_generate

    max_tokens = int(os.environ.get("UNSLOTH_MLX_REAL_MAX_TOKENS", "128"))
    config = getattr(model, "config", getattr(model, "_config", {}))
    prompt = _studio_prompt(
        processor,
        config,
        candidate,
        _studio_user_text(candidate),
    )
    metadata = {"processor_call": "mlx_vlm.generate"}
    generation_kwargs = {
        "max_tokens": max_tokens,
        "verbose": False,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "min_p": 0.0,
    }

    _stage(f"{candidate.modality}-preprocess")
    if candidate.modality == "image":
        image_size = (1024, 1024) if candidate.family == "qwen3-vl" else (224, 224)
        image = Image.new("RGB", image_size, (80, 120, 210))
        metadata["image_size"] = f"{image_size[0]}x{image_size[1]}"
        _stage("image-generate")
        result = vlm_generate(
            model,
            processor,
            prompt,
            image=[image],
            **generation_kwargs,
        )
    elif candidate.modality == "audio":
        import soundfile as sf

        audio, sample_rate = _tone_array()
        audio_path = tmp_dir / "tone.wav"
        sf.write(audio_path, audio, sample_rate)
        metadata["audio_samples"] = len(audio)
        _stage("audio-generate")
        result = vlm_generate(
            model,
            processor,
            prompt,
            audio=[str(audio_path)],
            **generation_kwargs,
        )
    else:
        inputs, video_metadata = _build_studio_video_inputs(
            candidate, processor, config
        )
        metadata.update(video_metadata)
        keys = _input_keys(inputs)
        if "input_ids" not in inputs:
            raise RuntimeError(
                f"Studio-shaped processor call did not return input_ids: keys={keys}"
            )
        prepared = {key: _to_mlx(value, mx) for key, value in inputs.items()}
        input_ids = prepared.pop("input_ids")
        pixel_values = prepared.pop("pixel_values", None)
        mask = prepared.pop("attention_mask", None)
        metadata.update(
            input_keys=",".join(keys[:8]),
            input_shape=str(_shape_list(input_ids)),
        )
        _stage("video-generate")
        result = vlm_generate(
            model,
            processor,
            prompt,
            input_ids=input_ids,
            pixel_values=pixel_values,
            mask=mask,
            **prepared,
            **generation_kwargs,
        )

    prompt_tokens = int(getattr(result, "prompt_tokens", 0) or 0)
    generation_tokens = int(getattr(result, "generation_tokens", 0) or 0)
    if generation_tokens <= 0:
        raise RuntimeError("Studio-shaped generation produced no new tokens")
    if (
        candidate.family == "qwen3-vl"
        and candidate.modality == "image"
        and prompt_tokens <= 2048
    ):
        raise RuntimeError(
            f"Studio-shaped prompt did not cross prefill chunk boundary: {prompt_tokens}"
        )

    return _common_summary(
        candidate,
        processor,
        mx,
        status="passed",
        studio_shape=True,
        prompt_tokens=prompt_tokens,
        generation_tokens=generation_tokens,
        text_head=str(getattr(result, "text", ""))[:80],
        **metadata,
    )


def _run_multimodal_generation(candidate: Candidate, tmp_dir: Path) -> dict:
    import gc
    import mlx.core as mx
    from unsloth_zoo.mlx.loader import FastMLXModel

    model = processor = None
    stage = "multimodal-load"
    try:
        if hasattr(mx.metal, "reset_peak_memory"):
            mx.metal.reset_peak_memory()

        _stage(stage)
        model, processor = FastMLXModel.from_pretrained(
            candidate.repo,
            text_only=False,
            max_seq_length=2048,
        )
        stage = "Studio-shaped generation"
        return _run_studio_shape_generation(
            candidate, model, processor, mx, tmp_dir
        )
    except Exception as exc:
        if stage != "multimodal-load":
            raise MultimodalExecutionError(stage, exc) from exc
        raise
    finally:
        del model, processor
        gc.collect()
        if "mx" in locals() and hasattr(mx.metal, "clear_cache"):
            mx.metal.clear_cache()


def _run_text_only_generation(candidate: Candidate) -> dict:
    import gc
    import mlx.core as mx
    from mlx_vlm import generate as vlm_generate
    from unsloth_zoo.mlx.loader import FastMLXModel

    model = processor = None
    try:
        if hasattr(mx.metal, "reset_peak_memory"):
            mx.metal.reset_peak_memory()

        _stage("text-load")
        model, processor = FastMLXModel.from_pretrained(
            candidate.repo,
            text_only=False,
            max_seq_length=2048,
        )
        config = getattr(model, "config", getattr(model, "_config", {}))
        text_candidate = Candidate(
            candidate.family,
            candidate.repo,
            modality="text",
            shard=candidate.shard,
            required=candidate.required,
        )
        prompt = _studio_prompt(
            processor,
            config,
            text_candidate,
            _text_prompt(processor),
        )
        max_tokens = int(os.environ.get("UNSLOTH_MLX_REAL_MAX_TOKENS", "128"))
        _stage("text-generate")
        result = vlm_generate(
            model,
            processor,
            prompt,
            max_tokens=max_tokens,
            temperature=0.0,
            verbose=False,
        )
        text = str(getattr(result, "text", ""))
        generation_tokens = int(getattr(result, "generation_tokens", 0) or 0)
        if generation_tokens <= 0:
            raise RuntimeError("pure-text VLM generation produced no tokens")
        return _common_summary(
            candidate,
            processor,
            mx,
            text_only_generation="passed",
            generation_tokens=generation_tokens,
            text_head=str(text)[:80],
        )
    finally:
        del model, processor
        gc.collect()
        if "mx" in locals() and hasattr(mx.metal, "clear_cache"):
            mx.metal.clear_cache()


def _child_main(payload: str) -> int:
    data = json.loads(base64.b64decode(payload.encode()).decode())
    candidate = Candidate(**data)
    tmp_dir = Path(tempfile.mkdtemp(prefix="unsloth_mlx_real_case_"))
    try:
        try:
            summary = _run_multimodal_generation(candidate, tmp_dir)
        except Exception as multimodal_exc:
            multimodal_error = _short_exception(multimodal_exc)
            print(f"Multimodal path failed for {candidate.family}: {multimodal_error}")
            if _is_resource_failure(None, multimodal_error):
                summary = {
                    "family": candidate.family,
                    "repo": candidate.repo,
                    "modality": candidate.modality,
                    "shard": candidate.shard,
                    "status": "skipped-resource",
                    "reason": f"multimodal resource failure: {multimodal_error}",
                }
            else:
                try:
                    text_summary = _run_text_only_generation(candidate)
                except Exception as text_exc:
                    text_error = _short_exception(text_exc)
                    status, classification = _classify_dual_path_failure(
                        multimodal_error,
                        text_error,
                    )
                    summary = {
                        "family": candidate.family,
                        "repo": candidate.repo,
                        "modality": candidate.modality,
                        "shard": candidate.shard,
                        "status": status,
                        "reason": (
                            f"{classification}; text-only path also failed; "
                            f"multimodal_error={multimodal_error}; "
                            f"text_error={text_error}"
                        ),
                    }
                else:
                    summary = dict(text_summary)
                    summary.update(
                        status="failed",
                        reason=(
                            f"text-only path passed but {candidate.modality} path failed: "
                            f"{multimodal_error}"
                        ),
                    )
        print("RESULT_JSON:" + json.dumps(summary, sort_keys=True))
        return 0
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _summary_markdown(results: list[dict]) -> str:
    counts = {}
    for result in results:
        status = result.get("status", "")
        counts[status] = counts.get(status, 0) + 1
    count_text = ", ".join(f"{status}={count}" for status, count in sorted(counts.items()))
    rows = [
        f"Status counts: {count_text or 'none'}",
        "",
        "| Shard | Family | Repo | Modality | Status | Tokens | Peak | Processor | Detail |",
        "| --- | --- | --- | --- | --- | ---: | ---: | --- | --- |",
    ]
    for result in results:
        detail = result.get("reason")
        if result.get("status") == "passed":
            parts = [
                f"requested_max_tokens={result.get('requested_max_tokens')}",
                f"prompt_tokens={result.get('prompt_tokens')}",
                f"download_s={result.get('download_elapsed_s')}",
            ]
            if result.get("studio_shape"):
                parts.append(
                    "studio_shape=True "
                    f"processor_call={result.get('processor_call')} "
                    f"input_shape={result.get('input_shape')} "
                    f"text_head={result.get('text_head', '')!r}"
                )
            else:
                parts.append(f"text_head={result.get('text_head', '')!r}")
            detail = " ".join(parts)
        rows.append(
            "| {shard} | {family} | `{repo}` | {modality} | {status} | {tokens} | {peak} | {processor} | {detail} |".format(
                shard=_markdown_cell(result.get("shard", "")),
                family=_markdown_cell(result.get("family", "")),
                repo=_markdown_cell(result.get("repo", "")),
                modality=_markdown_cell(result.get("modality", "")),
                status=_markdown_cell(result.get("status", "")),
                tokens=_markdown_cell(result.get("generation_tokens", "")),
                peak=_markdown_cell(result.get("peak_memory_gb", "")),
                processor=_markdown_cell(result.get("processor", "")),
                detail=_markdown_cell(detail),
            )
        )
    return "\n".join(rows) + "\n"


def _write_results_artifact(
    model_key: str | None,
    candidates: list[Candidate],
    results: list[dict],
) -> None:
    output_path = os.environ.get("UNSLOTH_MLX_REAL_RESULTS_PATH")
    if not output_path:
        return
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_key": model_key,
        "repo": candidates[0].repo if candidates else None,
        "required": any(candidate.required for candidate in candidates),
        "results": results,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _aggregate_results_main(results_dir: str) -> int:
    expected = json.loads(os.environ.get("UNSLOTH_MLX_REAL_EXPECTED_MATRIX", "{}"))
    expected_models = expected.get("include", [])
    payloads = {}
    malformed = []
    for path in sorted(Path(results_dir).rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            model_key = payload["model_key"]
            if not model_key or model_key in payloads:
                raise ValueError(f"duplicate or empty model_key={model_key!r}")
            payloads[model_key] = payload
        except Exception as exc:
            malformed.append((path, exc))

    results = []
    required_without_pass = []
    for model in expected_models:
        model_key = model["model_key"]
        payload = payloads.pop(model_key, None)
        if payload is None:
            results.append(
                {
                    "family": model["family"],
                    "repo": model["repo"],
                    "modality": model["modalities"],
                    "shard": "matrix",
                    "status": "failed-infrastructure",
                    "reason": "model job did not upload a result artifact",
                }
            )
            continue
        model_results = payload.get("results", [])
        results.extend(model_results)
        if (
            model.get("min_real_passes") == "1"
            and not any(result.get("status") == "passed" for result in model_results)
        ):
            required_without_pass.append(model_key)

    for model_key, payload in sorted(payloads.items()):
        results.extend(payload.get("results", []))
        malformed.append((Path(model_key), ValueError("unexpected model artifact")))
    for path, exc in malformed:
        results.append(
            {
                "family": "ci-report",
                "repo": str(path),
                "modality": "",
                "shard": "matrix",
                "status": "failed-infrastructure",
                "reason": str(exc),
            }
        )

    markdown = _summary_markdown(results)
    print(markdown)
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as handle:
            handle.write("## All-model MLX VLM/audio smoke\n\n")
            handle.write(markdown)
    if required_without_pass:
        print(f"Required models without a real pass: {required_without_pass}")
    failed = [
        result
        for result in results
        if str(result.get("status", "")).startswith("failed")
    ]
    return 1 if failed or required_without_pass else 0


def _parent_main(model_key: str | None = None) -> int:
    max_model_gb = float(os.environ.get("UNSLOTH_MLX_REAL_MAX_MODEL_GB", "8.0"))
    min_real_passes = int(os.environ.get("UNSLOTH_MLX_REAL_MIN_REAL_PASSES", "5"))
    timeout_s = int(os.environ.get("UNSLOTH_MLX_REAL_CASE_TIMEOUT_S", "600"))
    download_timeout_s = int(
        os.environ.get("UNSLOTH_MLX_REAL_DOWNLOAD_TIMEOUT_S", "600")
    )
    total_budget_s = int(os.environ.get("UNSLOTH_MLX_REAL_TOTAL_BUDGET_S", "9600"))
    disk_multiplier = float(os.environ.get("UNSLOTH_MLX_REAL_DISK_MULTIPLIER", "2.2"))
    shard = os.environ.get("UNSLOTH_MLX_REAL_SHARD", "all")
    model_key = model_key or os.environ.get("UNSLOTH_MLX_REAL_MODEL_KEY")
    if model_key:
        candidates = [
            candidate for candidate in CANDIDATES
            if _model_key(candidate) == model_key
        ]
    else:
        candidates = [
            candidate for candidate in CANDIDATES
            if shard in {"", "all"} or candidate.shard == shard
        ]
    if not candidates:
        print(f"No candidates selected for model_key={model_key!r}, shard={shard!r}")
        return 1
    if model_key and len({candidate.repo for candidate in candidates}) != 1:
        print(f"Model key {model_key!r} selected more than one repository")
        return 1
    print(
        f"Selected model_key={model_key!r}, shard={shard!r}; "
        f"candidates={len(candidates)}"
    )

    started_at = time.monotonic()
    results: list[dict] = []
    shared_hf_home = (
        Path(tempfile.mkdtemp(prefix=f"unsloth_mlx_real_{model_key}_"))
        if model_key
        else None
    )
    repo_hf_homes: dict[str, Path] = {}
    download_results: dict[str, dict] = {}

    try:
        for candidate in candidates:
            elapsed_total = time.monotonic() - started_at
            if elapsed_total + min(timeout_s, 300) > total_budget_s:
                results.append(
                    {
                        "family": candidate.family,
                        "repo": candidate.repo,
                        "modality": candidate.modality,
                        "shard": candidate.shard,
                        "status": "skipped-time",
                        "reason": f"overall budget {total_budget_s}s nearly exhausted",
                    }
                )
                continue

            print(f"==> {candidate.family}: {candidate.repo}")
            try:
                size = _candidate_size_bytes(candidate.repo)
            except Exception as exc:
                size = None
                print(f"Could not read model metadata for {candidate.repo}: {exc}")
                if not candidate.required:
                    results.append(
                        {
                            "family": candidate.family,
                            "repo": candidate.repo,
                            "modality": candidate.modality,
                            "shard": candidate.shard,
                            "status": "skipped-metadata",
                            "reason": str(exc)[:200],
                        }
                    )
                    continue

            if size is not None:
                size_gb = size / 1e9
                print(f"Repo size estimate: {size_gb:.2f} GB")
                if size_gb > max_model_gb:
                    results.append(
                        {
                            "family": candidate.family,
                            "repo": candidate.repo,
                            "modality": candidate.modality,
                            "shard": candidate.shard,
                            "status": "skipped-size",
                            "reason": f"{size_gb:.2f}GB > limit {max_model_gb:.2f}GB",
                        }
                    )
                    continue

            free_gb = _free_gib(Path.cwd())
            needed_gb = max(10.0, ((size or 0) / GiB) * disk_multiplier + 6.0)
            print(f"Free disk: {free_gb:.1f} GiB; required guard: {needed_gb:.1f} GiB")
            if free_gb < needed_gb:
                results.append(
                    {
                        "family": candidate.family,
                        "repo": candidate.repo,
                        "modality": candidate.modality,
                        "shard": candidate.shard,
                        "status": "skipped-disk",
                        "reason": f"{free_gb:.1f}GiB free < guard {needed_gb:.1f}GiB",
                    }
                )
                continue

            hf_home = shared_hf_home
            if hf_home is None:
                hf_home = repo_hf_homes.setdefault(
                    candidate.repo,
                    Path(tempfile.mkdtemp(prefix="unsloth_mlx_real_repo_")),
                )
            download_result = download_results.get(candidate.repo)
            if download_result is None:
                print(f"Download timeout: {download_timeout_s}s")
                download_result = _download_candidate(
                    candidate.repo,
                    hf_home,
                    download_timeout_s,
                )
                download_results[candidate.repo] = download_result
            if download_result["status"] != "downloaded":
                results.append(
                    {
                        "family": candidate.family,
                        "repo": candidate.repo,
                        "modality": candidate.modality,
                        "shard": candidate.shard,
                        **download_result,
                    }
                )
                continue

            print(f"Child timeout: {timeout_s}s")
            result = _run_child(candidate, timeout_s, hf_home)
            result.setdefault("shard", candidate.shard)
            result.setdefault(
                "download_elapsed_s",
                download_result.get("download_elapsed_s"),
            )
            results.append(result)
    finally:
        if shared_hf_home is not None:
            shutil.rmtree(shared_hf_home, ignore_errors=True)
        for hf_home in repo_hf_homes.values():
            shutil.rmtree(hf_home, ignore_errors=True)

    markdown = _summary_markdown(results)
    print(markdown)
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as handle:
            handle.write("## Real MLX VLM/audio smoke\n\n")
            handle.write(markdown)
    _write_results_artifact(model_key, candidates, results)

    multimodal_failures = [
        result
        for result in results
        if str(result.get("status", "")).startswith("failed")
    ]
    real_passes = [result for result in results if result.get("status") == "passed"]
    if multimodal_failures:
        print("Multimodal candidates failed despite working text-only path:")
        print(json.dumps(multimodal_failures, indent=2))
        return 1
    if min_real_passes > 0 and len(real_passes) < min_real_passes:
        print(
            f"Only {len(real_passes)} real model runs passed; "
            f"expected at least {min_real_passes}."
        )
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child")
    parser.add_argument("--matrix-json", action="store_true")
    parser.add_argument("--matrix-model-keys")
    parser.add_argument("--model-key")
    parser.add_argument("--summarize-results")
    parser.add_argument("--download")
    parser.add_argument("--download-cache")
    args = parser.parse_args()
    if args.child:
        return _child_main(args.child)
    if args.matrix_json:
        print(
            json.dumps(
                _model_matrix(args.matrix_model_keys),
                separators=(",", ":"),
            )
        )
        return 0
    if args.summarize_results:
        return _aggregate_results_main(args.summarize_results)
    if args.download:
        if not args.download_cache:
            parser.error("--download-cache is required with --download")
        from huggingface_hub import snapshot_download

        _stage("download")
        path = snapshot_download(
            args.download,
            cache_dir=args.download_cache,
            allow_patterns=[
                "*.json",
                "*.safetensors",
                "*.py",
                "*.model",
                "*.tiktoken",
                "*.txt",
                "*.jinja",
            ],
        )
        print(f"Downloaded snapshot: {path}")
        return 0
    return _parent_main(args.model_key)


if __name__ == "__main__":
    raise SystemExit(main())
