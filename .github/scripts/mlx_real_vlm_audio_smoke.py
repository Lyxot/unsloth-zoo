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
import textwrap
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
    shard: str = "small-a"
    required: bool = False


CANDIDATES = [
    Candidate("idefics3-smolvlm", "mlx-community/SmolVLM-256M-Instruct-4bit", required=True),
    Candidate("qwen2-vl", "mlx-community/Qwen2-VL-2B-Instruct-4bit", required=True),
    Candidate("qwen3-vl", "unsloth/Qwen3-VL-2B-Instruct", required=True),
    Candidate("qwen3.5-vl", "unsloth/Qwen3.5-0.8B"),
    Candidate("lfm2-vl", "mlx-community/LFM2.5-VL-1.6B-4bit", shard="small-b"),
    Candidate("jina-vlm", "jinaai/jina-vlm-mlx", shard="small-b"),
    Candidate("internvl-chat", "mlx-community/InternVL3-1B-4bit", shard="small-b"),
    Candidate("fastvlm", "mlx-community/FastVLM-0.5B-bf16", shard="small-b"),
    Candidate("gemma4", "unsloth/gemma-4-E2B-it-UD-MLX-4bit", shard="medium-a", required=True),
    Candidate("minicpm-v4.6", "mlx-community/MiniCPM-V-4.6-mxfp4", shard="medium-a"),
    Candidate("glm4v", "mlx-community/GLM-4.6V-Flash-mxfp4", shard="medium-a"),
    Candidate("deepseek-vl-v2", "mlx-community/deepseek-vl2-tiny-4bit", shard="medium-a"),
    Candidate("phi3-v", "mlx-community/Phi-3.5-vision-instruct-4bit", shard="medium-b"),
    Candidate("granite-vision", "mlx-community/granite-vision-3.2-2b-4bit", shard="medium-b"),
    Candidate("qwen2.5-vl", "mlx-community/Qwen2.5-VL-3B-Instruct-4bit", shard="medium-b"),
    Candidate("gemma3", "mlx-community/gemma-3-4b-it-4bit", shard="medium-b"),
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
    Candidate("qwen3.6-vl-moe", "unsloth/Qwen3.6-35B-A3B-NVFP4", shard="moe"),
    Candidate("qwen3-vl-moe", "mlx-community/Qwen3-VL-30B-A3B-Instruct-3bit", shard="moe"),
    Candidate("llama4", "mlx-community/Llama-4-Scout-17B-16E-Instruct-4bit", shard="moe"),
    Candidate("step3p7", "mlx-community/Step-3.7-Flash-4bit", shard="moe"),
    Candidate("ernie4.5-moe-vl", "mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit", shard="moe"),
    Candidate("phi4mm-audio", "Ferox-AI/Phi-4-multimodal-instruct-mlx-4bit", modality="audio", shard="audio-a"),
    Candidate("gemma3n-audio", "mlx-community/gemma-3n-E2B-it-4bit", modality="audio", shard="audio-a"),
    Candidate("gemma4-audio", "unsloth/gemma-4-E2B-it-UD-MLX-4bit", modality="audio", shard="audio-a"),
    Candidate("minicpmo-audio", "mlx-community/MiniCPM-o-4_5-4bit", modality="audio", shard="audio-a"),
    Candidate("qwen2-audio", "mlx-community/Qwen2-Audio-7B-Instruct-4bit", modality="audio", shard="audio-b"),
    Candidate("qwen2.5-omni-audio", "giangndm/qwen2.5-omni-3b-mlx-4bit", modality="audio", shard="audio-b"),
    Candidate("nemotron-h-nano-omni", "unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning", modality="audio", shard="audio-b"),
    Candidate("qwen3-omni-audio", "mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit", modality="audio", shard="audio-b"),
]


def _print_group(title: str, body: str) -> None:
    print(f"::group::{title}")
    print(body.rstrip())
    print("::endgroup::")


def _repo_cache_name(repo: str) -> str:
    return "models--" + repo.replace("/", "--")


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
            "no space left on device",
            "killed: 9",
            "signal 9",
            "metal command buffer",
            "resource exhausted",
        )
    )


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


def _run_child(candidate: Candidate, timeout_s: int) -> dict:
    hf_home = Path(tempfile.mkdtemp(prefix="unsloth_mlx_real_hf_"))
    payload = base64.b64encode(json.dumps(asdict(candidate)).encode()).decode()
    env = os.environ.copy()
    env.update(
        {
            "HF_HOME": str(hf_home),
            "HF_HUB_DISABLE_XET": "1",
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
                "status": "skipped-resource",
                "reason": f"resource failure, returncode={completed.returncode}",
                "elapsed_s": round(elapsed, 1),
            }
        return {
            "family": candidate.family,
            "repo": candidate.repo,
            "status": "failed",
            "reason": _failure_reason(completed.returncode, output),
            "elapsed_s": round(elapsed, 1),
        }
    except subprocess.TimeoutExpired as exc:
        output = _combined_output(exc.stdout, exc.stderr)
        _print_group(f"{candidate.family}: timeout output", output)
        return {
            "family": candidate.family,
            "repo": candidate.repo,
            "status": "skipped-resource",
            "reason": f"timeout after {timeout_s}s",
            "elapsed_s": timeout_s,
        }
    finally:
        shutil.rmtree(hf_home, ignore_errors=True)


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
                    {"type": "text", "text": "Describe this short tone."},
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
            return f"{audio_token}\nDescribe this short tone."

    if "deepseek_vl" in model_type:
        return "<image>\nDescribe this image in one word."

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe this image in one word."},
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
            return f"USER: {image_token}\nDescribe this image in one word.\nASSISTANT:"
        return f"{image_token}\nDescribe this image in one word."


def _text_prompt(tokenizer) -> str:
    tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": "Say hello."}]
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass
    return "Say hello."


def _write_tone(path: Path) -> None:
    import math
    import struct
    import wave

    sample_rate = 16000
    frames = []
    for index in range(sample_rate // 2):
        sample = int(12000 * math.sin(2 * math.pi * 440 * index / sample_rate))
        frames.append(struct.pack("<h", sample))
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"".join(frames))


def _short_exception(exc: BaseException) -> str:
    lines = traceback.format_exception_only(type(exc), exc)
    return " ".join(line.strip() for line in lines if line.strip())[:300]


def _peak_memory_gb(mx) -> float:
    if hasattr(mx.metal, "get_peak_memory"):
        return round(float(mx.metal.get_peak_memory() or 0.0) / GiB, 3)
    return 0.0


def _common_summary(candidate: Candidate, processor, mx, **extra) -> dict:
    summary = {
        "family": candidate.family,
        "repo": candidate.repo,
        "modality": candidate.modality,
        "shard": candidate.shard,
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
        "peak_memory_gb": _peak_memory_gb(mx),
    }
    summary.update(extra)
    return summary


def _run_multimodal_generation(candidate: Candidate, tmp_dir: Path) -> dict:
    import gc
    from PIL import Image
    import mlx.core as mx
    from mlx_vlm import generate
    from unsloth_zoo.mlx.loader import FastMLXModel

    model = processor = None
    try:
        if hasattr(mx.metal, "reset_peak_memory"):
            mx.metal.reset_peak_memory()

        model, processor = FastMLXModel.from_pretrained(
            candidate.repo,
            text_only=False,
            max_seq_length=128,
        )
        config = getattr(model, "config", getattr(model, "_config", {}))
        prompt = _child_prompt(processor, config, candidate.modality)

        kwargs = {
            "max_tokens": int(os.environ.get("UNSLOTH_MLX_REAL_MAX_TOKENS", "2")),
            "verbose": False,
        }
        if candidate.modality == "audio":
            audio_path = tmp_dir / "tone.wav"
            _write_tone(audio_path)
            result = generate(model, processor, prompt, audio=str(audio_path), **kwargs)
        else:
            image_path = tmp_dir / "image.png"
            Image.new("RGB", (48, 48), (80, 120, 210)).save(image_path)
            result = generate(model, processor, prompt, image=str(image_path), **kwargs)

        generation_tokens = int(getattr(result, "generation_tokens", 0) or 0)
        if generation_tokens <= 0:
            raise RuntimeError(f"no generation tokens produced: {result!r}")
        return _common_summary(
            candidate,
            processor,
            mx,
            status="passed",
            prompt_tokens=int(getattr(result, "prompt_tokens", 0) or 0),
            generation_tokens=generation_tokens,
            text_head=str(getattr(result, "text", ""))[:80],
        )
    finally:
        del model, processor
        gc.collect()
        if "mx" in locals() and hasattr(mx.metal, "clear_cache"):
            mx.metal.clear_cache()


def _run_text_only_generation(candidate: Candidate) -> dict:
    import gc
    import mlx.core as mx
    from mlx_lm import generate as text_generate
    from unsloth_zoo.mlx.loader import FastMLXModel

    model = tokenizer = None
    try:
        if hasattr(mx.metal, "reset_peak_memory"):
            mx.metal.reset_peak_memory()

        model, tokenizer = FastMLXModel.from_pretrained(
            candidate.repo,
            text_only=True,
            max_seq_length=128,
        )
        text = text_generate(
            model,
            tokenizer,
            _text_prompt(tokenizer),
            max_tokens=int(os.environ.get("UNSLOTH_MLX_REAL_MAX_TOKENS", "2")),
            verbose=False,
        )
        if not str(text).strip():
            raise RuntimeError("text-only generation returned empty text")
        return _common_summary(
            candidate,
            tokenizer,
            mx,
            text_only_generation="passed",
            text_head=str(text)[:80],
        )
    finally:
        del model, tokenizer
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
            try:
                text_summary = _run_text_only_generation(candidate)
            except Exception as text_exc:
                summary = {
                    "family": candidate.family,
                    "repo": candidate.repo,
                    "modality": candidate.modality,
                    "shard": candidate.shard,
                    "status": "unsupported-yet",
                    "reason": (
                        "text-only path also failed; "
                        f"multimodal_error={multimodal_error}; "
                        f"text_error={_short_exception(text_exc)}"
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
    rows = [
        "| Shard | Family | Repo | Status | Detail |",
        "| --- | --- | --- | --- | --- |",
    ]
    for result in results:
        detail = result.get("reason")
        if result.get("status") == "passed":
            detail = (
                f"{result.get('modality')} generation_tokens={result.get('generation_tokens')} "
                f"peak={result.get('peak_memory_gb')}GB"
            )
        rows.append(
            "| {shard} | {family} | `{repo}` | {status} | {detail} |".format(
                shard=result.get("shard", ""),
                family=result.get("family", ""),
                repo=result.get("repo", ""),
                status=result.get("status", ""),
                detail=(detail or "").replace("|", "\\|"),
            )
        )
    return "\n".join(rows) + "\n"


def _parent_main() -> int:
    max_model_gb = float(os.environ.get("UNSLOTH_MLX_REAL_MAX_MODEL_GB", "8.0"))
    min_real_passes = int(os.environ.get("UNSLOTH_MLX_REAL_MIN_REAL_PASSES", "5"))
    timeout_s = int(os.environ.get("UNSLOTH_MLX_REAL_CASE_TIMEOUT_S", "900"))
    total_budget_s = int(os.environ.get("UNSLOTH_MLX_REAL_TOTAL_BUDGET_S", "9600"))
    disk_multiplier = float(os.environ.get("UNSLOTH_MLX_REAL_DISK_MULTIPLIER", "2.2"))
    shard = os.environ.get("UNSLOTH_MLX_REAL_SHARD", "all")
    candidates = [
        candidate for candidate in CANDIDATES
        if shard in {"", "all"} or candidate.shard == shard
    ]
    if not candidates:
        print(f"No candidates selected for shard={shard!r}")
        return 1
    print(f"Selected shard={shard!r}; candidates={len(candidates)}")

    started_at = time.monotonic()
    results: list[dict] = []

    for candidate in candidates:
        elapsed_total = time.monotonic() - started_at
        if elapsed_total + min(timeout_s, 300) > total_budget_s:
            results.append(
                {
                    "family": candidate.family,
                    "repo": candidate.repo,
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
                    "shard": candidate.shard,
                    "status": "skipped-disk",
                    "reason": f"{free_gb:.1f}GiB free < guard {needed_gb:.1f}GiB",
                }
            )
            continue

        result = _run_child(candidate, timeout_s)
        result.setdefault("shard", candidate.shard)
        results.append(result)

        repo_cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub" / _repo_cache_name(candidate.repo)
        shutil.rmtree(repo_cache, ignore_errors=True)

    markdown = _summary_markdown(results)
    print(markdown)
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as handle:
            handle.write("## Real MLX VLM/audio smoke\n\n")
            handle.write(markdown)

    multimodal_failures = [result for result in results if result.get("status") == "failed"]
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
    args = parser.parse_args()
    if args.child:
        return _child_main(args.child)
    return _parent_main()


if __name__ == "__main__":
    raise SystemExit(main())
