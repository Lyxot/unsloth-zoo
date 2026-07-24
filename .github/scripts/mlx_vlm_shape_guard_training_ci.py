#!/usr/bin/env python3
"""Real-model CI proof for finite planned MLX VLM training.

Each matrix job downloads one model, then runs upstream main (compiled and
unplanned) and the feature checkout (compiled and planned) in fresh processes.
The parent compares loss/tokens and verifies that the feature run consumed only
the admitted finite signature catalog.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback


SEED = 3407
ROWS = 140
MAX_SEQ_LENGTH = 384
LOSS_RELATIVE_TOLERANCE = 0.008
RESULT_MARKER = "VLM_SHAPE_GUARD_RESULT:"
GIB = 1024 ** 3


@dataclass(frozen=True)
class Candidate:
    key: str
    family: str
    repo: str
    shard: str


# Image-capable model coverage mirrors Lyxot/unsloth-zoo#3. Video duplicates
# and audio-only candidates are omitted because this workflow exercises
# image+text training. Each repository gets an isolated runner so resource
# failures on very large checkpoints do not hide smaller-family results.
CANDIDATES = (
    Candidate(
        "idefics3-smolvlm",
        "Idefics3 / SmolVLM",
        "mlx-community/SmolVLM-256M-Instruct-4bit",
        "small",
    ),
    Candidate("qwen2-vl", "Qwen2-VL", "mlx-community/Qwen2-VL-2B-Instruct-4bit", "small"),
    Candidate("qwen3-vl", "Qwen3-VL", "unsloth/Qwen3-VL-2B-Instruct", "small"),
    Candidate("qwen3-5-vl", "Qwen3.5-VL", "unsloth/Qwen3.5-0.8B", "small"),
    Candidate("lfm2-vl", "LFM2-VL", "mlx-community/LFM2.5-VL-1.6B-4bit", "small"),
    Candidate("jina-vlm", "Jina VLM", "jinaai/jina-vlm-mlx", "small"),
    Candidate("internvl-chat", "InternVL Chat", "mlx-community/InternVL3-1B-4bit", "small"),
    Candidate("fastvlm", "FastVLM", "mlx-community/FastVLM-0.5B-bf16", "small"),
    Candidate("gemma4", "Gemma4", "unsloth/gemma-4-E2B-it-UD-MLX-4bit", "medium"),
    Candidate("minicpm-v4-6", "MiniCPM-V 4.6", "mlx-community/MiniCPM-V-4.6-mxfp4", "medium"),
    Candidate("glm4v", "GLM-4.6V", "mlx-community/GLM-4.6V-Flash-mxfp4", "medium"),
    Candidate(
        "deepseek-vl-v2",
        "DeepSeek-VL2 tiny",
        "mlx-community/deepseek-vl2-tiny-4bit",
        "medium",
    ),
    Candidate("phi3-v", "Phi-3.5 Vision", "mlx-community/Phi-3.5-vision-instruct-4bit", "medium"),
    Candidate(
        "granite-vision",
        "Granite Vision",
        "mlx-community/granite-vision-3.2-2b-4bit",
        "medium",
    ),
    Candidate("qwen2-5-vl", "Qwen2.5-VL", "mlx-community/Qwen2.5-VL-3B-Instruct-4bit", "medium"),
    Candidate("gemma3", "Gemma3", "unsloth/gemma-3-4b-it", "medium"),
    Candidate("llava", "LLaVA", "mlx-community/llava-1.5-7b-4bit", "large-a"),
    Candidate("llava-next", "LLaVA-NeXT", "mlx-community/llava-v1.6-mistral-7b-4bit", "large-a"),
    Candidate("llava-bunny", "LLaVA Bunny", "BAAI/Bunny-v1_1-Llama-3-8B-V", "large-a"),
    Candidate("gemma3n", "Gemma3n", "mlx-community/gemma-3n-E2B-it-4bit", "large-a"),
    Candidate("idefics2", "Idefics2", "mlx-community/idefics2-8b-4bit", "large-a"),
    Candidate("molmo", "Molmo", "mlx-community/Molmo-7B-D-0924-4bit", "large-a"),
    Candidate("aya-vision", "Aya Vision", "mlx-community/aya-vision-8b-4bit", "large-b"),
    Candidate("mllama", "Mllama", "mlx-community/Llama-3.2-11B-Vision-Instruct-4bit", "large-b"),
    Candidate("zaya1-vl", "ZAYA1-VL", "OsaurusAI/ZAYA1-VL-8B-MXFP4", "large-b"),
    Candidate("youtu-vl", "Youtu-VL", "tencent/Youtu-VL-4B-Instruct", "large-b"),
    Candidate("pixtral", "Pixtral", "mlx-community/pixtral-12b-4bit", "large-b"),
    Candidate("molmo2", "Molmo2", "mlx-community/Molmo2-8B-4bit", "large-b"),
    Candidate("kimi-vl", "Kimi-VL", "mlx-community/Kimi-VL-A3B-Thinking-4bit", "large-b"),
    Candidate(
        "deepseek-vl-v2-small",
        "DeepSeek-VL2 small",
        "mlx-community/deepseek-vl2-small-4bit",
        "large-b",
    ),
    Candidate(
        "mistral3-vl",
        "Mistral3-VL",
        "mlx-community/Mistral-Small-3.1-24B-Instruct-2503-4bit",
        "large-b",
    ),
    Candidate("qwen3-6-moe", "Qwen3.6 MoE", "unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit", "moe"),
    Candidate(
        "qwen3-vl-moe",
        "Qwen3-VL MoE",
        "mlx-community/Qwen3-VL-30B-A3B-Instruct-3bit",
        "moe",
    ),
    Candidate("llama4", "Llama4", "mlx-community/Llama-4-Scout-17B-16E-Instruct-4bit", "moe"),
    Candidate("step3p7", "Step-3.7", "mlx-community/Step-3.7-Flash-4bit", "moe"),
    Candidate(
        "ernie4-5-moe-vl",
        "ERNIE-4.5 MoE VL",
        "mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit",
        "moe",
    ),
)

SHARD_POLICIES = {
    "small": {
        "max_model_gb": 8.0,
        "disk_multiplier": 2.0,
        "timeout_minutes": 45,
        "case_timeout_s": 900,
    },
    "medium": {
        "max_model_gb": 12.0,
        "disk_multiplier": 1.7,
        "timeout_minutes": 60,
        "case_timeout_s": 1200,
    },
    "large-a": {
        "max_model_gb": 18.0,
        "disk_multiplier": 1.35,
        "timeout_minutes": 75,
        "case_timeout_s": 1800,
    },
    "large-b": {
        "max_model_gb": 26.0,
        "disk_multiplier": 1.25,
        "timeout_minutes": 90,
        "case_timeout_s": 2100,
    },
    "moe": {
        "max_model_gb": 40.0,
        "disk_multiplier": 1.15,
        "timeout_minutes": 105,
        "case_timeout_s": 2400,
    },
}


class ChildRunError(RuntimeError):
    def __init__(self, message: str, output: str, returncode: int | None):
        super().__init__(message)
        self.output = output
        self.returncode = returncode


def is_resource_failure(returncode: int | None, output: str) -> bool:
    if returncode in {-9, 137, 247}:
        return True
    lowered = output.lower()
    return any(
        marker in lowered
        for marker in (
            "out of memory",
            "outofmemory",
            "cannot allocate memory",
            "unable to allocate",
            "virtual memory exhausted",
            "malloc: can't allocate region",
            "no space left on device",
            "killed: 9",
            "signal 9",
        )
    )


def candidate_size_bytes(repo: str) -> int | None:
    from huggingface_hub import HfApi

    info = HfApi().model_info(repo, files_metadata=True)
    total = sum(
        getattr(sibling, "size", None) or 0
        for sibling in info.siblings
    )
    return total if total > 0 else None


def stable_digest(value) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def candidate_for_key(key: str) -> Candidate:
    for candidate in CANDIDATES:
        if candidate.key == key:
            return candidate
    raise ValueError(f"Unknown model key {key!r}")


def model_matrix(selected_keys: str | None = None) -> dict:
    selected = {
        key.strip()
        for key in str(selected_keys or "").split(",")
        if key.strip()
    }
    known = {candidate.key for candidate in CANDIDATES}
    unknown = sorted(selected - known)
    if unknown:
        raise ValueError(f"Unknown model keys: {unknown}")
    include = []
    for candidate in CANDIDATES:
        if selected and candidate.key not in selected:
            continue
        policy = SHARD_POLICIES[candidate.shard]
        include.append(
            {
                **asdict(candidate),
                "rows": ROWS,
                **policy,
                "large_model": candidate.shard in {"large-a", "large-b", "moe"},
            }
        )
    return {"include": include}


def make_fixture():
    """Build 140 deterministic rows with image and text shape diversity."""
    from PIL import Image, ImageDraw

    # These dimensions are distinct post-prepare families on dynamic-image
    # processors and remain valid inputs for fixed-resolution processors.
    image_sizes = (
        (28, 112),
        (56, 56),
        (112, 28),
        (28, 168),
        (168, 28),
        (28, 196),
        (56, 84),
        (84, 56),
        (196, 28),
        (28, 224),
        (224, 28),
        (56, 112),
        (140, 56),
        (84, 84),
        (56, 168),
        (168, 56),
        (56, 196),
        (84, 140),
        (140, 84),
        (196, 56),
    )
    repeat_counts = (0, 61, 104, 177, 219, 264, 309)
    rows = []
    manifest = []
    for family, (image_width, image_height) in enumerate(image_sizes):
        short_side = min(image_width, image_height)
        for width_index, base_repeat_count in enumerate(repeat_counts):
            image = Image.new(
                "RGB",
                (image_width, image_height),
                (
                    (family * 53 + width_index * 17) % 256,
                    (family * 89 + width_index * 29) % 256,
                    (family * 31 + width_index * 47) % 256,
                ),
            )
            draw = ImageDraw.Draw(image)
            inset = 2 + (width_index % max(2, short_side // 8))
            draw.rectangle(
                (
                    inset,
                    inset,
                    image_width - inset - 1,
                    image_height - inset - 1,
                ),
                outline=(
                    (255 - family * 11) % 256,
                    (20 + width_index * 23) % 256,
                    90,
                ),
                width=max(1, short_side // 28),
            )
            repeat_count = base_repeat_count + family
            prompt = (
                "Inspect the patterned rectangle and report its group and index."
                + " detail" * repeat_count
            )
            answer = (
                f"The rectangle belongs to image group {family} and sample "
                f"{width_index}."
            )
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": answer}],
                },
            ]
            row_id = f"{family}:{width_index}"
            rows.append(
                {
                    "messages": messages,
                    "images": [image],
                    "_shape_guard_ci_row_id": row_id,
                }
            )
            manifest.append(
                {
                    "row_id": row_id,
                    "family": family,
                    "image_size": [image_width, image_height],
                    "width_index": width_index,
                    "repeat_count": repeat_count,
                    "messages": messages,
                    "image_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                }
            )
    assert len(rows) == ROWS
    order = list(range(len(rows)))
    random.Random(SEED).shuffle(order)
    return (
        [rows[index] for index in order],
        [manifest[index] for index in order],
    )


def array_width(batch) -> int | None:
    value = batch.get("input_ids") if isinstance(batch, dict) else None
    shape = getattr(value, "shape", None)
    if shape is None or len(shape) < 2:
        return None
    return int(shape[1])


def model_type(model) -> str | None:
    config = getattr(model, "_config", None)
    if isinstance(config, dict):
        return config.get("model_type")
    return getattr(config, "model_type", None)


def child_main(payload_text: str) -> int:
    payload = json.loads(payload_text)
    candidate = candidate_for_key(payload["model_key"])
    source = Path(payload["source"]).resolve()
    mode = payload["mode"]
    if mode not in ("main", "planned"):
        raise ValueError(f"Unknown child mode {mode!r}")
    sys.path.insert(0, str(source))

    import mlx.core as mx
    import psutil
    import unsloth_zoo
    from unsloth_zoo.mlx.loader import FastMLXModel
    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig
    import unsloth_zoo.mlx.utils as utils_module

    imported = Path(unsloth_zoo.__file__).resolve()
    if not imported.is_relative_to(source):
        raise RuntimeError(f"Imported {imported}, expected source under {source}")

    try:
        os.nice(10)
    except OSError:
        pass
    mx.random.seed(SEED)
    rows, fixture_manifest = make_fixture()
    fixture_digest = stable_digest(fixture_manifest)
    process = psutil.Process()

    model, processor = FastMLXModel.from_pretrained(
        candidate.repo,
        max_seq_length=MAX_SEQ_LENGTH,
        text_only=False,
        load_in_4bit=True,
        random_state=SEED,
    )
    actual_arch = model_type(model)
    model = FastMLXModel.get_peft_model(
        model,
        r=4,
        lora_alpha=4,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="mlx",
        random_state=SEED,
        max_seq_length=MAX_SEQ_LENGTH,
        train_vision=False,
        train_projector=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
    )
    mx.eval(model.parameters(), model.trainable_parameters())
    mx.synchronize()

    output_dir = Path(tempfile.gettempdir()) / (
        f"mlx-vlm-shape-{candidate.key}-{mode}-{os.getpid()}"
    )
    training_args = MLXTrainingConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        max_steps=ROWS,
        warmup_steps=5,
        learning_rate=2e-4,
        logging_steps=1,
        optim="adamw",
        weight_decay=0.001,
        lr_scheduler_type="linear",
        seed=SEED,
        output_dir=str(output_dir),
        report_to="none",
        save_steps=0,
        eval_steps=0,
        max_seq_length=MAX_SEQ_LENGTH,
        use_cce=True,
        compile=True,
        compile_mode="best_effort",
        compile_auto_tune=False,
        gradient_checkpointing=True,
        dataset_order="sequential",
        preserve_dataset_order=True,
        completion_only_loss=False,
        append_eos=False,
        disable_memory_limits=True,
        compile_max_variants=None,
    )
    trainer = MLXTrainer(
        model=model,
        tokenizer=processor,
        processor=processor,
        train_dataset=rows,
        args=training_args,
    )
    trainer.save_model = lambda *args, **kwargs: None

    materializations = []
    planned_catalog = set()
    catalog_counts = {"raw": None, "planned": None}
    finite_plan_type = getattr(utils_module, "FiniteVLMBatchPlan", None)
    original_materialize = (
        getattr(finite_plan_type, "materialize", None)
        if finite_plan_type is not None
        else None
    )
    if original_materialize is not None:

        def recording_materialize(
            self,
            index,
            target_width=None,
            *,
            phase=None,
        ):
            batch = original_materialize(
                self,
                index,
                target_width=target_width,
                phase=phase,
            )
            family_digest = None
            row_id = None
            if phase is not None and getattr(self, "_descriptors", None) is not None:
                family_digest = hashlib.sha256(
                    repr(self.batch_family(index)).encode()
                ).hexdigest()
                scheduled_rows = self._schedule[int(index)]
                if len(scheduled_rows) == 1:
                    item = self._rows[scheduled_rows[0]].item
                    if isinstance(item, dict):
                        row_id = item.get("_shape_guard_ci_row_id")
                shape_plan = getattr(self, "_shape_plan", None)
                if shape_plan is not None:
                    catalog_counts["raw"] = len(shape_plan.raw_catalog)
                    catalog_counts["planned"] = len(shape_plan.planned_catalog)
                    planned_catalog.update(
                        (
                            signature[1],
                            hashlib.sha256(
                                repr(signature[2]).encode()
                            ).hexdigest(),
                            int(signature[3]),
                        )
                        for signature in shape_plan.planned_catalog
                    )
            materializations.append(
                {
                    "index": int(index),
                    "phase": phase,
                    "family_digest": family_digest,
                    "observed_width": array_width(batch),
                    "row_id": row_id,
                }
            )
            return batch

        finite_plan_type.materialize = recording_materialize

    steps = []
    last_materialization_count = 0

    def on_step(
        step,
        total_steps,
        loss,
        learning_rate,
        tokens_per_second,
        peak_gb,
        elapsed,
        num_tokens,
        grad_norm=None,
    ):
        nonlocal last_materialization_count
        mx.synchronize()
        recent = materializations[last_materialization_count:]
        last_materialization_count = len(materializations)
        materialized = next(
            (item for item in reversed(recent) if item["phase"] is not None),
            None,
        )
        steps.append(
            {
                "step": int(step),
                "loss": float(loss),
                "trained_tokens": int(num_tokens),
                "observed_width": (
                    None
                    if materialized is None
                    else materialized["observed_width"]
                ),
                "phase": (
                    None if materialized is None else materialized["phase"]
                ),
                "batch_index": (
                    None if materialized is None else materialized["index"]
                ),
                "row_id": (
                    None if materialized is None else materialized["row_id"]
                ),
                "family_digest": (
                    None
                    if materialized is None
                    else materialized["family_digest"]
                ),
                "rss_bytes": int(process.memory_info().rss),
                "mlx_peak_bytes": int(mx.get_peak_memory()),
            }
        )

    trainer.add_step_callback(on_step)
    started = time.perf_counter()
    try:
        train_output = trainer.train()
        mx.synchronize()
    finally:
        if original_materialize is not None:
            finite_plan_type.materialize = original_materialize
        shutil.rmtree(output_dir, ignore_errors=True)
    wall_seconds = time.perf_counter() - started
    guard = dict(train_output["compile_shape_guard"])

    observed_signatures = {
        (
            item["phase"],
            item["family_digest"],
            item["observed_width"],
        )
        for item in materializations
        if item["phase"] is not None
        and item["family_digest"] is not None
        and item["observed_width"] is not None
    }
    admitted_widths = {
        int(width)
        for widths in guard.get("planned_endpoints", {}).values()
        for width in widths
    }
    observed_widths = {
        int(item["observed_width"])
        for item in materializations
        if item["phase"] is not None and item["observed_width"] is not None
    }
    observed_indices = [
        int(step["batch_index"])
        for step in steps
        if step["batch_index"] is not None
    ]
    observed_row_ids = [
        step["row_id"]
        for step in steps
        if step["row_id"] is not None
    ]
    expected_row_ids = [item["row_id"] for item in fixture_manifest]

    plan_verified = False
    plan_supported = False
    plan_skip_reason = None
    if mode == "planned":
        raw = int(guard.get("raw_signatures") or 0)
        planned = int(guard.get("planned_signatures") or 0)
        action = guard.get("action")
        if action in ("exact", "bucket"):
            plan_supported = True
            expected_action = "exact" if raw <= 128 else "bucket"
            checks = {
                "threshold_action": action == expected_action,
                "raw_positive": raw > 0,
                "planned_positive": planned > 0,
                "planned_at_most_128": planned <= 128,
                "raw_catalog_count": catalog_counts["raw"] == raw,
                "planned_catalog_snapshot": (
                    catalog_counts["planned"] == planned
                ),
                "installed_catalog_count": len(planned_catalog) == planned,
                "observed_catalog_count": len(observed_signatures) == planned,
                "observed_catalog": observed_signatures == planned_catalog,
                "observed_widths": observed_widths == admitted_widths,
                "training_order": observed_indices == list(range(ROWS)),
                "training_row_identity": observed_row_ids == expected_row_ids,
                "compile_enabled": train_output.get("compile_enabled") is True,
                "full_step": train_output.get("compile_scope") == "full_step",
                "lazy_batches": guard.get("lazy_batches") is True,
            }
            failed = [name for name, passed in checks.items() if not passed]
            if failed:
                raise RuntimeError(
                    f"Planned guard verification failed: {failed}; "
                    f"report={guard}; "
                    f"observed_signatures={len(observed_signatures)}"
                )
            plan_verified = True
        elif (
            action == "not_applicable"
            and guard.get("reason") == "vlm_compile_unqualified"
        ):
            plan_skip_reason = str(guard.get("reason") or "not_applicable")
        else:
            raise RuntimeError(
                f"Unexpected planned guard action {action!r}: {guard}"
            )
    else:
        if guard.get("action") not in ("not_applicable", "disabled"):
            raise RuntimeError(
                f"Main unexpectedly returned a planned guard report: {guard}"
            )

    result = {
        "status": "completed",
        "mode": mode,
        "candidate": asdict(candidate),
        "source": str(source),
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=source,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "imported_zoo": str(imported),
        "fixture_digest": fixture_digest,
        "rows": len(rows),
        "model_type": actual_arch,
        "train_loss": float(train_output["train_loss"]),
        "trained_tokens": int(train_output["trained_tokens"]),
        "train_runtime_seconds": float(train_output["train_runtime"]),
        "train_wall_seconds": wall_seconds,
        "useful_tokens_per_second": (
            int(train_output["trained_tokens"])
            / float(train_output["train_runtime"])
        ),
        "peak_rss_bytes": max(step["rss_bytes"] for step in steps),
        "mlx_peak_bytes": int(mx.get_peak_memory()),
        "guard": guard,
        "compile_enabled": bool(train_output.get("compile_enabled")),
        "compile_scope": train_output.get("compile_scope"),
        "compile_reason": train_output.get("compile_reason"),
        "plan_supported": plan_supported,
        "plan_verified": plan_verified,
        "plan_skip_reason": plan_skip_reason,
        "observed_signature_count": len(observed_signatures),
        "planned_catalog_count": len(planned_catalog),
        "observed_indices": observed_indices,
        "observed_row_ids": observed_row_ids,
        "observed_widths": sorted(observed_widths),
        "admitted_widths": sorted(admitted_widths),
        "trained_token_series": [step["trained_tokens"] for step in steps],
        "step_count": len(steps),
    }
    print(RESULT_MARKER + json.dumps(result, sort_keys=True), flush=True)
    return 0


def run_child(
    candidate: Candidate,
    mode: str,
    source: Path,
    timeout_seconds: int,
    log_path: Path,
) -> tuple[dict, str]:
    payload = json.dumps(
        {
            "model_key": candidate.key,
            "mode": mode,
            "source": str(source.resolve()),
        },
        separators=(",", ":"),
    )
    env = os.environ.copy()
    env.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONPATH": str(source.resolve()),
        }
    )
    try:
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--child", payload],
            cwd=source,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        output = "\n".join(
            text
            for text in (
                error.stdout.decode(errors="replace")
                if isinstance(error.stdout, bytes)
                else error.stdout,
                error.stderr.decode(errors="replace")
                if isinstance(error.stderr, bytes)
                else error.stderr,
            )
            if text
        )
        log_path.write_text(output, encoding="utf-8")
        raise ChildRunError(
            f"{candidate.key} {mode} timed out after {timeout_seconds}s",
            output,
            None,
        ) from error
    output = completed.stdout + completed.stderr
    log_path.write_text(output, encoding="utf-8")
    print(f"::group::{candidate.key} {mode}")
    print(output.rstrip())
    print("::endgroup::")
    matches = re.findall(
        rf"^{re.escape(RESULT_MARKER)}(.+)$",
        output,
        flags=re.MULTILINE,
    )
    if completed.returncode != 0 or not matches:
        raise ChildRunError(
            f"{candidate.key} {mode} failed with exit {completed.returncode}",
            output,
            completed.returncode,
        )
    return json.loads(matches[-1]), output


def compare_runs(main_result: dict, planned_result: dict) -> dict:
    if main_result["fixture_digest"] != planned_result["fixture_digest"]:
        raise RuntimeError("Main and planned fixtures differ")
    if main_result["trained_token_series"] != planned_result["trained_token_series"]:
        raise RuntimeError("Main and planned trained-token series differ")
    if main_result["trained_tokens"] != planned_result["trained_tokens"]:
        raise RuntimeError("Main and planned trained-token totals differ")
    main_loss = float(main_result["train_loss"])
    planned_loss = float(planned_result["train_loss"])
    if not math.isfinite(main_loss) or not math.isfinite(planned_loss):
        raise RuntimeError(
            f"Non-finite loss: main={main_loss}, planned={planned_loss}"
        )
    loss_relative = abs(planned_loss - main_loss) / max(abs(main_loss), 1e-12)
    if loss_relative > LOSS_RELATIVE_TOLERANCE:
        raise RuntimeError(
            f"Loss difference {loss_relative:.6%} exceeds "
            f"{LOSS_RELATIVE_TOLERANCE:.2%}: "
            f"main={main_loss}, planned={planned_loss}"
        )
    guard = planned_result["guard"]
    plan_supported = bool(planned_result["plan_supported"])
    if plan_supported and not planned_result["plan_verified"]:
        raise RuntimeError("Planned child did not verify its shape plan")
    if plan_supported and (
        main_result.get("compile_enabled") is not True
        or main_result.get("compile_scope") != "full_step"
    ):
        raise RuntimeError(
            "Main did not remain a compiled full-step baseline: "
            f"enabled={main_result.get('compile_enabled')!r}, "
            f"scope={main_result.get('compile_scope')!r}"
        )
    if not plan_supported and main_result.get("compile_enabled") is True:
        raise RuntimeError(
            "Feature lost compiled-training qualification relative to main: "
            f"main scope={main_result.get('compile_scope')!r}, "
            f"feature reason={planned_result.get('plan_skip_reason')!r}"
        )
    return {
        "loss_relative": loss_relative,
        "loss_tolerance": LOSS_RELATIVE_TOLERANCE,
        "trained_tokens_exact": True,
        "fixture_exact": True,
        "plan_supported": plan_supported,
        "plan_verified": bool(planned_result["plan_verified"]),
        "plan_skip_reason": planned_result.get("plan_skip_reason"),
        "action": guard["action"],
        "raw_signatures": guard["raw_signatures"],
        "planned_signatures": guard["planned_signatures"],
        "padding_work_fraction": guard["padding_work_fraction"],
        "max_width_stretch": guard["max_width_stretch"],
        "runtime_delta": (
            planned_result["train_runtime_seconds"]
            / main_result["train_runtime_seconds"]
            - 1.0
        ),
        "rss_delta": (
            planned_result["peak_rss_bytes"]
            / main_result["peak_rss_bytes"]
            - 1.0
        ),
    }


def parent_main(args) -> int:
    candidate = candidate_for_key(args.model_key)
    feature_source = Path(args.feature_source).resolve()
    main_source = Path(args.main_source).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "result.json"
    payload = {
        "status": "starting",
        "candidate": asdict(candidate),
    }
    try:
        from huggingface_hub import snapshot_download

        size_bytes = candidate_size_bytes(candidate.repo)
        payload["repo_size_bytes"] = size_bytes
        max_model_gb = float(args.max_model_gb)
        if size_bytes is not None and size_bytes / 1e9 > max_model_gb:
            payload.update(
                {
                    "status": "skipped-resource",
                    "resource_reason": "repository_size",
                    "error": (
                        f"repository is {size_bytes / 1e9:.2f} GB, above "
                        f"the {max_model_gb:.2f} GB runner policy"
                    ),
                }
            )
            result_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n"
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        disk_multiplier = float(args.disk_multiplier)
        free_gib = shutil.disk_usage(output_dir).free / GIB
        needed_gib = max(
            10.0,
            ((size_bytes or 0) / GIB) * disk_multiplier + 6.0,
        )
        payload["free_disk_gib"] = free_gib
        payload["required_disk_gib"] = needed_gib
        if free_gib < needed_gib:
            payload.update(
                {
                    "status": "skipped-resource",
                    "resource_reason": "disk_headroom",
                    "error": (
                        f"{free_gib:.1f} GiB free, below the "
                        f"{needed_gib:.1f} GiB download/training guard"
                    ),
                }
            )
            result_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n"
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        download_started = time.perf_counter()
        snapshot_download(candidate.repo)
        payload["download_seconds"] = time.perf_counter() - download_started
        timeout_seconds = int(os.environ.get("VLM_SHAPE_CASE_TIMEOUT_S", "2400"))
        main_result, _ = run_child(
            candidate,
            "main",
            main_source,
            timeout_seconds,
            output_dir / "main.log",
        )
        planned_result, _ = run_child(
            candidate,
            "planned",
            feature_source,
            timeout_seconds,
            output_dir / "planned.log",
        )
        comparison = compare_runs(main_result, planned_result)
        status = (
            "passed"
            if comparison["plan_verified"]
            else "skipped-unqualified"
        )
        payload.update(
            {
                "status": status,
                "main": main_result,
                "planned": planned_result,
                "comparison": comparison,
            }
        )
        print(
            "VLM_SHAPE_GUARD_COMPARISON:"
            + json.dumps(
                {
                    "model_key": candidate.key,
                    **comparison,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    except BaseException as error:
        resource_failure = (
            isinstance(error, ChildRunError)
            and is_resource_failure(error.returncode, error.output)
        ) or is_resource_failure(None, traceback.format_exc())
        payload.update(
            {
                "status": (
                    "skipped-resource" if resource_failure else "failed"
                ),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
        )
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return (
        0
        if payload["status"]
        in {"passed", "skipped-unqualified", "skipped-resource"}
        else 1
    )


def markdown_summary(results: list[dict]) -> str:
    lines = [
        "## MLX VLM planned-training proof",
        "",
        "| Model | Status | Action | Signatures | Loss diff | Runtime delta | Padding work |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        candidate = result.get("candidate", {})
        comparison = result.get("comparison", {})
        signatures = ""
        if comparison:
            signatures = (
                f"{comparison.get('raw_signatures')} → "
                f"{comparison.get('planned_signatures')}"
            )
        lines.append(
            "| {model} | {status} | {action} | {signatures} | {loss} | "
            "{runtime} | {padding} |".format(
                model=candidate.get("family", candidate.get("key", "?")),
                status=result.get("status", "missing"),
                action=comparison.get("action", ""),
                signatures=signatures,
                loss=(
                    f"{comparison['loss_relative']:.3%}"
                    if "loss_relative" in comparison
                    else ""
                ),
                runtime=(
                    f"{comparison['runtime_delta']:+.1%}"
                    if "runtime_delta" in comparison
                    else ""
                ),
                padding=(
                    f"{comparison['padding_work_fraction']:.2%}"
                    if "padding_work_fraction" in comparison
                    else ""
                ),
            )
        )
    return "\n".join(lines) + "\n"


def summarize_main(results_dir: str) -> int:
    root = Path(results_dir)
    results = []
    malformed = []
    for path in sorted(root.rglob("result.json")):
        try:
            results.append(json.loads(path.read_text()))
        except Exception as error:
            malformed.append((str(path), str(error)))
    expected_text = os.environ.get("VLM_SHAPE_EXPECTED_MATRIX", "")
    expected = json.loads(expected_text).get("include", []) if expected_text else []
    expected_keys = {item["key"] for item in expected}
    found_keys = {
        result.get("candidate", {}).get("key")
        for result in results
        if result.get("candidate", {}).get("key")
    }
    for missing in sorted(expected_keys - found_keys):
        results.append(
            {
                "status": "missing",
                "candidate": {"key": missing, "family": missing},
                "error": "model job did not upload a result artifact",
            }
        )
    summary = markdown_summary(results)
    print(summary)
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as handle:
            handle.write(summary)
    failed = [
        result
        for result in results
        if result.get("status") in {"failed", "missing", "starting"}
    ]
    passed = [result for result in results if result.get("status") == "passed"]
    if malformed:
        print("Malformed artifacts:", json.dumps(malformed, indent=2))
    if not passed:
        print("No model produced a verified finite VLM shape plan")
    if failed or malformed or not passed:
        print("Failed results:", json.dumps(failed, indent=2, sort_keys=True))
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-json", action="store_true")
    parser.add_argument("--matrix-model-keys")
    parser.add_argument("--model-key")
    parser.add_argument("--feature-source")
    parser.add_argument("--main-source")
    parser.add_argument("--output-dir")
    parser.add_argument("--max-model-gb", type=float)
    parser.add_argument("--disk-multiplier", type=float)
    parser.add_argument("--child")
    parser.add_argument("--summarize-results")
    args = parser.parse_args()
    if args.matrix_json:
        print(
            json.dumps(
                model_matrix(args.matrix_model_keys),
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        return 0
    if args.child:
        return child_main(args.child)
    if args.summarize_results:
        return summarize_main(args.summarize_results)
    required = {
        "--model-key": args.model_key,
        "--feature-source": args.feature_source,
        "--main-source": args.main_source,
        "--output-dir": args.output_dir,
        "--max-model-gb": args.max_model_gb,
        "--disk-multiplier": args.disk_multiplier,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        parser.error("required for a model run: " + ", ".join(missing))
    return parent_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
