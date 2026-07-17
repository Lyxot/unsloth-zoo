#!/usr/bin/env python3
"""Disposable Qwen3.5 long-tail benchmark for issue 6990."""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import random
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import fields
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import mlx.core as mx
import numpy as np
import psutil
from huggingface_hub import snapshot_download


SEED = 6990
REPETITIONS = 4
MODEL_ID = "Qwen/Qwen3.5-0.8B"


class PrivateMetalResourceProbe:
    """Validated MLX 0.32.0 allocator-resource probe for CI diagnostics."""

    _ALLOCATOR_SYMBOL = "__ZN3mlx4core5metal9allocatorEv"
    _ACTIVE_OFFSET = 0xA8
    _PEAK_OFFSET = 0xB0
    _NUM_RESOURCES_OFFSET = 0xC8
    _RESOURCE_LIMIT_OFFSET = 0xD0

    def __init__(self):
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise RuntimeError("Metal resource probe requires Darwin arm64")

        mlx_dir = Path(mx.__file__).resolve().parent
        self.lib_path = (mlx_dir / "lib" / "libmlx.dylib").resolve()
        nm = subprocess.run(
            ["/usr/bin/nm", "-a", str(self.lib_path)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        symbol_value = None
        for line in nm.splitlines():
            parts = line.split()
            if len(parts) >= 3 and parts[-1] == self._ALLOCATOR_SYMBOL:
                symbol_value = int(parts[0], 16)
                break
        if symbol_value is None:
            raise RuntimeError("private MLX allocator symbol was not found")

        dyld = ctypes.CDLL(None)
        dyld._dyld_image_count.restype = ctypes.c_uint32
        dyld._dyld_get_image_name.argtypes = [ctypes.c_uint32]
        dyld._dyld_get_image_name.restype = ctypes.c_char_p
        dyld._dyld_get_image_header.argtypes = [ctypes.c_uint32]
        dyld._dyld_get_image_header.restype = ctypes.c_void_p

        image_header = None
        for index in range(dyld._dyld_image_count()):
            raw_name = dyld._dyld_get_image_name(index)
            if raw_name and Path(raw_name.decode()).resolve() == self.lib_path:
                image_header = int(dyld._dyld_get_image_header(index))
                break
        if image_header is None:
            raise RuntimeError("loaded libmlx image was not found in dyld")

        allocator_fn = ctypes.CFUNCTYPE(ctypes.c_void_p)(
            image_header + symbol_value
        )
        self._address = int(allocator_fn())
        if not self._address:
            raise RuntimeError("private MLX allocator singleton was null")

        validations = {
            "active": (self._read(self._ACTIVE_OFFSET), int(mx.get_active_memory())),
            "peak": (self._read(self._PEAK_OFFSET), int(mx.get_peak_memory())),
            "limit": (
                self.limit(),
                int(mx.device_info()["resource_limit"]),
            ),
        }
        mismatches = {
            name: values
            for name, values in validations.items()
            if values[0] != values[1]
        }
        if mismatches:
            raise RuntimeError(
                f"private MLX allocator layout validation failed: {mismatches}"
            )

    def _read(self, offset):
        return int(ctypes.c_size_t.from_address(self._address + offset).value)

    def current(self):
        return self._read(self._NUM_RESOURCES_OFFSET)

    def limit(self):
        return self._read(self._RESOURCE_LIMIT_OFFSET)


def long_tail_widths():
    sampled = random.Random(SEED + 4).sample(range(96, 513), 139)
    base = [8, 13, 21, 34, 55, *sampled]
    random.Random(SEED + 104).shuffle(base)
    return tuple(base) * REPETITIONS


def build_dataset(widths):
    rng = np.random.default_rng(SEED)
    rows = []
    digest = hashlib.sha256()
    for row_index, width in enumerate(widths):
        ids = rng.integers(4, 4096, size=int(width), dtype=np.int32)
        labels = ids.copy()
        digest.update(np.asarray([row_index, width], dtype=np.int64).tobytes())
        digest.update(ids.tobytes())
        digest.update(labels.tobytes())
        rows.append({"input_ids": ids.tolist(), "labels": labels.tolist()})
    return rows, digest.hexdigest()


def timeout_handler(_signum, _frame):
    raise TimeoutError("long-tail benchmark exceeded 900 seconds")


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def write_summary(result):
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return
    metrics = result["metrics"]
    cycles = metrics["cycle_metrics"]
    lines = [
        f"## Long-tail benchmark: `{result['metadata']['mode']}`",
        "",
        f"- Source: `{result['metadata']['source_commit']}`",
        f"- Raw/target signatures: {metrics['raw_signature_count']} / "
        f"{metrics['observed_target_signature_count']}",
        f"- Wall time: {metrics['train_wall_seconds']:.2f}s",
        f"- Useful tokens/s: {metrics['useful_tokens_per_second']:.1f}",
        f"- Retained resource delta: {metrics['resources_retained_training_delta']}",
        f"- Peak resource count: {metrics['resources_peak_sampled']}",
        f"- Resource limit: {result['metadata']['resource_limit']}",
        f"- MLX cache before final clear: "
        f"{metrics['cache_before_final_clear_gb']:.1f} GB",
        f"- Peak RSS: {metrics['rss_peak_sampled_gb']:.2f} GB",
        "",
        "| Cycle | Seconds | Useful tokens/s | Resources | Cache GB | RSS GB |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for cycle in cycles:
        lines.append(
            f"| {cycle['cycle']} | {cycle['seconds']:.2f} | "
            f"{cycle['useful_tokens_per_second']:.1f} | "
            f"{cycle['resources_end']} | {cycle['cache_gb_end']:.1f} | "
            f"{cycle['rss_gb_end']:.2f} |"
        )
    with open(summary_path, "a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def run_benchmark(args):
    source = args.source.resolve()
    sys.path.insert(0, str(source))

    from unsloth_zoo.mlx.loader import FastMLXModel
    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig
    import unsloth_zoo.mlx.utils as utils_module

    imported_source = Path(utils_module.__file__).resolve()
    if not imported_source.is_relative_to(source):
        raise RuntimeError(
            f"imported unsloth_zoo from {imported_source}, expected {source}"
        )

    widths = long_tail_widths()
    base_widths = widths[: len(widths) // REPETITIONS]
    dataset, dataset_digest = build_dataset(widths)
    model_path = snapshot_download(MODEL_ID)
    probe = PrivateMetalResourceProbe()
    process = psutil.Process()

    mx.random.seed(SEED)
    model, tokenizer = FastMLXModel.from_pretrained(
        model_path,
        max_seq_length=max(widths),
        load_in_4bit=True,
        load_in_8bit=False,
        load_in_16bit=False,
        text_only=True,
        random_state=SEED,
        trust_remote_code=False,
    )
    model = FastMLXModel.get_peft_model(
        model,
        r=4,
        lora_alpha=8,
        lora_dropout=0,
        random_state=SEED,
        use_gradient_checkpointing=False,
    )
    mx.eval(model.parameters(), model.trainable_parameters())
    mx.synchronize()
    mx.clear_cache()
    mx.synchronize()

    materializations = []
    prepare_samples = []
    step_rows = []
    finite_plan_class = getattr(utils_module, "FiniteTextBatchPlan", None)
    original_materialize = (
        None if finite_plan_class is None else finite_plan_class.materialize
    )

    def recording_materialize(self, index, *, phase=None):
        raw_width = int(self.batch_width(index))
        result = original_materialize(self, index, phase=phase)
        materializations.append({
            "raw_width": raw_width,
            "target_width": int(result[0].shape[1]),
            "phase": phase,
        })
        return result

    if finite_plan_class is not None:
        finite_plan_class.materialize = recording_materialize

    original_prepare = MLXTrainer._prepare_data

    def recording_prepare(self, is_vlm):
        started = time.perf_counter()
        result = original_prepare(self, is_vlm)
        mx.synchronize()
        prepare_samples.append({
            "seconds": time.perf_counter() - started,
            "resources": probe.current(),
        })
        return result

    MLXTrainer._prepare_data = recording_prepare

    config = dict(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        max_steps=len(widths),
        warmup_steps=0,
        learning_rate=2e-4,
        lr_scheduler_type="constant",
        logging_steps=1,
        save_steps=0,
        eval_steps=0,
        report_to="none",
        seed=SEED,
        max_seq_length=max(widths),
        use_cce=True,
        compile=True,
        compile_mode="strict",
        compile_auto_tune=False,
        compile_trace=False,
        gradient_checkpointing=True,
        dataset_order="sequential",
        preserve_dataset_order=True,
        append_eos=False,
        completion_only_loss=False,
        memory_limit_gb=0.0,
        wired_limit_gb=0.0,
        cache_limit_gb=0.0,
        max_grad_norm=0.0,
        max_grad_value=0.0,
        max_grad_leaf_norm=0.0,
    )
    unknown = set(config) - {field.name for field in fields(MLXTrainingConfig)}
    if unknown:
        raise RuntimeError(f"unknown config fields: {sorted(unknown)}")

    captured = io.StringIO()
    try:
        with tempfile.TemporaryDirectory(prefix="pr6990-long-tail-") as output_dir:
            config["output_dir"] = output_dir
            trainer = MLXTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=dataset,
                args=MLXTrainingConfig(**config),
            )
            callback_index = 0

            def on_step(
                step, total, loss, lr, tokens_per_second, peak_gb, elapsed,
                num_tokens, grad_norm=None,
            ):
                nonlocal callback_index
                mx.synchronize()
                event = (
                    materializations[callback_index]
                    if finite_plan_class is not None
                    else {
                        "raw_width": int(widths[callback_index]),
                        "target_width": int(widths[callback_index]),
                        "phase": "single",
                    }
                )
                step_rows.append({
                    "step": int(step),
                    "raw_width": event["raw_width"],
                    "target_width": event["target_width"],
                    "phase": event["phase"],
                    "loss": float(loss),
                    "resources_current": probe.current(),
                    "cache_bytes": int(mx.get_cache_memory()),
                    "rss_bytes": int(process.memory_info().rss),
                })
                callback_index += 1

            trainer.add_step_callback(on_step)
            train_started = time.perf_counter()
            with contextlib.redirect_stdout(captured):
                train_output = trainer.train()
            mx.synchronize()
            train_wall_seconds = time.perf_counter() - train_started
    except Exception:
        print(captured.getvalue(), file=sys.stderr)
        raise
    finally:
        if finite_plan_class is not None:
            finite_plan_class.materialize = original_materialize
        MLXTrainer._prepare_data = original_prepare

    if [row["raw_width"] for row in step_rows] != list(widths):
        raise RuntimeError("training order or membership changed")
    for row, step_time in zip(step_rows, trainer._step_times):
        row["step_time_seconds"] = float(step_time)

    resources_before_final_clear = probe.current()
    cache_before_final_clear = int(mx.get_cache_memory())
    mx.clear_cache()
    mx.synchronize()
    resources_after_final_clear = probe.current()

    seen_targets = set()
    for row in step_rows:
        signature = (row["phase"], row["target_width"])
        row["new_target_signature"] = signature not in seen_targets
        seen_targets.add(signature)

    cycle_metrics = []
    cycle_size = len(base_widths)
    for cycle in range(REPETITIONS):
        rows = step_rows[cycle * cycle_size:(cycle + 1) * cycle_size]
        seconds = sum(row["step_time_seconds"] for row in rows)
        tokens = sum(row["raw_width"] - 1 for row in rows)
        cycle_metrics.append({
            "cycle": cycle + 1,
            "seconds": seconds,
            "useful_tokens_per_second": tokens / seconds,
            "resources_end": rows[-1]["resources_current"],
            "cache_gb_end": rows[-1]["cache_bytes"] / 1e9,
            "rss_gb_end": rows[-1]["rss_bytes"] / 1e9,
        })

    metrics = {
        "prepare_seconds": prepare_samples[-1]["seconds"],
        "train_wall_seconds": train_wall_seconds,
        "trained_tokens": int(train_output["trained_tokens"]),
        "useful_tokens_per_second": (
            float(train_output["trained_tokens"]) / train_wall_seconds
        ),
        "raw_signature_count": len(set(widths)),
        "observed_target_signature_count": len(seen_targets),
        "padding_tokens_observed": sum(
            row["target_width"] - row["raw_width"] for row in step_rows
        ),
        "resources_peak_sampled": max(
            row["resources_current"] for row in step_rows
        ),
        "resources_before_final_clear": resources_before_final_clear,
        "resources_after_final_clear": resources_after_final_clear,
        "resources_retained_training_delta": (
            resources_after_final_clear - prepare_samples[-1]["resources"]
        ),
        "cache_before_final_clear_gb": cache_before_final_clear / 1e9,
        "rss_peak_sampled_gb": max(row["rss_bytes"] for row in step_rows) / 1e9,
        "loss_final": float(trainer._train_loss_history[-1]),
        "finite_losses": bool(np.isfinite(trainer._train_loss_history).all()),
        "cycle_metrics": cycle_metrics,
        "shape_guard": train_output.get("compile_shape_guard"),
        "compile_enabled": bool(train_output["compile_enabled"]),
        "compile_scope": train_output["compile_scope"],
    }
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=source, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    result = {
        "metadata": {
            "mode": args.mode,
            "source": str(source),
            "source_commit": source_commit,
            "rows": len(widths),
            "repetitions": REPETITIONS,
            "schedule_digest": hashlib.sha256(
                json.dumps(widths, separators=(",", ":")).encode()
            ).hexdigest(),
            "dataset_digest": dataset_digest,
            "seed": SEED,
            "resource_limit": probe.limit(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "device_info": mx.device_info(),
            "versions": {
                name: package_version(name)
                for name in ("mlx", "mlx-lm", "mlx-vlm", "numpy", "torch")
            },
            "memory_limit_gb": None,
            "wired_limit_gb": None,
            "cache_limit_gb": None,
            "clear_cache_each_step": False,
            "nice": os.getpriority(os.PRIO_PROCESS, 0),
        },
        "metrics": metrics,
        "steps": step_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True))
    write_summary(result)
    compact = {
        "mode": args.mode,
        "commit": source_commit,
        "wall_seconds": round(train_wall_seconds, 3),
        "useful_tokens_per_second": round(metrics["useful_tokens_per_second"], 3),
        "target_signatures": metrics["observed_target_signature_count"],
        "peak_resources": metrics["resources_peak_sampled"],
        "retained_resources": metrics["resources_retained_training_delta"],
        "cache_gb": round(metrics["cache_before_final_clear_gb"], 3),
        "rss_gb": round(metrics["rss_peak_sampled_gb"], 3),
        "cycles": [round(item["seconds"], 3) for item in cycle_metrics],
    }
    print("BENCHMARK_RESULT_JSON=" + json.dumps(compact, sort_keys=True))
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--mode", choices=("main", "auto"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(900)
    try:
        return run_benchmark(args)
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    raise SystemExit(main())
