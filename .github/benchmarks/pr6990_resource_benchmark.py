#!/usr/bin/env python3
"""Disposable same-runner Qwen3.5 benchmark for issue 6990."""

from __future__ import annotations

import argparse
import ctypes
import gzip
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
import tempfile
import time
from dataclasses import fields
from pathlib import Path


SEED = 6990
MODEL_ID = "Qwen/Qwen3.5-0.8B"
DATASET_ID = "mlabonne/FineTome-100k"
MAX_FINE_LENGTH = 1024
RESOURCE_STOP = 350_000
SYNTHETIC_MODES = ("main", "auto", "cap32", "cap64", "cap128", "cap256")

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def digest_json(value):
    payload = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def synthetic_widths(name):
    rng = random.Random(SEED + {"small": 1, "clustered": 2, "irregular": 3, "long_tail": 4}[name])
    if name == "small":
        widths = rng.sample(range(37, 401), 24)
    elif name == "clustered":
        widths = []
        for low, high in ((72, 108), (176, 216), (288, 332), (432, 480)):
            widths.extend(rng.sample(range(low, high + 1), 16))
    elif name == "irregular":
        widths = rng.sample(range(41, 513), 96)
    else:
        widths = [8, 13, 21, 34, 55, *rng.sample(range(96, 513), 139)]
    random.Random(SEED + 100 + len(widths)).shuffle(widths)
    return widths


def synthetic_rows(name):
    import numpy as np

    widths = synthetic_widths(name)
    rng = np.random.default_rng(SEED)
    rows = []
    for width in widths:
        ids = rng.integers(4, 4096, size=width, dtype=np.int32).tolist()
        rows.append({"input_ids": ids, "labels": list(ids)})
    return rows, widths, 1, 512


def normalize_conversation(conversation):
    roles = {"human": "user", "gpt": "assistant"}
    messages = []
    for turn in conversation:
        role = roles.get(turn["from"], turn["from"])
        messages.append({"role": role, "content": turn["value"]})
    return messages


def write_gzip_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(value, handle, separators=(",", ":"), sort_keys=True)


def read_gzip_json(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def percentile_map(values):
    import numpy as np

    points = (0, 1, 5, 25, 50, 75, 90, 95, 99, 100)
    result = np.percentile(np.asarray(values, dtype=np.int64), points)
    return {str(point): round(float(value), 3) for point, value in zip(points, result)}


def prepare_finetome(args):
    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    model_path = snapshot_download(MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    dataset = load_dataset(DATASET_ID, split="train")

    raw_lengths = []
    truncated_lengths = []
    counts = {}
    representatives = {}
    scan_digest = hashlib.sha256()
    truncated_rows = 0
    for index, row in enumerate(dataset):
        messages = normalize_conversation(row["conversations"])
        ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )
        if isinstance(ids, dict):
            ids = ids["input_ids"]
        ids = [int(token) for token in ids]
        raw_length = len(ids)
        width = min(raw_length, MAX_FINE_LENGTH)
        if width < 2:
            continue
        raw_lengths.append(raw_length)
        truncated_lengths.append(width)
        counts[width] = counts.get(width, 0) + 1
        if width not in representatives:
            representatives[width] = {
                "dataset_index": index,
                "input_ids": ids[:MAX_FINE_LENGTH],
                "labels": ids[:MAX_FINE_LENGTH],
            }
        if raw_length > MAX_FINE_LENGTH:
            truncated_rows += 1
        scan_digest.update(index.to_bytes(8, "little"))
        scan_digest.update(raw_length.to_bytes(8, "little"))
        if (index + 1) % 5000 == 0:
            print(f"prescan rows={index + 1} distinct={len(counts)}", flush=True)

    ranked_widths = sorted(counts, key=lambda width: (-counts[width], width))
    if len(ranked_widths) < 144:
        raise RuntimeError(f"FineTome exposed only {len(ranked_widths)} widths")
    selected_widths = ranked_widths[:144]
    random.Random(SEED + 200).shuffle(selected_widths)
    full_widths = sorted(representatives)
    random.Random(SEED + 201).shuffle(full_widths)

    def rows_for(widths):
        return [representatives[width] for width in widths]

    selected_rows = rows_for(selected_widths)
    bundle = {
        "metadata": {
            "dataset_id": DATASET_ID,
            "dataset_rows": len(dataset),
            "usable_rows": len(raw_lengths),
            "max_sequence_length": MAX_FINE_LENGTH,
            "chat_template_source": MODEL_ID,
            "distinct_truncated_widths": len(counts),
            "raw_length_percentiles": percentile_map(raw_lengths),
            "truncated_length_percentiles": percentile_map(truncated_lengths),
            "truncated_rows": truncated_rows,
            "truncation_rate": truncated_rows / len(raw_lengths),
            "min_truncated_width": min(counts),
            "max_truncated_width": max(counts),
            "scan_digest": scan_digest.hexdigest(),
            "selected_width_strategy": "144 most frequent distinct truncated widths",
            "selected_widths": selected_widths,
            "selected_width_counts": [counts[width] for width in selected_widths],
            "selected_schedule_digest": digest_json(selected_widths * 4),
            "full_diversity_schedule_digest": digest_json(full_widths),
            "versions": {
                name: package_version(name)
                for name in ("datasets", "transformers", "huggingface_hub")
            },
        },
        "selected_cycle_rows": selected_rows,
        "full_diversity_rows": rows_for(full_widths),
    }
    write_gzip_json(args.output, bundle)
    args.stats.write_text(json.dumps(bundle["metadata"], indent=2, sort_keys=True))
    print("FINETOME_PRESCAN_JSON=" + json.dumps(bundle["metadata"], sort_keys=True))


class PrivateMetalResourceProbe:
    """Validated MLX 0.32.0 allocator-resource probe for CI diagnostics."""

    _ALLOCATOR_SYMBOL = "__ZN3mlx4core5metal9allocatorEv"
    _ACTIVE_OFFSET = 0xA8
    _PEAK_OFFSET = 0xB0
    _NUM_RESOURCES_OFFSET = 0xC8
    _RESOURCE_LIMIT_OFFSET = 0xD0

    def __init__(self, mx):
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise RuntimeError("Metal resource probe requires Darwin arm64")
        lib_path = (Path(mx.__file__).resolve().parent / "lib/libmlx.dylib").resolve()
        nm = subprocess.run(
            ["/usr/bin/nm", "-a", str(lib_path)],
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
            if raw_name and Path(raw_name.decode()).resolve() == lib_path:
                image_header = int(dyld._dyld_get_image_header(index))
                break
        if image_header is None:
            raise RuntimeError("loaded libmlx image was not found in dyld")
        allocator_fn = ctypes.CFUNCTYPE(ctypes.c_void_p)(image_header + symbol_value)
        self._address = int(allocator_fn())
        if not self._address:
            raise RuntimeError("private MLX allocator singleton was null")
        validations = {
            "active": (self._read(self._ACTIVE_OFFSET), int(mx.get_active_memory())),
            "peak": (self._read(self._PEAK_OFFSET), int(mx.get_peak_memory())),
            "limit": (self.limit(), int(mx.device_info()["resource_limit"])),
        }
        mismatches = {key: value for key, value in validations.items() if value[0] != value[1]}
        if mismatches:
            raise RuntimeError(f"private MLX allocator layout validation failed: {mismatches}")

    def _read(self, offset):
        return int(ctypes.c_size_t.from_address(self._address + offset).value)

    def current(self):
        return self._read(self._NUM_RESOURCES_OFFSET)

    def limit(self):
        return self._read(self._RESOURCE_LIMIT_OFFSET)


def load_cell_rows(args):
    if args.schedule.startswith("synthetic_"):
        return synthetic_rows(args.schedule.removeprefix("synthetic_"))
    bundle = read_gzip_json(args.bundle)
    if args.schedule == "finetome_144x4":
        cycle = bundle["selected_cycle_rows"]
        rows = cycle * 4
        return rows, [len(row["input_ids"]) for row in rows], 4, MAX_FINE_LENGTH
    rows = bundle["full_diversity_rows"]
    return rows, [len(row["input_ids"]) for row in rows], 1, MAX_FINE_LENGTH


def cell_checkpoint(path, metadata, steps, status, error=None):
    value = {"metadata": metadata, "status": status, "steps": steps}
    if error is not None:
        value["error"] = str(error)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    temporary.replace(path)


def run_cell(args):
    cell_started = time.perf_counter()
    source = args.source.resolve()
    sys.path.insert(0, str(source))

    import mlx.core as mx
    import numpy as np
    import psutil
    from huggingface_hub import snapshot_download

    from unsloth_zoo.mlx.loader import FastMLXModel
    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig
    import unsloth_zoo.mlx.utils as utils_module

    imported_source = Path(utils_module.__file__).resolve()
    if not imported_source.is_relative_to(source):
        raise RuntimeError(f"imported {imported_source}, expected source {source}")

    rows, widths, repetitions, max_length = load_cell_rows(args)
    dataset_digest = digest_json(rows)
    metadata = {
        "schedule": args.schedule,
        "mode": args.mode,
        "run_label": args.run_label,
        "rows": len(rows),
        "repetitions": repetitions,
        "raw_signatures": len(set(widths)),
        "schedule_digest": digest_json(widths),
        "dataset_digest": dataset_digest,
        "seed": SEED,
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=source, check=True,
            capture_output=True, text=True,
        ).stdout.strip(),
        "max_sequence_length": max_length,
        "memory_limit_gb": None,
        "wired_limit_gb": None,
        "cache_limit_gb": None,
        "clear_cache_during_training": False,
        "resource_stop": RESOURCE_STOP,
        "nice": os.getpriority(os.PRIO_PROCESS, 0),
        "runner_name": os.environ.get("RUNNER_NAME"),
        "runner_arch": os.environ.get("RUNNER_ARCH"),
    }
    cell_checkpoint(args.output, metadata, [], "starting")
    probe = PrivateMetalResourceProbe(mx)
    process = psutil.Process()
    model_setup_started = time.perf_counter()
    model_path = snapshot_download(MODEL_ID)

    mx.random.seed(SEED)
    model, tokenizer = FastMLXModel.from_pretrained(
        model_path,
        max_seq_length=max_length,
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
        use_gradient_checkpointing=True,
        max_seq_length=max_length,
    )
    mx.eval(model.parameters(), model.trainable_parameters())
    mx.synchronize()
    mx.clear_cache()
    mx.reset_peak_memory()
    mx.synchronize()
    model_setup_seconds = time.perf_counter() - model_setup_started

    materializations = []
    prepare_samples = []
    step_rows = []
    finite_plan_class = getattr(utils_module, "FiniteTextBatchPlan", None)
    original_materialize = None if finite_plan_class is None else finite_plan_class.materialize

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
            "active_bytes": int(mx.get_active_memory()),
            "cache_bytes": int(mx.get_cache_memory()),
            "rss_bytes": int(process.memory_info().rss),
        })
        return result

    MLXTrainer._prepare_data = recording_prepare
    config = dict(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        max_steps=len(rows),
        warmup_steps=0,
        learning_rate=2e-4,
        lr_scheduler_type="constant",
        logging_steps=1,
        save_steps=0,
        eval_steps=0,
        report_to="none",
        seed=SEED,
        max_seq_length=max_length,
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
    if args.mode.startswith("cap"):
        config["compile_max_variants"] = int(args.mode.removeprefix("cap"))
    unknown = set(config) - {field.name for field in fields(MLXTrainingConfig)}
    if unknown:
        raise RuntimeError(f"unknown config fields: {sorted(unknown)}")

    training_started = None
    trainer = None
    train_output = None
    try:
        with tempfile.TemporaryDirectory(prefix="pr6990-resource-cell-") as output_dir:
            config["output_dir"] = output_dir
            trainer = MLXTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=rows,
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
                    "active_bytes": int(mx.get_active_memory()),
                    "cache_bytes": int(mx.get_cache_memory()),
                    "peak_bytes": int(mx.get_peak_memory()),
                    "rss_bytes": int(process.memory_info().rss),
                })
                callback_index += 1
                if callback_index % 16 == 0:
                    cell_checkpoint(args.output, metadata, step_rows, "running")
                if step_rows[-1]["resources_current"] >= RESOURCE_STOP:
                    cell_checkpoint(
                        args.output,
                        metadata,
                        step_rows,
                        "resource_safety_stop",
                        f"Metal resource count reached {step_rows[-1]['resources_current']}",
                    )
                    os._exit(75)

            trainer.add_step_callback(on_step)
            training_started = time.perf_counter()
            train_output = trainer.train()
            mx.synchronize()
            train_wall_seconds = time.perf_counter() - training_started
    except Exception as error:
        cell_checkpoint(args.output, metadata, step_rows, "failed", error)
        raise
    finally:
        if finite_plan_class is not None:
            finite_plan_class.materialize = original_materialize
        MLXTrainer._prepare_data = original_prepare

    if [row["raw_width"] for row in step_rows] != list(widths):
        raise RuntimeError("training order or membership changed")
    for row, step_time in zip(step_rows, trainer._step_times):
        row["step_time_seconds"] = float(step_time)

    resources_before_clear = probe.current()
    active_before_clear = int(mx.get_active_memory())
    cache_before_clear = int(mx.get_cache_memory())
    peak_before_clear = int(mx.get_peak_memory())
    rss_before_clear = int(process.memory_info().rss)
    mx.clear_cache()
    mx.synchronize()
    resources_after_clear = probe.current()

    seen_targets = set()
    for row in step_rows:
        signature = (row["phase"], row["target_width"])
        row["new_target_signature"] = signature not in seen_targets
        seen_targets.add(signature)
    cycle_size = len(rows) // repetitions
    cycles = []
    for cycle_index in range(repetitions):
        cycle_rows = step_rows[cycle_index * cycle_size:(cycle_index + 1) * cycle_size]
        seconds = sum(row["step_time_seconds"] for row in cycle_rows)
        useful_tokens = sum(row["raw_width"] - 1 for row in cycle_rows)
        cycles.append({
            "cycle": cycle_index + 1,
            "seconds": seconds,
            "useful_tokens": useful_tokens,
            "useful_tokens_per_second": useful_tokens / seconds,
            "resources_end": cycle_rows[-1]["resources_current"],
            "rss_bytes_end": cycle_rows[-1]["rss_bytes"],
        })
    raw_work = sum(width * width for width in widths)
    target_work = sum(row["target_width"] ** 2 for row in step_rows)
    useful_tokens = sum(width - 1 for width in widths)
    padded_tokens = sum(row["target_width"] - row["raw_width"] for row in step_rows)
    metrics = {
        "prepare": prepare_samples[-1],
        "model_setup_seconds": model_setup_seconds,
        "train_wall_seconds": train_wall_seconds,
        "total_cell_wall_seconds": time.perf_counter() - cell_started,
        "useful_tokens": useful_tokens,
        "useful_tokens_per_second": useful_tokens / train_wall_seconds,
        "trained_tokens_reported": int(train_output["trained_tokens"]),
        "observed_target_signature_count": len(seen_targets),
        "padding_tokens": padded_tokens,
        "padding_token_fraction": padded_tokens / sum(widths),
        "added_quadratic_work_fraction": target_work / raw_work - 1.0,
        "first_cycle_seconds": cycles[0]["seconds"],
        "reuse_cycle_seconds": [cycle["seconds"] for cycle in cycles[1:]],
        "reuse_cycle_mean_seconds": (
            sum(cycle["seconds"] for cycle in cycles[1:]) / (repetitions - 1)
            if repetitions > 1 else None
        ),
        "cycles": cycles,
        "resources_peak_sampled": max(
            resources_before_clear,
            max(row["resources_current"] for row in step_rows),
        ),
        "resources_before_final_clear": resources_before_clear,
        "resources_after_final_clear": resources_after_clear,
        "resources_retained_training_delta": resources_after_clear - prepare_samples[-1]["resources"],
        "active_bytes_before_final_clear": active_before_clear,
        "cache_bytes_before_final_clear": cache_before_clear,
        "peak_bytes_before_final_clear": peak_before_clear,
        "rss_bytes_before_final_clear": rss_before_clear,
        "rss_peak_sampled_bytes": max(row["rss_bytes"] for row in step_rows),
        "loss_final": float(trainer._train_loss_history[-1]),
        "finite_losses": bool(np.isfinite(trainer._train_loss_history).all()),
        "shape_guard": train_output.get("compile_shape_guard"),
        "compile_enabled": bool(train_output["compile_enabled"]),
        "compile_scope": train_output["compile_scope"],
    }
    metadata.update({
        "resource_limit": probe.limit(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "device_info": mx.device_info(),
        "versions": {
            name: package_version(name)
            for name in ("mlx", "mlx-lm", "mlx-vlm", "numpy", "torch")
        },
    })
    result = {"metadata": metadata, "status": "completed", "metrics": metrics, "steps": step_rows}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True))
    print("CELL_RESULT_JSON=" + json.dumps({
        "schedule": args.schedule,
        "mode": args.mode,
        "seconds": round(train_wall_seconds, 3),
        "useful_tokens_per_second": round(metrics["useful_tokens_per_second"], 3),
        "signatures": len(seen_targets),
        "peak_resources": metrics["resources_peak_sampled"],
        "retained_resources": metrics["resources_retained_training_delta"],
    }, sort_keys=True))


def find_competing_processes():
    output = subprocess.run(
        ["ps", "-axo", "pid=,ppid=,command="],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    competitors = []
    for line in output.splitlines():
        parts = line.strip().split(maxsplit=2)
        if len(parts) != 3:
            continue
        pid, _ppid, command = parts
        lower = command.lower()
        if int(pid) == os.getpid():
            continue
        if "python" in lower and any(term in lower for term in ("mlx", "train", "benchmark")):
            competitors.append(line.strip())
    return competitors


def mode_source(mode, main_source, feature_source):
    return main_source if mode == "main" else feature_source


def run_subprocess_cell(
    args,
    schedule,
    mode,
    run_label,
    output_dir,
    timeout,
    run_number,
):
    competitors = find_competing_processes()
    if competitors:
        raise RuntimeError("competing training processes found:\n" + "\n".join(competitors))
    cell_name = f"{run_number:02d}__{schedule}__{mode}"
    output = output_dir / f"{cell_name}.json"
    log = output_dir / f"{cell_name}.log"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "cell",
        "--source", str(mode_source(mode, args.main_source, args.feature_source)),
        "--schedule", schedule,
        "--mode", mode,
        "--run-label", run_label,
        "--output", str(output),
        "--log", str(log),
    ]
    if args.bundle:
        command.extend(("--bundle", str(args.bundle)))
    print(f"CELL_START schedule={schedule} mode={mode} timeout={timeout}s", flush=True)
    started = time.monotonic()
    with log.open("a", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            ["nice", "-n", "10", *command],
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        while process.poll() is None:
            elapsed = time.monotonic() - started
            if elapsed >= timeout:
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                return {
                    "schedule": schedule,
                    "mode": mode,
                    "run_label": run_label,
                    "status": "timeout",
                    "seconds": elapsed,
                    "output": str(output),
                    "log": str(log),
                }
            if int(elapsed) and int(elapsed) % 60 == 0:
                print(f"CELL_HEARTBEAT schedule={schedule} mode={mode} elapsed={elapsed:.0f}s", flush=True)
            time.sleep(1)
    elapsed = time.monotonic() - started
    if output.exists():
        result = json.loads(output.read_text())
    else:
        result = {"status": "missing_output"}
    status = result.get("status", "unknown")
    print(
        f"CELL_END schedule={schedule} mode={mode} status={status} "
        f"exit={process.returncode} elapsed={elapsed:.1f}s",
        flush=True,
    )
    if process.returncode not in (0, 75):
        tail = log.read_text(errors="replace").splitlines()[-60:]
        print("\n".join(tail), file=sys.stderr)
    return {
        "schedule": schedule,
        "mode": mode,
        "run_label": run_label,
        "status": status,
        "exit_code": process.returncode,
        "elapsed_seconds": elapsed,
        "output": str(output),
        "log": str(log),
    }


def matrix_plan(kind, only_schedule=None):
    if kind == "synthetic":
        schedules = [only_schedule] if only_schedule else ["small", "clustered", "irregular", "long_tail"]
        return [
            (f"synthetic_{schedule}", mode, "single-pass")
            for schedule in schedules
            for mode in SYNTHETIC_MODES
        ]
    return [
        ("finetome_144x4", "main", "pass-1-main-first"),
        ("finetome_144x4", "auto", "pass-1-middle"),
        ("finetome_144x4", "cap256", "pass-1-exact-last"),
        ("finetome_144x4", "cap256", "pass-2-exact-first"),
        ("finetome_144x4", "auto", "pass-2-middle"),
        ("finetome_144x4", "main", "pass-2-main-last"),
        ("finetome_full_diversity", "auto", "full-diversity"),
    ]


def render_summary(kind, records, output_dir, prescan=None):
    completed = []
    for record in records:
        path = Path(record.get("output", ""))
        if record["status"] == "completed" and path.exists():
            completed.append(json.loads(path.read_text()))
    summary = {
        "kind": kind,
        "records": records,
        "prescan": prescan,
        "completed_cells": len(completed),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    lines = [
        f"# PR 6990 {kind} benchmark",
        "",
        "| Schedule | Run | Mode | Status | Time s | Useful tok/s | Signatures | Peak resources | Retained resources | Padding |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        path = Path(record.get("output", ""))
        result = json.loads(path.read_text()) if record["status"] == "completed" and path.exists() else None
        if result is None:
            lines.append(
                f"| {record['schedule']} | {record['run_label']} | "
                f"{record['mode']} | {record['status']} |  |  |  |  |  |  |"
            )
            continue
        metrics = result["metrics"]
        lines.append(
            f"| {record['schedule']} | {record['run_label']} | "
            f"{record['mode']} | completed | "
            f"{metrics['train_wall_seconds']:.2f} | {metrics['useful_tokens_per_second']:.1f} | "
            f"{metrics['observed_target_signature_count']} | {metrics['resources_peak_sampled']} | "
            f"{metrics['resources_retained_training_delta']} | {100 * metrics['padding_token_fraction']:.2f}% |"
        )
    report = "\n".join(lines) + "\n"
    (output_dir / "summary.md").write_text(report)
    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a", encoding="utf-8") as handle:
            handle.write(report)


def run_matrix(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    records = []
    unsafe_modes = set()
    for run_number, (schedule, mode, run_label) in enumerate(
        matrix_plan(args.kind, args.only_schedule),
        start=1,
    ):
        if mode in unsafe_modes:
            records.append({
                "schedule": schedule,
                "mode": mode,
                "run_label": run_label,
                "status": "skipped_after_resource_stop",
            })
            continue
        remaining = args.budget_seconds - (time.monotonic() - started)
        if remaining < 300:
            records.append({
                "schedule": schedule,
                "mode": mode,
                "run_label": run_label,
                "status": "skipped_time_budget",
            })
            continue
        timeout = int(min(args.cell_timeout, remaining - 180))
        record = run_subprocess_cell(
            args,
            schedule,
            mode,
            run_label,
            args.output_dir,
            timeout,
            run_number,
        )
        records.append(record)
        if record["status"] == "resource_safety_stop":
            unsafe_modes.add(mode)
    prescan = None
    if args.bundle:
        prescan = read_gzip_json(args.bundle)["metadata"]
    render_summary(args.kind, records, args.output_dir, prescan)
    failures = [record for record in records if record["status"] not in (
        "completed", "resource_safety_stop", "skipped_after_resource_stop", "skipped_time_budget",
    )]
    return 1 if failures else 0


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--stats", type=Path, required=True)
    cell = subparsers.add_parser("cell")
    cell.add_argument("--source", type=Path, required=True)
    cell.add_argument("--schedule", required=True)
    cell.add_argument("--mode", choices=SYNTHETIC_MODES, required=True)
    cell.add_argument("--run-label", required=True)
    cell.add_argument("--bundle", type=Path)
    cell.add_argument("--output", type=Path, required=True)
    cell.add_argument("--log", type=Path, required=True)
    matrix = subparsers.add_parser("matrix")
    matrix.add_argument("--kind", choices=("synthetic", "finetome"), required=True)
    matrix.add_argument("--main-source", type=Path, required=True)
    matrix.add_argument("--feature-source", type=Path, required=True)
    matrix.add_argument("--bundle", type=Path)
    matrix.add_argument("--only-schedule", choices=("small", "clustered", "irregular", "long_tail"))
    matrix.add_argument("--output-dir", type=Path, required=True)
    matrix.add_argument("--cell-timeout", type=int, required=True)
    matrix.add_argument("--budget-seconds", type=int, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.action == "prepare":
        prepare_finetome(args)
        return 0
    if args.action == "cell":
        run_cell(args)
        return 0
    return run_matrix(args)


if __name__ == "__main__":
    raise SystemExit(main())
