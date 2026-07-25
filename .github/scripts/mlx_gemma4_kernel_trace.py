#!/usr/bin/env python3
"""Trace the first real Gemma4 VLM forward at module boundaries."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time


SEED = 3407
RESULT_MARKER = "GEMMA4_KERNEL_TRACE:"
TRACE_CLASS_NAMES = {
    "Attention",
    "DecoderLayer",
    "Gemma4TextModel",
    "MLP",
    "Model",
    "MultimodalEmbedder",
    "VisionAttention",
    "VisionMLP",
    "VisionModel",
    "VisionPatchEmbedder",
    "VisionPooler",
    "VisionTransformerBlock",
}


def _load_fixture_builder():
    runner_path = Path(__file__).with_name("mlx_vlm_shape_guard_training_ci.py")
    spec = importlib.util.spec_from_file_location(
        "_gemma4_fixture_runner", runner_path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _first_array(value, mx):
    if isinstance(value, mx.array):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            found = _first_array(item, mx)
            if found is not None:
                return found
    for attribute in ("logits", "inputs_embeds"):
        item = getattr(value, attribute, None)
        if isinstance(item, mx.array):
            return item
    return None


def _install_boundary_trace(model, records):
    import mlx.core as mx

    module_names = {
        id(module): name or "<root>"
        for name, module in model.named_modules()
    }
    originals = []
    seen_classes = set()
    for _name, module in model.named_modules():
        cls = type(module)
        if cls in seen_classes or cls.__name__ not in TRACE_CLASS_NAMES:
            continue
        seen_classes.add(cls)
        original = cls.__call__

        def traced_call(self, *args, __original=original, **kwargs):
            name = module_names.get(id(self))
            if name is not None:
                value = _first_array(args, mx)
                if value is not None:
                    records.append((f"{name}:input", value))
            output = __original(self, *args, **kwargs)
            if name is not None:
                value = _first_array(output, mx)
                if value is not None:
                    records.append((f"{name}:output", value))
            return output

        cls.__call__ = traced_call
        originals.append((cls, original))
    return originals


def _restore_trace(originals):
    for cls, original in reversed(originals):
        cls.__call__ = original


def _trace_payload(label, value):
    import mlx.core as mx
    import numpy as np

    value32 = value.astype(mx.float32)
    flat = value32.reshape(-1)
    size = int(flat.size)
    sample_count = min(size, 1024)
    indices = np.unique(
        np.linspace(0, max(size - 1, 0), sample_count, dtype=np.int64)
    )
    sample = flat[mx.array(indices)]
    finite = mx.isfinite(value32)
    safe = mx.where(finite, value32, mx.zeros_like(value32))
    stats = mx.stack(
        [
            safe.mean(),
            mx.sqrt(mx.mean(mx.square(safe))),
            safe.min(),
            safe.max(),
            finite.astype(mx.float32).mean(),
        ]
    )
    return {
        "label": label,
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "_sample": sample,
        "_stats": stats,
    }


def _finish_payload(item):
    import numpy as np

    sample = np.asarray(item.pop("_sample"))
    stats = np.asarray(item.pop("_stats"))
    digest = hashlib.sha256()
    digest.update(str((sample.shape, sample.dtype)).encode())
    digest.update(sample.tobytes())
    item.update(
        {
            "sample_sha256": digest.hexdigest(),
            "sample": sample.tolist(),
            "mean": float(stats[0]),
            "rms": float(stats[1]),
            "min": float(stats[2]),
            "max": float(stats[3]),
            "finite_fraction": float(stats[4]),
        }
    )
    return item


def child_main(payload_text):
    payload = json.loads(payload_text)
    source = Path(payload["source"]).resolve()
    model_path = Path(payload["model_path"]).resolve()
    output_dir = Path(payload["output_dir"]).resolve()
    sys.path.insert(0, str(source))

    import mlx.core as mx
    from unsloth_zoo.mlx.loader import FastMLXModel
    import unsloth_zoo.mlx.trainer as trainer_module
    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig
    from unsloth_zoo.mlx.utils import (
        iter_mlx_norm_output_cast_classes,
        make_vlm_baseline_loss_fn,
        restore_mlx_norm_output_cast_state,
        set_mlx_norm_output_cast_to_input_dtype,
        snapshot_mlx_norm_output_cast_state,
    )

    try:
        os.nice(10)
    except OSError:
        pass
    mx.random.seed(SEED)
    fixture = _load_fixture_builder()
    rows, manifest = fixture.make_fixture()

    model, processor = FastMLXModel.from_pretrained(
        str(model_path),
        max_seq_length=384,
        text_only=False,
        load_in_4bit=True,
        random_state=SEED,
    )
    model = FastMLXModel.get_peft_model(
        model,
        r=4,
        lora_alpha=4,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="mlx",
        random_state=SEED,
        max_seq_length=384,
        train_vision=False,
        train_projector=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
    )
    mx.eval(model.parameters(), model.trainable_parameters())
    mx.synchronize()

    class BatchCaptured(Exception):
        pass

    captured = []
    original_create = trainer_module.create_vlm_batches

    def capture_create(*args, **kwargs):
        batches = original_create(*args, **kwargs)
        captured.append(batches[0])
        raise BatchCaptured

    trainer_module.create_vlm_batches = capture_create
    trainer_output = Path(tempfile.mkdtemp(prefix="gemma4-kernel-trace-"))
    try:
        args = MLXTrainingConfig(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            max_steps=1,
            warmup_steps=5,
            learning_rate=2e-4,
            logging_steps=1,
            optim="adamw",
            weight_decay=0.001,
            lr_scheduler_type="linear",
            seed=SEED,
            output_dir=str(trainer_output),
            report_to="none",
            save_steps=0,
            eval_steps=0,
            max_seq_length=384,
            use_cce=False,
            compile=False,
            gradient_checkpointing=True,
            dataset_order="sequential",
            preserve_dataset_order=True,
            completion_only_loss=False,
            append_eos=False,
            disable_memory_limits=True,
        )
        trainer = MLXTrainer(
            model=model,
            tokenizer=processor,
            processor=processor,
            train_dataset=rows,
            args=args,
        )
        try:
            trainer.train()
        except BatchCaptured:
            pass
    finally:
        trainer_module.create_vlm_batches = original_create
        shutil.rmtree(trainer_output, ignore_errors=True)
    if not captured:
        raise RuntimeError("failed to capture the first real training batch")
    batch = captured[0]
    mx.eval([value for value in batch.values() if isinstance(value, mx.array)])

    norm_state = snapshot_mlx_norm_output_cast_state(
        iter_mlx_norm_output_cast_classes(model)
    )
    set_mlx_norm_output_cast_to_input_dtype(True, model)
    boundary_records = []
    originals = _install_boundary_trace(model, boundary_records)
    try:
        loss_fn = make_vlm_baseline_loss_fn(model)
        started = time.perf_counter()
        loss, tokens = loss_fn(model, batch)
        trace_items = [
            _trace_payload(label, value)
            for label, value in boundary_records
        ]
        mx.eval(loss, tokens, *[
            value
            for item in trace_items
            for value in (item["_sample"], item["_stats"])
        ])
        mx.synchronize()
        elapsed = time.perf_counter() - started
    finally:
        _restore_trace(originals)
        restore_mlx_norm_output_cast_state(norm_state)

    result = {
        "status": "completed",
        "model_revision": model_path.name,
        "device_info": mx.device_info(),
        "fixture_row": manifest[0]["row_id"],
        "fixture_messages_sha256": manifest[0]["messages_sha256"],
        "fixture_image_sha256": manifest[0]["image_sha256"],
        "loss": float(loss.item()),
        "supervised_tokens": int(tokens.item()),
        "elapsed_seconds": elapsed,
        "mlx_peak_bytes": int(mx.get_peak_memory()),
        "traces": [_finish_payload(item) for item in trace_items],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "trace.json").write_text(
        json.dumps(result, indent=2, sort_keys=True)
    )
    print(
        RESULT_MARKER
        + json.dumps(
            {
                key: value
                for key, value in result.items()
                if key != "traces"
            }
            | {"trace_count": len(result["traces"])},
            sort_keys=True,
        ),
        flush=True,
    )


def parent_main(source, model_path, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        {
            "source": str(source),
            "model_path": str(model_path),
            "output_dir": str(output_dir),
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
            "PYTHONPATH": str(source),
        }
    )
    completed = subprocess.run(
        [sys.executable, __file__, "--child", payload],
        env=env,
        cwd=source,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    text = completed.stdout + completed.stderr
    (output_dir / "job.log").write_text(text)
    print(text, end="")
    if completed.returncode:
        raise SystemExit(completed.returncode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--child")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.child:
        child_main(args.child)
        return
    parent_main(
        args.source.resolve(),
        args.model_path.resolve(),
        args.output_dir.resolve(),
    )


if __name__ == "__main__":
    main()
