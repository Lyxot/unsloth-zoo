#!/usr/bin/env python3
"""Compare one real Gemma4 VLM training step eager versus compiled."""

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
RESULT_MARKER = "GEMMA4_HOSTED_DIAG:"


def _load_fixture_builder():
    runner_path = Path(__file__).with_name(
        "mlx_vlm_shape_guard_training_ci.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_gemma4_fixture_runner", runner_path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _array_digest(value):
    import numpy as np

    array = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(str((array.shape, array.dtype)).encode())
    digest.update(array.tobytes())
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": digest.hexdigest(),
    }


def child_main(payload_text):
    payload = json.loads(payload_text)
    source = Path(payload["source"]).resolve()
    model_path = Path(payload["model_path"]).resolve()
    compiled = bool(payload["compiled"])
    sys.path.insert(0, str(source))

    import mlx.core as mx
    import numpy as np
    from unsloth_zoo.mlx.loader import FastMLXModel
    import unsloth_zoo.mlx.trainer as trainer_module
    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig

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

    batch_record = {}
    original_create = trainer_module.create_vlm_batches

    def recording_create(*args, **kwargs):
        batches = original_create(*args, **kwargs)
        batch = batches[0]
        mx.eval(
            [
                value
                for value in batch.values()
                if isinstance(value, mx.array)
            ]
        )
        for key in (
            "input_ids",
            "labels",
            "attention_mask",
            "pixel_values",
            "mm_token_type_ids",
        ):
            if key in batch and isinstance(batch[key], mx.array):
                batch_record[key] = _array_digest(batch[key])
        labels = np.asarray(batch["labels"])
        attention = np.asarray(batch["attention_mask"])
        batch_record["supervised_tokens"] = int(
            np.logical_and(
                labels[:, 1:] != -100,
                attention[:, 1:] != 0,
            ).sum()
        )
        return batches

    trainer_module.create_vlm_batches = recording_create
    output_dir = Path(tempfile.mkdtemp(prefix="gemma4-hosted-diag-"))
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
            output_dir=str(output_dir),
            report_to="none",
            save_steps=0,
            eval_steps=0,
            max_seq_length=384,
            use_cce=True,
            compile=compiled,
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
            args=args,
        )
        trainer.save_model = lambda *args, **kwargs: None
        started = time.perf_counter()
        output = trainer.train()
        mx.synchronize()
        elapsed = time.perf_counter() - started
    finally:
        trainer_module.create_vlm_batches = original_create
        shutil.rmtree(output_dir, ignore_errors=True)

    result = {
        "mode": "compiled" if compiled else "eager",
        "model_revision": model_path.name,
        "fixture_row": manifest[0]["row_id"],
        "fixture_messages_sha256": manifest[0]["messages_sha256"],
        "fixture_image_sha256": manifest[0]["image_sha256"],
        "batch": batch_record,
        "loss": float(output["train_loss"]),
        "trained_tokens": int(output["trained_tokens"]),
        "compile_enabled": bool(output["compile_enabled"]),
        "compile_scope": output["compile_scope"],
        "elapsed_seconds": elapsed,
        "mlx_peak_bytes": int(mx.get_peak_memory()),
    }
    print(RESULT_MARKER + json.dumps(result, sort_keys=True), flush=True)


def run_child(source, model_path, compiled, log_path):
    payload = json.dumps(
        {
            "source": str(source),
            "model_path": str(model_path),
            "compiled": compiled,
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
    log_path.write_text(text)
    print(text, end="")
    if completed.returncode:
        raise SystemExit(completed.returncode)
    markers = [
        line.removeprefix(RESULT_MARKER)
        for line in text.splitlines()
        if line.startswith(RESULT_MARKER)
    ]
    if not markers:
        raise RuntimeError("diagnostic child emitted no result")
    return json.loads(markers[-1])


def parent_main(source, model_path, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for compiled in (False, True):
        mode = "compiled" if compiled else "eager"
        results[mode] = run_child(
            source,
            model_path,
            compiled,
            output_dir / f"{mode}.log",
        )
    if results["eager"]["batch"] != results["compiled"]["batch"]:
        raise RuntimeError("eager and compiled batch digests differ")
    if results["eager"]["trained_tokens"] != results["compiled"]["trained_tokens"]:
        raise RuntimeError("eager and compiled trained-token counts differ")
    summary = {
        "status": "completed",
        **results,
        "loss_relative_difference": abs(
            results["compiled"]["loss"] - results["eager"]["loss"]
        )
        / max(abs(results["eager"]["loss"]), 1e-12),
    }
    (output_dir / "result.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    print(RESULT_MARKER + json.dumps(summary, sort_keys=True))


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
