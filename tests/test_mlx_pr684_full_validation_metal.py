"""Deep behavioral validation of the PR 684 MLX trainer rework on Apple Silicon.

Complements test_mlx_training_e2e_metal.py (basic text LoRA smoke) with the
paths that only real Metal training can prove:

1. resume_from_checkpoint determinism: a stop+resume run reproduces the
   fresh run's losses step for step (validates the #751 resume logic inside
   the reworked training loop: optimizer state restore, batch fast-forward,
   LR schedule offset).
2. train_on_responses_only completion-only training: exact step count for
   epoch-based runs (pins the epoch double-counting fix) and finite losses
   through the labeled-batch path.
3. Epoch-based unlabeled runs: num_train_epochs drives the step count when
   max_steps is disabled.
4. SGD with gradient-coupled weight decay end to end.
5. Real VLM LoRA training: tiny 4-bit SmolVLM through the VLM collation,
   label masking, CCE loss, and adapter save pipeline.
"""

import gc
import glob
import json
import os

import pytest

try:
    import mlx.core as mx
    _METAL = mx.metal.is_available()
except Exception:
    _METAL = False

if not _METAL:
    print("NOTICE: Metal unavailable; PR 684 full-validation tests will be skipped.")

metal_only = pytest.mark.skipif(not _METAL, reason="requires Apple Silicon Metal")


def _simulation_shim_installed():
    """Whether another test module has swapped MLX for the torch double."""
    import sys

    return "mlx_simulation" in sys.modules or "mlx_simulation" in getattr(
        sys.modules.get("mlx.core"), "__name__", "",
    )


real_runtime_only = pytest.mark.skipif(
    not _METAL, reason="requires Apple Silicon Metal",
)

TEXT_MODEL = "mlx-community/SmolLM-135M-Instruct-4bit"
# Qwen2-VL: smallest VLM whose processor resolves cleanly under current
# transformers (the mlx-community SmolVLM-256M repo ships a preprocessor
# config AutoImageProcessor cannot map).
VLM_MODEL = "mlx-community/Qwen2-VL-2B-Instruct-4bit"


def _chat_dataset(n=12):
    # ChatML matches SmolLM-Instruct's template so response masking can
    # anchor on the literal role markers.
    return [
        {
            "text": (
                f"<|im_start|>user\nWhat is {i} plus {i}?<|im_end|>\n"
                f"<|im_start|>assistant\nThe answer is {2 * i}.<|im_end|>\n"
            )
        }
        for i in range(n)
    ]


def _make_text_trainer(tmp_path, dataset, **config_overrides):
    from unsloth_zoo.mlx.loader import FastMLXModel
    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig

    model, tokenizer = FastMLXModel.from_pretrained(TEXT_MODEL, max_seq_length=256)
    model = FastMLXModel.get_peft_model(model, r=8, lora_alpha=16, lora_dropout=0)
    config = dict(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        max_steps=8,
        warmup_steps=2,
        learning_rate=5e-4,
        logging_steps=1,
        output_dir=str(tmp_path),
        seed=3407,
        report_to="none",
    )
    config.update(config_overrides)
    return MLXTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=MLXTrainingConfig(**config),
    )


def _assert_finite(hist):
    assert all(
        isinstance(l, float) and l == l and abs(l) != float("inf") for l in hist
    ), f"non-finite losses: {hist}"


def _tiny_text_model(family, **overrides):
    """A real architecture at toy size, with random weights and no download."""
    import importlib

    if _simulation_shim_installed():
        pytest.skip("the MLX simulation shim is installed in this process")
    module = importlib.import_module(f"mlx_lm.models.{family}")
    return module.Model(module.ModelArgs(model_type=family, **overrides))


def _tiny_qwen2(layers=2, vocab=128):
    return _tiny_text_model(
        "qwen2", hidden_size=32, num_hidden_layers=layers, intermediate_size=64,
        num_attention_heads=2, rms_norm_eps=1e-5, vocab_size=vocab,
        num_key_value_heads=2,
    )


def test_attention_that_bypasses_the_patch_is_refused():
    """The interception count is what keeps non-text attention out."""
    from unsloth_zoo.mlx.compile import _probe_refusal

    import mlx.nn as nn

    class HandRolledAttention(nn.Module):
        """Attends without the fused entry point, as gemma2 and olmo do."""

        def __call__(self, x, *args, **kwargs):
            return x

    model = _tiny_qwen2(layers=2)
    # Whatever this model's verdict is, it is not about the layer count, so
    # the assertion below cannot be satisfied by an unrelated refusal.
    baseline = _probe_refusal(model, 32, 128)
    assert baseline is None or "layers" not in baseline

    model.model.layers.append(HandRolledAttention())

    refusal = _probe_refusal(model, 32, 128)
    assert refusal is not None
    assert "3 layers" in refusal and "2 calls" in refusal

    # The other direction is the load-bearing one, and an under-count-only
    # check would pass everything above while still admitting it: a second
    # consumer of this entry point sharing the batch's geometry is exactly
    # what the mask composer cannot tell apart by itself.
    class SecondConsumer(nn.Module):
        """A layer that reaches the shared entry point one extra time."""

        def __init__(self, wrapped):
            super().__init__()
            self.wrapped = wrapped

        def __call__(self, x, *args, **kwargs):
            spare = mx.fast.scaled_dot_product_attention(
                x[:, None], x[:, None], x[:, None], scale=1.0, mask=None,
            )
            return self.wrapped(x, *args, **kwargs) + 0 * spare[:, 0]

    extra = _tiny_qwen2(layers=2)
    extra.model.layers[0] = SecondConsumer(extra.model.layers[0])

    surplus = _probe_refusal(extra, 32, 128)
    assert surplus is not None
    assert "3 calls" in surplus and "2 layers" in surplus

    class TrainingOnlyConsumer(nn.Module):
        """Switched by the mode hook, not by the flag."""

        modes_seen = []

        def _set_training_mode(self, mode):
            type(self).modes_seen.append(mode)
            super()._set_training_mode(mode)

        """Reaches the entry point only while training, as a dropout-guarded
        or checkpointing attention path would."""

        def __init__(self, wrapped):
            super().__init__()
            self.wrapped = wrapped

        def __call__(self, x, *args, **kwargs):
            out = self.wrapped(x, *args, **kwargs)
            if self.training:
                spare = mx.fast.scaled_dot_product_attention(
                    x[:, None], x[:, None], x[:, None], scale=1.0, mask=None,
                )
                out = out + 0 * spare[:, 0]
            return out

    # The whole probe runs in training mode, so a consumer that only appears
    # there is seen by the ordinary count. Dropout stays active and the
    # comparisons replay the same random state rather than switching it off.
    hidden = _tiny_qwen2(layers=2)
    hidden.model.layers[0] = TrainingOnlyConsumer(hidden.model.layers[0])
    # Left in evaluation mode, which is how a loaded model arrives: the
    # trainer switches it afterwards. Calling train() here first would let the
    # probe inherit the right answer instead of establishing it.
    hidden.eval()
    assert hidden.training is False

    while_training = _probe_refusal(hidden, 32, 128)
    assert while_training is not None
    assert "3 calls" in while_training and "2 layers" in while_training
    assert hidden.training is False
    # The transition went through the module's own hook in both directions,
    # which is what the trainer's mode switch does.
    assert set(TrainingOnlyConsumer.modes_seen) == {True, False}


def test_isolation_is_required_to_survive_the_backward():
    """Forward isolation does not imply backward isolation, so both are asked.

    Exercised directly rather than through the whole probe: with a mask that
    leaks, the forward checks refuse first and this stage is never reached, so
    a test going through the front door would pass without running it.
    """
    import numpy as np

    from unsloth_zoo.mlx.compile import (
        SegmentMaskBuffers, _backward_isolation_refusal, _probe_document_cuts,
        _probe_segments, install_safe_sdpa_mask_patch,
        make_segment_mask_composer, set_sdpa_mask_composer,
    )

    model = _tiny_qwen2(layers=2)
    length, boundary, vocab = 64, 32, 128
    tokens = (mx.arange(length, dtype=mx.int32) % vocab)[None]
    disturbed = mx.array(np.array(tokens))
    disturbed[0, boundary - 8:boundary] = (
        disturbed[0, boundary - 8:boundary] + 1
    ) % boundary
    cuts = _probe_document_cuts(length)
    segments = _probe_segments(length, cuts)
    # The last document, observed while an earlier one is disturbed.
    observed = mx.array(list(range(cuts[-1], length)))
    normaliser = float(length * vocab)

    install_safe_sdpa_mask_patch()
    buffers = SegmentMaskBuffers()
    previous = set_sdpa_mask_composer(make_segment_mask_composer(buffers.read))
    try:
        buffers.engage("train", segments)
        assert _backward_isolation_refusal(
            model, tokens, disturbed, observed, normaliser,
            window=lambda: buffers.within("train"),
        ) is None

        # The same call with nothing masking the documents apart: the first
        # document now reaches the second, so the second's gradient moves.
        buffers.engage("train", None)
        leaked = _backward_isolation_refusal(
            model, tokens, disturbed, observed, normaliser,
            window=lambda: buffers.within("train"),
        )
    finally:
        buffers.engage("train", None)
        set_sdpa_mask_composer(previous)

    assert leaked is not None
    assert "does not survive the backward" in leaked


@pytest.mark.parametrize("length", [64, 256])
def test_a_single_wrongly_permitted_position_is_caught(length):
    """Isolation is bitwise, and the perturbation has to be able to see it."""
    import unsloth_zoo.mlx.compile as compile_module
    from unsloth_zoo.mlx.compile import _probe_refusal

    # Vocabulary covering the row, so the two documents hold separate
    # embedding rows and the probe will build a stimulus at all.
    model = _tiny_qwen2(layers=2, vocab=512)
    assert _probe_refusal(model, length, 512) is None

    honest = compile_module.make_segment_mask_composer

    def leaking(read):
        inner = honest(read)

        def compose(q, k, mask):
            composed = inner(q, k, mask)
            if composed is None or composed.dtype != mx.bool_:
                return composed
            if composed.shape[2] < 4:
                return composed
            leaked = mx.array(composed)
            # One position of the first document, reachable by a late query of
            # the second, and far from the boundary between them.
            leaked[:, :, composed.shape[2] - 2, 0] = True
            return leaked

        return compose

    try:
        compile_module.make_segment_mask_composer = leaking
        refusal = _probe_refusal(model, length, 512)
    finally:
        compile_module.make_segment_mask_composer = honest

    assert refusal is not None
    assert "moved another document" in refusal


def test_engagement_retraces_the_compiled_step():
    """Why registering the buffer is what makes the window safe under compile."""
    from unsloth_zoo.mlx.compile import (
        SegmentMaskBuffers, install_safe_sdpa_mask_patch,
        make_segment_mask_composer, set_sdpa_mask_composer,
    )

    buffers = SegmentMaskBuffers()
    composed = []
    honest = make_segment_mask_composer(buffers.read)

    def watching(q, k, mask):
        result = honest(q, k, mask)
        composed.append(result is not None)
        return result

    install_safe_sdpa_mask_patch()
    previous = set_sdpa_mask_composer(watching)
    try:
        state = [buffers.buffer("train")]

        def attend(q, k, v):
            return mx.fast.scaled_dot_product_attention(
                q, k, v, scale=1.0, mask="causal",
            )

        step = mx.compile(attend, inputs=state, outputs=state)
        ones = mx.zeros((1, 2, 4, 8))

        buffers.engage("train", None)
        with buffers.within("train"):
            mx.eval(step(ones, ones, ones))
        assert composed == [False]

        # Same shapes, so the graph would be reused if the buffer were not an input.
        buffers.engage("train", mx.array([[0, 0, 1, 1]]))
        with buffers.within("train"):
            mx.eval(step(ones, ones, ones))
        assert composed == [False, True]

        # Crossing the window with the engagement unchanged must retrace.
        crossing = SegmentMaskBuffers()
        crossing.engage("train", mx.array([[0, 0, 1, 1]]))
        seen = []
        honest_again = make_segment_mask_composer(crossing.read)

        def counting(q, k, mask):
            result = honest_again(q, k, mask)
            seen.append(result is not None)
            return result

        set_sdpa_mask_composer(counting)
        def attend_again(q, k, v):
            return mx.fast.scaled_dot_product_attention(
                q, k, v, scale=1.0, mask="causal",
            )

        crossed = mx.compile(
            attend_again, inputs=[crossing.buffer("train")],
            outputs=[crossing.buffer("train")],
        )
        varied = mx.random.normal((1, 2, 4, 8))
        outside = crossed(varied, varied, varied)
        with crossing.within("train"):
            inside = crossed(varied, varied, varied)
        mx.eval(outside, inside)

        assert seen == [False, True]
        assert not mx.allclose(outside, inside).item()
    finally:
        set_sdpa_mask_composer(previous)


def test_resume_from_checkpoint_matches_fresh_run(tmp_path):
    """Stop+resume reproduces the fresh run's losses step for step."""
    fresh_dir = tmp_path / "fresh"
    resume_dir = tmp_path / "resume"

    trainer = _make_text_trainer(
        fresh_dir, _chat_dataset(), max_steps=6, save_steps=3,
    )
    trainer.train()
    fresh_hist = list(trainer._train_loss_history)
    assert len(fresh_hist) == 6, f"fresh run logged {len(fresh_hist)} losses"
    _assert_finite(fresh_hist)

    ckpt = str(fresh_dir / "checkpoint-3")
    assert os.path.isfile(os.path.join(ckpt, "adapters.safetensors"))
    assert os.path.isfile(os.path.join(ckpt, "optimizer_state.safetensors"))
    assert os.path.isfile(os.path.join(ckpt, "trainer_state.json"))
    with open(os.path.join(ckpt, "trainer_state.json")) as f:
        saved_state = json.load(f)
    assert saved_state["global_step"] == 3
    # A resume replays the data stream, so the checkpoint has to record the
    # batches it trained on or a changed one cannot be detected.
    shaping = saved_state["data_shaping"]
    assert shaping["stream_digest"]
    assert shaping["planned_visits"] == 6 * shaping["grad_accum"]

    # Fresh process state: new base model, same seeds, resume from step 3.
    resumed = _make_text_trainer(
        resume_dir, _chat_dataset(), max_steps=6, save_steps=0,
    )
    resumed.train(resume_from_checkpoint=ckpt)
    resumed_hist = list(resumed._train_loss_history)
    assert len(resumed_hist) == 6, f"resumed run logged {len(resumed_hist)} losses"
    _assert_finite(resumed_hist)

    # Restored prefix is the checkpointed history; post-resume steps must
    # track the fresh run. Same seeds + restored Adam moments mean the only
    # tolerated difference is float accumulation noise.
    for i, (a, b) in enumerate(zip(fresh_hist, resumed_hist), start=1):
        assert abs(a - b) <= 1e-5 * max(1.0, abs(a)), (
            f"step {i}: fresh={a!r} resumed={b!r}\n"
            f"fresh={fresh_hist}\nresumed={resumed_hist}"
        )


@metal_only
def test_train_on_responses_only_epoch_step_count(tmp_path):
    """Completion-only 3-epoch run executes exactly 3 epochs of steps."""
    from unsloth_zoo.mlx.trainer import train_on_responses_only

    trainer = _make_text_trainer(
        tmp_path, _chat_dataset(12), max_steps=0, num_train_epochs=3,
    )
    train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )
    trainer.train()
    hist = trainer._train_loss_history
    # 12 samples / bs 2 = 6 batches per epoch, 3 epochs = 18 steps. The
    # pre-fix epoch double-count produced 54.
    assert len(hist) == 18, f"expected 18 steps, got {len(hist)}"
    _assert_finite(hist)


@metal_only
def test_epoch_based_unlabeled_step_count(tmp_path):
    """num_train_epochs drives total steps when max_steps is disabled."""
    trainer = _make_text_trainer(
        tmp_path, _chat_dataset(12), max_steps=0, num_train_epochs=2,
    )
    trainer.train()
    hist = trainer._train_loss_history
    assert len(hist) == 12, f"expected 12 steps (6 batches x 2 epochs), got {len(hist)}"
    _assert_finite(hist)


@metal_only
def test_sgd_coupled_weight_decay_e2e(tmp_path):
    """SGD path trains with momentum and gradient-coupled weight decay."""
    trainer = _make_text_trainer(
        tmp_path, _chat_dataset(), max_steps=4,
        optim="sgd", weight_decay=0.01, learning_rate=1e-3,
    )
    trainer.train()
    hist = trainer._train_loss_history
    assert len(hist) == 4
    _assert_finite(hist)


@metal_only
def test_vlm_lora_training_e2e(tmp_path):
    """Real VLM LoRA fit: collation, label masking, CCE, save."""
    from PIL import Image

    from unsloth_zoo.mlx.loader import FastMLXModel
    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig

    colors = ["red", "green", "blue", "yellow"]
    dataset = [
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "What color is this square?"},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": f"This square is {color}."}],
                },
            ],
            "images": [Image.new("RGB", (64, 64), color)],
        }
        for color in colors
    ]

    model, processor = FastMLXModel.from_pretrained(VLM_MODEL, max_seq_length=512)
    model = FastMLXModel.get_peft_model(model, r=8, lora_alpha=16, lora_dropout=0)
    trainer = MLXTrainer(
        model=model,
        tokenizer=processor,
        train_dataset=dataset,
        args=MLXTrainingConfig(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            max_steps=4,
            warmup_steps=1,
            learning_rate=1e-4,
            logging_steps=1,
            output_dir=str(tmp_path),
            seed=3407,
            report_to="none",
        ),
    )
    assert trainer._is_vlm, "SmolVLM was not detected as a VLM"
    trainer.train()
    hist = trainer._train_loss_history
    assert len(hist) == 4
    _assert_finite(hist)
    saved = glob.glob(os.path.join(str(tmp_path), "**", "*.safetensors"), recursive=True)
    assert saved, "no adapter safetensors saved at end of VLM training"


def _color_square_dataset(colors):
    """Synthetic VLM dataset: one solid-color 64x64 square per message."""
    from PIL import Image
    return [
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "What color is this square?"},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": f"This square is {color}."}],
                },
            ],
            "images": [Image.new("RGB", (64, 64), color)],
        }
        for color in colors
    ]


def _make_vlm_trainer(tmp_path, dataset, **config_overrides):
    from unsloth_zoo.mlx.loader import FastMLXModel
    from unsloth_zoo.mlx.trainer import MLXTrainer, MLXTrainingConfig

    model, processor = FastMLXModel.from_pretrained(VLM_MODEL, max_seq_length=512)
    model = FastMLXModel.get_peft_model(model, r=8, lora_alpha=16, lora_dropout=0)
    config = dict(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        max_steps=4,
        warmup_steps=1,
        learning_rate=1e-4,
        logging_steps=1,
        output_dir=str(tmp_path),
        seed=3407,
        report_to="none",
    )
    config.update(config_overrides)
    return MLXTrainer(
        model=model,
        tokenizer=processor,
        train_dataset=dataset,
        args=MLXTrainingConfig(**config),
    )


@metal_only
def test_vlm_resume_from_checkpoint_matches_fresh_run(tmp_path):
    """VLM stop+resume reproduces the fresh run's losses step for step.

    Mirrors test_resume_from_checkpoint_matches_fresh_run for a VLM model
    so the resume code path (optimizer state restore, batch fast-forward,
    LR schedule offset) is exercised through the multimodal collator and
    image processor in addition to the text-only path.
    """
    fresh_dir = tmp_path / "fresh"
    resume_dir = tmp_path / "resume"

    colors = ["red", "green", "blue", "yellow", "purple", "orange"]
    dataset = _color_square_dataset(colors)

    trainer = _make_vlm_trainer(
        fresh_dir, dataset, max_steps=6, save_steps=3,
    )
    assert trainer._is_vlm, f"{VLM_MODEL} was not detected as a VLM"
    trainer.train()
    fresh_hist = list(trainer._train_loss_history)
    assert len(fresh_hist) == 6, f"fresh run logged {len(fresh_hist)} losses"
    _assert_finite(fresh_hist)

    ckpt_dir = fresh_dir / "checkpoint-3"
    assert (ckpt_dir / "adapters.safetensors").is_file()
    assert (ckpt_dir / "optimizer_state.safetensors").is_file()
    assert (ckpt_dir / "trainer_state.json").is_file()
    with open(ckpt_dir / "trainer_state.json") as f:
        saved_state = json.load(f)
    assert saved_state["global_step"] == 3
    ckpt = str(ckpt_dir)

    # Free the fresh trainer before loading the second 2B model (memory-tight runners).
    del trainer
    gc.collect()

    # Fresh process state: new base model, same seeds, resume from step 3.
    resumed = _make_vlm_trainer(
        resume_dir, _color_square_dataset(colors), max_steps=6, save_steps=0,
    )
    resumed.train(resume_from_checkpoint=ckpt)
    resumed_hist = list(resumed._train_loss_history)
    assert len(resumed_hist) == 6, f"resumed run logged {len(resumed_hist)} losses"
    _assert_finite(resumed_hist)

    # Restored prefix is the checkpointed history; post-resume steps must
    # track the fresh run. Same seeds + restored Adam moments mean the only
    # tolerated difference is float accumulation noise.
    for i, (a, b) in enumerate(zip(fresh_hist, resumed_hist), start=1):
        assert abs(a - b) <= 1e-5 * max(1.0, abs(a)), (
            f"step {i}: fresh={a!r} resumed={b!r}\n"
            f"fresh={fresh_hist}\nresumed={resumed_hist}"
        )
