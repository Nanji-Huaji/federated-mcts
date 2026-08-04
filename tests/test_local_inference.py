"""Tests for ``local_inference.py`` typed seam and ``UncertaintyInferenceFramework``.

All tests use **typed concrete fakes** implementing the ``_inference_types``
Protocols — no real transformers or GPU required.
"""

from __future__ import annotations

import unittest

import torch

from federated_mcts.models._inference_types import (
    InferenceDependencies,
)
from federated_mcts.models.local_inference import UncertaintyInferenceFramework


# ── typed concrete fakes ──────────────────────────────────────────────


class FakeModelOutput:
    """Fake implementing ``ModelOutputProtocol`` — carries a ``logits`` field."""

    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits


class FakeModel:
    """Fake implementing ``ModelProtocol`` without escape hatches.

    Accepts an optional ``forward_fn`` callable that customises what
    ``__call__`` returns (default: random logits).  ``to()`` records device.
    """

    def __init__(self) -> None:
        self._device: str = "cpu"
        self.to_calls: list[str] = []
        self.call_calls: list[torch.Tensor] = []
        self.forward_fn: object | None = None

    def to(self, device: str) -> "FakeModel":
        self.to_calls.append(device)
        self._device = device
        return self

    def __call__(self, input_ids: torch.Tensor) -> FakeModelOutput:
        self.call_calls.append(input_ids.clone())
        if self.forward_fn is not None:
            fn = self.forward_fn
            if callable(fn):
                return fn(input_ids)
        batch, seq = input_ids.shape
        return FakeModelOutput(torch.randn(batch, seq, 100))


class FakeTokenizer:
    """Fake implementing ``TokenizerProtocol`` — records calls and returns stubs."""

    def __init__(self) -> None:
        self.pad_token: str | None = None
        self.eos_token: str | None = "<eos>"
        self.eos_token_id: int | None = 0
        self.encode_calls: list[tuple[str, str]] = []
        self.decode_calls: list[tuple[torch.Tensor, bool]] = []

    def encode(
        self, text: str, return_tensors: str = "pt"
    ) -> torch.Tensor:
        self.encode_calls.append((text, return_tensors))
        return torch.tensor([[1, 2, 3]])

    def decode(
        self,
        token_ids: torch.Tensor,
        skip_special_tokens: bool = True,
    ) -> str:
        self.decode_calls.append((token_ids.clone(), skip_special_tokens))
        return "decoded text"


# ── typed concrete loader factories ──────────────────────────────────


class _CapturingTokenizerLoader:
    """``TokenizerLoader`` that records the model-path and returns a fake."""

    def __init__(self) -> None:
        self.captured_path: str | None = None
        self.times_called: int = 0

    def __call__(self, model_path: str) -> FakeTokenizer:
        self.captured_path = model_path
        self.times_called += 1
        return FakeTokenizer()


class _CapturingModelLoader:
    """``ModelLoader`` that records the model-path and returns a fake."""

    def __init__(self) -> None:
        self.captured_path: str | None = None
        self.times_called: int = 0

    def __call__(self, model_path: str) -> FakeModel:
        self.captured_path = model_path
        self.times_called += 1
        return FakeModel()


class _FixedModelLoader:
    """``ModelLoader`` always returning the *same* ``FakeModel`` instance."""

    def __init__(self, model: FakeModel) -> None:
        self._model = model
        self.captured_path: str | None = None

    def __call__(self, model_path: str) -> FakeModel:
        self.captured_path = model_path
        return self._model


class _FixedTokenizerLoader:
    """``TokenizerLoader`` always returning the *same* ``FakeTokenizer`` instance."""

    def __init__(self, tokenizer: FakeTokenizer) -> None:
        self._tokenizer = tokenizer
        self.captured_path: str | None = None

    def __call__(self, model_path: str) -> FakeTokenizer:
        self.captured_path = model_path
        return self._tokenizer


# ── helpers ───────────────────────────────────────────────────────────


def _fake_uncertainty_fn(logits: torch.Tensor, current_token: int) -> float:
    """Deterministic stub: always returns 0.5."""
    _ = logits
    _ = current_token
    return 0.5


def _always_eos_forward(input_ids: torch.Tensor) -> FakeModelOutput:
    """A forwarded override that always predicts token 0 (eos)."""
    batch, seq = input_ids.shape
    logits = torch.zeros(batch, seq, 100)
    logits[:, :, 0] = 100.0
    return FakeModelOutput(logits)


# ── tests ─────────────────────────────────────────────────────────────


class TestUncertaintyInferenceFramework(unittest.TestCase):
    """Unit tests for ``UncertaintyInferenceFramework`` with typed fakes."""

    # ── model-name contract ───────────────────────────────────────

    def test_model_name_is_passed_unchanged_to_both_loaders(self) -> None:
        tok_loader = _CapturingTokenizerLoader()
        mod_loader = _CapturingModelLoader()
        deps = InferenceDependencies(
            tokenizer_loader=tok_loader, model_loader=mod_loader
        )

        path = "openai-community/gpt2"
        UncertaintyInferenceFramework(path, device="cpu", deps=deps)

        self.assertEqual(tok_loader.captured_path, path)
        self.assertEqual(mod_loader.captured_path, path)
        self.assertEqual(tok_loader.times_called, 1)
        self.assertEqual(mod_loader.times_called, 1)

    # ── .to(device) contract ──────────────────────────────────────

    def test_device_is_stored_as_attribute(self) -> None:
        framework = UncertaintyInferenceFramework(
            "m",
            device="cuda:1",
            deps=InferenceDependencies(
                tokenizer_loader=_CapturingTokenizerLoader(),
                model_loader=_CapturingModelLoader(),
            ),
        )
        self.assertEqual(framework.device, "cuda:1")

    def test_model_to_device_retention_via_shared_fake(self) -> None:
        shared_model = FakeModel()
        deps = InferenceDependencies(
            tokenizer_loader=_CapturingTokenizerLoader(),
            model_loader=_FixedModelLoader(shared_model),
        )
        UncertaintyInferenceFramework("m", device="cuda:3", deps=deps)

        self.assertEqual(shared_model.to_calls, ["cuda:3"])

    # ── tokenizer pad_token setup ─────────────────────────────────

    def test_pad_token_set_when_none(self) -> None:
        tok = FakeTokenizer()
        tok.pad_token = None
        tok.eos_token = "<eos>"

        deps = InferenceDependencies(
            tokenizer_loader=_FixedTokenizerLoader(tok),
            model_loader=_CapturingModelLoader(),
        )
        UncertaintyInferenceFramework("m", device="cpu", deps=deps)

        self.assertEqual(tok.pad_token, "<eos>")

    def test_pad_token_unchanged_when_already_set(self) -> None:
        tok = FakeTokenizer()
        tok.pad_token = "[PAD]"

        deps = InferenceDependencies(
            tokenizer_loader=_FixedTokenizerLoader(tok),
            model_loader=_CapturingModelLoader(),
        )
        UncertaintyInferenceFramework("m", device="cpu", deps=deps)

        self.assertEqual(tok.pad_token, "[PAD]")

    # ── return contract ───────────────────────────────────────────

    def test_generate_return_shape(self) -> None:
        shared_model = FakeModel()
        deps = InferenceDependencies(
            tokenizer_loader=_CapturingTokenizerLoader(),
            model_loader=_FixedModelLoader(shared_model),
        )
        framework = UncertaintyInferenceFramework(
            "m", device="cpu", deps=deps
        )

        result = framework.generate_with_uncertainty(
            "hello",
            _fake_uncertainty_fn,
            max_length=8,
            do_sample=False,
        )

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)
        generated_text, avg_unc, per_token = result

        self.assertIsInstance(generated_text, str)
        self.assertIsInstance(avg_unc, float)
        self.assertIsInstance(per_token, list)
        self.assertEqual(avg_unc, 0.5)

    def test_generate_stops_at_eos(self) -> None:
        shared_model = FakeModel()
        shared_model.forward_fn = _always_eos_forward

        tok = FakeTokenizer()
        tok.eos_token_id = 0

        deps = InferenceDependencies(
            tokenizer_loader=_FixedTokenizerLoader(tok),
            model_loader=_FixedModelLoader(shared_model),
        )
        framework = UncertaintyInferenceFramework(
            "m", device="cpu", deps=deps
        )

        result = framework.generate_with_uncertainty(
            "test",
            _fake_uncertainty_fn,
            max_length=100,
            do_sample=False,
        )

        self.assertEqual(len(result[2]), 1)

    def test_batch_generate_collects_all_results(self) -> None:
        shared_model = FakeModel()
        deps = InferenceDependencies(
            tokenizer_loader=_CapturingTokenizerLoader(),
            model_loader=_FixedModelLoader(shared_model),
        )
        framework = UncertaintyInferenceFramework(
            "m", device="cpu", deps=deps
        )

        prompts = ["a", "b", "c"]
        results = framework.batch_generate_with_uncertainty(
            prompts, _fake_uncertainty_fn, max_length=8, do_sample=False
        )

        self.assertEqual(len(results), 3)
        for r in results:
            self.assertIsInstance(r, tuple)
            self.assertEqual(len(r), 3)

    # ── default deps (integration-light smoke) ────────────────────

    def test_default_deps_do_not_crash(self) -> None:
        try:
            framework = UncertaintyInferenceFramework(
                "hf-internal-testing/tiny-random-gpt2",
                device="cpu",
            )
        except Exception:
            self.skipTest(
                "network or model unavailable — skipping default deps smoke test"
            )

        self.assertIsNotNone(framework.tokenizer)
        self.assertIsNotNone(framework.model)
