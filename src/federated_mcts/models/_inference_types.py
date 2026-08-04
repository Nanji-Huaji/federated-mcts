"""Typed seam for uncertainty inference — Protocols + injectable dependency container.

This module defines the structural interfaces (Protocols) for tokenizers and
causal language models consumed by UncertaintyInferenceFramework, plus a frozen
dataclass that groups injectable loader factories.  Every concrete adapter
(delegating to AutoTokenizer / AutoModelForCausalLM) lives here as a private
implementation detail.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase, PreTrainedModel


# ---------------------------------------------------------------------------
# Structural Protocols (the seam)
# ---------------------------------------------------------------------------


class TokenizerProtocol(Protocol):
    """Structural interface a tokenizer must satisfy for uncertainty inference."""

    def encode(self, text: str, return_tensors: str = ...) -> torch.Tensor: ...

    def decode(
        self, token_ids: torch.Tensor, skip_special_tokens: bool = ...
    ) -> str: ...

    pad_token: str | None
    eos_token: str | None
    eos_token_id: int | None


class ModelOutputProtocol(Protocol):
    """Structural interface matching a causal-LM forward pass output."""

    logits: torch.Tensor


class ModelProtocol(Protocol):
    """Structural interface a causal-LM must satisfy for uncertainty inference."""

    def to(self, device: str) -> ModelProtocol: ...

    def __call__(self, input_ids: torch.Tensor) -> ModelOutputProtocol: ...


# ---------------------------------------------------------------------------
# Loader factories (callable protocols)
# ---------------------------------------------------------------------------


class TokenizerLoader(Protocol):
    """Factory: model_path -> TokenizerProtocol."""

    def __call__(self, model_path: str) -> TokenizerProtocol: ...


class ModelLoader(Protocol):
    """Factory: model_path -> ModelProtocol."""

    def __call__(self, model_path: str) -> ModelProtocol: ...


# ---------------------------------------------------------------------------
# Injectable dependency container (frozen, avoid parameter bloat)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InferenceDependencies:
    """Grouped loader factories injectable into UncertaintyInferenceFramework."""

    tokenizer_loader: TokenizerLoader
    model_loader: ModelLoader


# ---------------------------------------------------------------------------
# Private adapter wrappers — produced by the default loaders and checked to
# structurally satisfy the Protocols at import time, not at each call.
# ---------------------------------------------------------------------------


class _TokenizerAdapter:
    """Wraps a HuggingFace tokenizer to satisfy ``TokenizerProtocol``.

    The HF tokenizer's ``encode`` returns ``list[int] | torch.Tensor``
    depending on ``return_tensors``; this adapter narrows to ``torch.Tensor``
    (the only shape consumed downstream).
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self._tok: PreTrainedTokenizerBase = tokenizer

    @property
    def pad_token(self) -> str | None:
        return self._tok.pad_token

    @pad_token.setter
    def pad_token(self, value: str | None) -> None:
        self._tok.pad_token = value

    @property
    def eos_token(self) -> str | None:
        return self._tok.eos_token

    @eos_token.setter
    def eos_token(self, value: str | None) -> None:
        self._tok.eos_token = value

    @property
    def eos_token_id(self) -> int | None:
        return self._tok.eos_token_id

    @eos_token_id.setter
    def eos_token_id(self, value: int | None) -> None:
        self._tok.eos_token_id = value

    def encode(self, text: str, return_tensors: str = "pt") -> torch.Tensor:
        result = self._tok.encode(text, return_tensors=return_tensors)
        assert isinstance(result, torch.Tensor), (
            f"expected torch.Tensor from tokenizer.encode, got {type(result)}"
        )
        return result

    def decode(
        self,
        token_ids: torch.Tensor,
        skip_special_tokens: bool = True,
    ) -> str:
        result = self._tok.decode(token_ids, skip_special_tokens=skip_special_tokens)
        assert isinstance(result, str), (
            f"expected str from tokenizer.decode, got {type(result)}"
        )
        return result


class _ModelAdapter:
    """Wraps a HuggingFace causal-LM to satisfy ``ModelProtocol``."""

    def __init__(self, model: PreTrainedModel) -> None:
        self._model: PreTrainedModel = model

    def to(self, device: str) -> ModelProtocol:
        """Move the underlying model to *device* and return self."""
        nn.Module.to(self._model, device)  # bypass HF _Wrapped decorator
        return self

    def __call__(self, input_ids: torch.Tensor) -> ModelOutputProtocol:
        return self._model(input_ids)


# ---------------------------------------------------------------------------
# Default concrete loaders (no Any, cast, or escape hatches)
# ---------------------------------------------------------------------------


def _default_tokenizer_loader(model_path: str) -> TokenizerProtocol:
    """Load a real HF tokenizer, return it through the typed adapter."""
    tok = AutoTokenizer.from_pretrained(model_path)
    return _TokenizerAdapter(tok)


def _default_model_loader(model_path: str) -> ModelProtocol:
    """Load a real HF causal-LM, return it through the typed adapter."""
    model = AutoModelForCausalLM.from_pretrained(model_path)
    return _ModelAdapter(model)
