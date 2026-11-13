"""Model adapters, per-run usage scopes, and canonical pricing.

This package is deliberately import-light: importing it (or ``api_client`` /
``usage`` / ``rates`` / ``clients``) must not pull in ``torch`` or
``transformers``.  The local inference adapter needs both, so it is resolved
lazily through PEP 562 ``__getattr__`` and only imported when its names are
actually requested.  Code paths that only call HTTP endpoints or read usage
therefore stay cheap and dependency-free.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from federated_mcts.models.api_client import (
    EARLY_STOP_FUNCTION,
    EarlyStopException,
    _create_completion,
    chatgpt,
    gpt,
)
from federated_mcts.models.clients import ClientCallable, create_clients
from federated_mcts.models.rates import (
    DEFAULT_RATE_CARD,
    LEGACY_UNKNOWN_MODEL_RATE,
    RATE_CARD_VERSION,
    ModelRate,
    RateCard,
    UnknownModelRateError,
    cost_raw_units,
)
from federated_mcts.models.usage import (
    UsageTracker,
    get_model_usage_summary,
    get_usage_tracker,
    gpt_usage,
    print_usage_summary,
    reset_usage_stats,
    usage_scope,
)
from federated_mcts.models.vllm_client import VLLMClient, create_vllm_gpt_function

if TYPE_CHECKING:
    from federated_mcts.models.local_inference import (
        DraftModel,
        UncertaintyInferenceFramework,
    )

_LAZY_NAMES = frozenset({"DraftModel", "UncertaintyInferenceFramework"})

__all__ = [
    "EARLY_STOP_FUNCTION",
    "EarlyStopException",
    "ClientCallable",
    "DraftModel",
    "ModelRate",
    "RATE_CARD_VERSION",
    "RateCard",
    "UncertaintyInferenceFramework",
    "UnknownModelRateError",
    "UsageTracker",
    "VLLMClient",
    "_create_completion",
    "chatgpt",
    "cost_raw_units",
    "create_clients",
    "create_vllm_gpt_function",
    "DEFAULT_RATE_CARD",
    "LEGACY_UNKNOWN_MODEL_RATE",
    "get_model_usage_summary",
    "get_usage_tracker",
    "gpt",
    "gpt_usage",
    "print_usage_summary",
    "reset_usage_stats",
    "usage_scope",
]


def __getattr__(name: str) -> Any:
    """Resolve torch-backed local-inference names on first use only."""
    if name in _LAZY_NAMES:
        from federated_mcts.models import local_inference

        value = getattr(local_inference, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _LAZY_NAMES)
