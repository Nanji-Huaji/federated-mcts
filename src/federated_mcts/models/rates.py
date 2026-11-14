"""Single versioned rate card and exact raw-unit billing.

Raw units in this project are USD-denominated model prices with no currency
conversion: one raw unit equals one dollar.  Prices are published per *one
million tokens*; billing therefore converts a token count to raw units with an
exact :class:`~decimal.Decimal` computation instead of binary floats, so two
callers that price the same usage always obtain the same decimal string.

Two policies exist deliberately:

``strict=True`` (the default)
    Billing.  An unknown model is an error: no price is invented.

``strict=False``
    Legacy display.  ``UsageTracker`` historically showed unknown models at the
    old 1.0/2.0 per-million fallback so summaries printed instead of raising.
    That compatibility is explicit here and must not be used to charge a budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from types import MappingProxyType
from typing import Mapping

RAW_UNIT_DENOMINATOR = Decimal(1_000_000)
"""Tokens per published rate unit (prices are quoted per 1M tokens)."""

RATE_CARD_VERSION = "2026-11-12.1"
"""Version of the canonical rate table below.

Bump this whenever a price changes.  Callers may persist the version with
priced artifacts so a cost can be traced back to the table that produced it.
"""


@dataclass(frozen=True, slots=True)
class ModelRate:
    """Published price for one model, in raw units per 1M tokens."""

    prompt: Decimal
    completion: Decimal


def _rate(prompt: str, completion: str) -> ModelRate:
    return ModelRate(prompt=Decimal(prompt), completion=Decimal(completion))


# The legacy fallback that ``UsageTracker`` always used for unknown models.
LEGACY_UNKNOWN_MODEL_RATE = ModelRate(
    prompt=Decimal("1.0"), completion=Decimal("2.0")
)
"""Display-only price for unknown models.  Never valid for strict billing."""


class UnknownModelRateError(ValueError):
    """Raised when strict billing is asked to price an unlisted model.

    Subclasses :class:`ValueError` so callers that historically failed closed
    on unknown billed models with ``ValueError`` keep working unchanged.
    """

    def __init__(self, model: str, rate_card_version: str) -> None:
        self.model = model
        self.rate_card_version = rate_card_version
        super().__init__(
            f"Unknown billed model {model!r}: no rate in RateCard "
            f"{rate_card_version!r}; add it to models/rates.py before charging"
        )


@dataclass(frozen=True, slots=True)
class RateCard:
    """Immutable versioned price table plus the pure billing operation.

    The card is data; :meth:`cost_raw_units` is the only computation callers
    need.  Keeping both behind one object means a run can record exactly which
    table priced it and tests can substitute a tiny card without touching the
    process global.
    """

    version: str
    rates: Mapping[str, ModelRate]
    unknown_model_rate: ModelRate | None = LEGACY_UNKNOWN_MODEL_RATE

    def __post_init__(self) -> None:
        if not isinstance(self.version, str) or not self.version:
            raise ValueError("RateCard.version must be a non-empty string")
        # Defensive copy: a mutable dict handed in later must not be able to
        # change billing underneath a running experiment.
        if not isinstance(self.rates, MappingProxyType):
            object.__setattr__(self, "rates", MappingProxyType(dict(self.rates)))

    def rate_for(self, model: str, *, strict: bool = True) -> ModelRate:
        """Return the published rate for *model*.

        Raises :class:`UnknownModelRateError` for unknown models when
        *strict* is true.  With ``strict=False`` the explicit legacy display
        fallback is returned instead.
        """
        rate = self.rates.get(model)
        if rate is not None:
            return rate
        if strict or self.unknown_model_rate is None:
            raise UnknownModelRateError(model, self.version)
        return self.unknown_model_rate

    def cost_raw_units(
        self,
        model: str,
        completion_tokens: int,
        prompt_tokens: int,
        *,
        strict: bool = True,
    ) -> Decimal:
        """Price *completion_tokens* + *prompt_tokens* for *model*.

        Returns exact raw units (USD-equivalent) as :class:`Decimal`.
        Token counts must be non-negative integers; booleans, floats, and
        negative values are rejected so a malformed usage record cannot be
        silently billed as zero.
        """
        completion = _token_count("completion_tokens", completion_tokens)
        prompt = _token_count("prompt_tokens", prompt_tokens)
        rate = self.rate_for(model, strict=strict)
        return (
            Decimal(completion) * rate.completion
            + Decimal(prompt) * rate.prompt
        ) / RAW_UNIT_DENOMINATOR


def _token_count(name: str, value: object) -> int:
    """Validate and normalise one token counter to a non-negative ``int``."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer token count, got bool")
    if not isinstance(value, int):
        raise TypeError(
            f"{name} must be an integer token count, got {type(value).__name__}"
        )
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


# ── Canonical table (the single source of truth) ───────────────────────
# Keep this in sync with any published provider price change and bump
# RATE_CARD_VERSION in the same edit.
DEFAULT_RATE_CARD = RateCard(
    version=RATE_CARD_VERSION,
    rates={
        "gpt-4o": _rate("2.5", "10.0"),
        "gpt-4o-2024-08-06": _rate("2.5", "10.0"),
        "gpt-4": _rate("15.0", "30.0"),
        "gpt-3.5-turbo": _rate("0.75", "1.5"),
        "meta-llama-3.1-8b-instruct@q4_k_m": _rate("0.0", "0.0"),
        "phi-3-medium-4k-instruct": _rate("0.0", "0.0"),
        "deepseek-v4-flash": _rate("0.14", "0.28"),
        "deepseek-v4-pro": _rate("0.435", "0.87"),
        "qwen3.7-plus": _rate("0.4", "1.6"),
        "qwen3.6-plus": _rate("0.4", "1.6"),
        # Locally served vLLM models are zero-cost by contract.
        "Qwen/Qwen2.5-7B-Instruct": _rate("0.0", "0.0"),
        "Qwen/Qwen2.5-32B-Instruct": _rate("0.0", "0.0"),
    },
)


def cost_raw_units(
    model: str,
    completion_tokens: int,
    prompt_tokens: int,
    *,
    strict: bool = True,
    rate_card: RateCard = DEFAULT_RATE_CARD,
) -> Decimal:
    """Pure functional form of :meth:`RateCard.cost_raw_units`."""
    return rate_card.cost_raw_units(
        model, completion_tokens, prompt_tokens, strict=strict
    )


__all__ = [
    "DEFAULT_RATE_CARD",
    "LEGACY_UNKNOWN_MODEL_RATE",
    "ModelRate",
    "RATE_CARD_VERSION",
    "RAW_UNIT_DENOMINATOR",
    "RateCard",
    "UnknownModelRateError",
    "cost_raw_units",
]
