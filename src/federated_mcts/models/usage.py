"""Token usage and raw-unit cost tracking for model API calls.

Pricing lives in :mod:`federated_mcts.models.rates`; this module only
accumulates counters and delegates the arithmetic to the canonical
:data:`~federated_mcts.models.rates.DEFAULT_RATE_CARD`.

The tracker used by :func:`get_usage_tracker` is resolved through a
:class:`contextvars.ContextVar`.  ``usage_scope()`` installs a tracker for one
logical run, so two runs in the same process can execute in parallel without
sharing counters.  Outside every scope the historical module-level singleton is
returned, preserving the old process-wide behaviour for scripts and tests.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from decimal import Decimal
from typing import Dict, Iterator

from federated_mcts.models.rates import DEFAULT_RATE_CARD, RateCard


class UsageTracker:
    """Thread-safe tracker of token usage and raw-unit cost per model."""

    def __init__(self, rate_card: RateCard = DEFAULT_RATE_CARD):
        self._lock = threading.Lock()
        self._rate_card = rate_card
        self._model_usage: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"completion_tokens": 0, "prompt_tokens": 0, "total_calls": 0}
        )

    def record(self, model: str, completion_tokens: int, prompt_tokens: int):
        with self._lock:
            self._model_usage[model]["completion_tokens"] += completion_tokens
            self._model_usage[model]["prompt_tokens"] += prompt_tokens
            self._model_usage[model]["total_calls"] += 1

    def get_summary(self) -> Dict:
        with self._lock:
            summary = {}
            for model, usage in self._model_usage.items():
                total_tokens = usage["completion_tokens"] + usage["prompt_tokens"]
                cost = float(
                    self._cost_raw_units(
                        model, usage["completion_tokens"], usage["prompt_tokens"]
                    )
                )
                summary[model] = {
                    "completion_tokens": usage["completion_tokens"],
                    "prompt_tokens": usage["prompt_tokens"],
                    "total_tokens": total_tokens,
                    "total_calls": usage["total_calls"],
                    "cost": cost,
                }
            return summary

    def get_total_tokens(self) -> int:
        with self._lock:
            return sum(
                u["completion_tokens"] + u["prompt_tokens"]
                for u in self._model_usage.values()
            )

    def get_total_cost(self) -> float:
        return float(self.get_total_cost_raw_units())

    def get_total_cost_raw_units(self) -> Decimal:
        """Exact aggregate spend as Decimal raw units.

        Companion to the legacy float :meth:`get_total_cost`; budget and
        evidence code must use this value so decimal sums do not drift.
        """
        with self._lock:
            total = Decimal("0")
            for model, usage in self._model_usage.items():
                total += self._cost_raw_units(
                    model, usage["completion_tokens"], usage["prompt_tokens"]
                )
            return total

    def get_model_list(self):
        with self._lock:
            return list(self._model_usage.keys())

    def reset(self):
        with self._lock:
            self._model_usage.clear()

    def print_summary(self):
        print()
        print("=" * 60)
        print("TOKEN USAGE SUMMARY")
        print("=" * 60)
        summary = self.get_summary()
        total_cost = 0.0
        for model, stats in summary.items():
            print(f"\nModel: {model}")
            print(f"  Calls: {stats['total_calls']}")
            print(f"  Prompt tokens: {stats['prompt_tokens']:,}")
            print(f"  Completion tokens: {stats['completion_tokens']:,}")
            print(f"  Total tokens: {stats['total_tokens']:,}")
            print(f"  Cost: USD {stats['cost']:.4f}")
            total_cost += stats["cost"]
        print(f"\nTotal cost across all models: USD {total_cost:.4f}")
        print("=" * 60)

    def _cost_raw_units(
        self, model: str, completion_tokens: int, prompt_tokens: int
    ) -> Decimal:
        """Price one model's counters with legacy-display fallback semantics."""
        return self._rate_card.cost_raw_units(
            model,
            completion_tokens,
            prompt_tokens,
            strict=False,
        )

    @staticmethod
    def _calculate_cost(
        model: str, completion_tokens: int, prompt_tokens: int
    ) -> float:
        """Legacy float cost view; exact arithmetic lives in ``rates``."""
        return float(
            DEFAULT_RATE_CARD.cost_raw_units(
                model,
                completion_tokens,
                prompt_tokens,
                strict=False,
            )
        )


# ── Per-run scope ──────────────────────────────────────────────────────


_current_usage_tracker: ContextVar[UsageTracker | None] = ContextVar(
    "federated_mcts_usage_tracker", default=None
)


@contextmanager
def usage_scope(tracker: UsageTracker | None = None) -> Iterator[UsageTracker]:
    """Bind a usage tracker to the current context for one logical run.

    With no argument a fresh :class:`UsageTracker` is created and yielded, so
    the caller can read that run's counters afterwards.  Passing a tracker
    explicitly lets an outer runner own the object (for example to aggregate
    across tasks).  On exit the previous binding is restored, including after
    an exception.

    ``ContextVar`` isolation is per thread/task.  Work handed to a thread pool
    must propagate the context explicitly (``contextvars.copy_context()``) if
    it should record into the scoped tracker; this module deliberately does not
    mutate thread-global state to fake that propagation.
    """
    scoped = tracker if tracker is not None else UsageTracker()
    token = _current_usage_tracker.set(scoped)
    try:
        yield scoped
    finally:
        _current_usage_tracker.reset(token)


# Global singleton for backward compatibility
_global_usage_tracker = UsageTracker()


# Legacy-style accessor functions
def get_usage_tracker() -> UsageTracker:
    """Return the scoped tracker when one is active, else the global one."""
    scoped = _current_usage_tracker.get()
    return scoped if scoped is not None else _global_usage_tracker


def get_model_usage_summary():
    return get_usage_tracker().get_summary()


def reset_usage_stats():
    get_usage_tracker().reset()


def print_usage_summary():
    get_usage_tracker().print_summary()


def gpt_usage(backend="gpt-4o"):
    """Legacy compatibility — returns aggregate stats."""
    summary = get_usage_tracker().get_summary()
    llm_completion = 0
    llm_prompt = 0
    slm_completion = 0
    slm_prompt = 0
    total_completion = 0
    total_prompt = 0
    for model, stats in summary.items():
        total_completion += stats["completion_tokens"]
        total_prompt += stats["prompt_tokens"]
        if "gpt" in model.lower() or model == backend:
            llm_completion += stats["completion_tokens"]
            llm_prompt += stats["prompt_tokens"]
        else:
            slm_completion += stats["completion_tokens"]
            slm_prompt += stats["prompt_tokens"]
    cost = float(
        DEFAULT_RATE_CARD.cost_raw_units(
            "gpt-4o", llm_completion, llm_prompt, strict=False
        )
    )
    return {
        "llm_completion_tokens": llm_completion,
        "llm_prompt_tokens": llm_prompt,
        "slm_completion_tokens": slm_completion,
        "slm_prompt_tokens": slm_prompt,
        "total_completion_tokens": total_completion,
        "total_prompt_tokens": total_prompt,
        "cost": cost,
        "model_usage": summary,
    }
