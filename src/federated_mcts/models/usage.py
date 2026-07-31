"""Token usage and cost tracking for model API calls."""

import threading
from collections import defaultdict
from typing import Dict


class UsageTracker:
    """Thread-safe singleton tracking token usage and cost per model."""

    def __init__(self):
        self._lock = threading.Lock()
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
                cost = self._calculate_cost(model, usage["completion_tokens"], usage["prompt_tokens"])
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
            return sum(u["completion_tokens"] + u["prompt_tokens"] for u in self._model_usage.values())

    def get_total_cost(self) -> float:
        with self._lock:
            return sum(
                self._calculate_cost(m, u["completion_tokens"], u["prompt_tokens"])
                for m, u in self._model_usage.items()
            )

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
            print(f"  Cost: \u00a5{stats['cost']:.4f}")
            total_cost += stats["cost"]
        print(f"\nTotal cost across all models: \u00a5{total_cost:.4f}")
        print("=" * 60)

    @staticmethod
    def _calculate_cost(model: str, completion_tokens: int, prompt_tokens: int) -> float:
        model_prices = {
            "gpt-4o": {"prompt": 2.5, "completion": 10.0},
            "gpt-4": {"prompt": 15.0, "completion": 30.0},
            "gpt-3.5-turbo": {"prompt": 0.75, "completion": 1.5},
            "meta-llama-3.1-8b-instruct@q4_k_m": {"prompt": 0.0, "completion": 0.0},
            "phi-3-medium-4k-instruct": {"prompt": 0.0, "completion": 0.0},
            "deepseek-v4-flash": {"prompt": 0.14, "completion": 0.28},
            "deepseek-v4-pro": {"prompt": 0.435, "completion": 0.87},
        }
        price = model_prices.get(model, {"prompt": 1.0, "completion": 2.0})
        return completion_tokens * price["completion"] / 1_000_000 + prompt_tokens * price["prompt"] / 1_000_000


# Global singleton for backward compatibility
_global_usage_tracker = UsageTracker()

# Legacy-style accessor functions
def get_usage_tracker() -> UsageTracker:
    return _global_usage_tracker


def get_model_usage_summary():
    return _global_usage_tracker.get_summary()


def reset_usage_stats():
    _global_usage_tracker.reset()


def print_usage_summary():
    _global_usage_tracker.print_summary()


def gpt_usage(backend="gpt-4o"):
    """Legacy compatibility — returns aggregate stats."""
    summary = _global_usage_tracker.get_summary()
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
    cost = llm_completion * 10 / 1_000_000 + llm_prompt * 2.5 / 1_000_000
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
