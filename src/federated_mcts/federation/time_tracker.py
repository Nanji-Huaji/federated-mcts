"""Time/latency tracking for federated MCTS steps."""

from collections.abc import Iterable
from typing import TypedDict


class ClientTiming(TypedDict):
    generation: float
    evaluation: float


class TimeTracker:
    """Tracks generation and evaluation latency per client and globally."""

    def __init__(self, client_names: Iterable[str] | None = None) -> None:
        self.generation: float = 0.0
        self.evaluation: float = 0.0
        self.client_stats: dict[str, ClientTiming] = {}
        if client_names:
            for name in client_names:
                self.client_stats[name] = {"generation": 0.0, "evaluation": 0.0}

    def ensure_client(self, client_name: str) -> None:
        if client_name not in self.client_stats:
            self.client_stats[client_name] = {"generation": 0.0, "evaluation": 0.0}

    def record_generation(self, client_name: str, duration: float) -> None:
        self.ensure_client(client_name)
        self.client_stats[client_name]["generation"] += duration

    def record_evaluation(self, client_name: str, duration: float) -> None:
        self.ensure_client(client_name)
        self.client_stats[client_name]["evaluation"] += duration

    def accumulate_step(self, step_client_times: dict[str, ClientTiming]) -> None:
        """Add step-level max times to global totals."""
        max_gen = max(
            (stats["generation"] for stats in step_client_times.values()),
            default=0.0
        )
        max_eval = max(
            (stats["evaluation"] for stats in step_client_times.values()),
            default=0.0
        )
        self.generation += max_gen
        self.evaluation += max_eval
        for client_name, stats in step_client_times.items():
            self.ensure_client(client_name)
            self.client_stats[client_name]["generation"] += stats["generation"]
            self.client_stats[client_name]["evaluation"] += stats["evaluation"]

    def reset(self) -> None:
        self.generation = 0.0
        self.evaluation = 0.0
        self.client_stats.clear()

    @property
    def latency_dict(self) -> dict[str, float]:
        return {"generation": self.generation, "evaluation": self.evaluation}
