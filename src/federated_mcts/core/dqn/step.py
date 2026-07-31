"""Per-step outcome contract of the DQN search session."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DqnStepOutcome:
    candidates: list[str]
    values: list[float]
    selected: list[str]
    stopped: bool
    transitions: list[dict]
    search_metrics: dict
    eval_seconds: float
