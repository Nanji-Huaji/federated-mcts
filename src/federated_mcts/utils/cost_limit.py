"""Estimated API-cost limits for bounded experiment runs."""

from __future__ import annotations


def total_cost_usd(summary: dict[str, dict[str, float]]) -> float:
    """Sum the project-estimated per-model costs in a usage summary."""
    return sum(float(stats.get("cost", 0.0)) for stats in summary.values())


def cost_exceeded(summary: dict[str, dict[str, float]], limit: float | None) -> bool:
    """Return whether the estimated cumulative cost reached a configured cap."""
    return limit is not None and total_cost_usd(summary) >= limit
