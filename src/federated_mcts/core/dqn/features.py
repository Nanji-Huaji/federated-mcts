"""Pre-decision state features for the Budget-Aware DQN controller.

The state is computed BEFORE the current-step action is taken, so it may use
only candidate/dedup structure, previous-step value statistics and consumed
budgets — never the current step's ranking values.  The vector is finite,
normalized into [0, 1] and has 12 dims (13 when the previous joint-rank
flag is included).
"""

from __future__ import annotations

import numpy as np

MAX_CANDIDATE_COUNT: int = 32
_VALUE_HIGH_THRESHOLD: float = 0.9


class StateVector(np.ndarray):
    """float32 1-D state vector.  A tuple of int indices is treated as a
    fancy index (e.g. state[(0, 1, 2)]) so feature groups can be selected
    with a tuple constant."""

    def __new__(cls, values):
        return np.asarray(values, dtype=np.float32).view(cls)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            key = list(key)
        return super().__getitem__(key)


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))


def _previous_stats(values) -> tuple[float, float, float, float, float, float]:
    """Returns (mean, max, min, std, high-fraction, range) of previous
    values, all zero when no previous values exist."""
    if not values:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    maximum = float(array.max())
    minimum = float(array.min())
    std = float(array.std())
    high = float(np.count_nonzero(array >= _VALUE_HIGH_THRESHOLD) / array.size)
    return mean, maximum, minimum, std, high, maximum - minimum


def extract_state_features(
    *,
    candidates,
    states,
    previous_values,
    step: int,
    total_steps: int,
    tokens_consumed: float,
    token_budget: float,
    latency_consumed: float,
    latency_budget: float,
    previous_joint_rank: bool | None = None,
) -> np.ndarray:
    raw_count = len(candidates)
    unique_count = len(states)
    mean, maximum, minimum, std, high, span = _previous_stats(previous_values)
    features = [
        _clamp01(raw_count / MAX_CANDIDATE_COUNT),
        _clamp01(unique_count / MAX_CANDIDATE_COUNT),
        1.0 - (unique_count / raw_count) if raw_count else 0.0,
        mean,
        maximum,
        minimum,
        std,
        high,
        _clamp01(step / total_steps) if total_steps else 0.0,
        _clamp01(tokens_consumed / token_budget) if token_budget > 0 else 0.0,
        _clamp01(latency_consumed / latency_budget) if latency_budget > 0 else 0.0,
        span,
    ]
    if previous_joint_rank is not None:
        features.append(1.0 if previous_joint_rank else 0.0)
    return StateVector(features)
