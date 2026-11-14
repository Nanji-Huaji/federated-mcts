"""Reward component assembly for recorded DQN transitions."""

from __future__ import annotations

from typing import TypedDict

from federated_mcts.core.dqn.rewards import (
    correctness_reward,
    latency_penalty,
    oracle_distance_reward,
    token_penalty,
)


class RewardComponents(TypedDict):
    correctness: float
    distance: float
    token_penalty: float
    latency_penalty: float
    total: float


def build_reward_components(
    *,
    success: bool,
    done: bool,
    distance_before: int | None,
    distance_after: int | None,
    distance_scale: float,
    tokens: float,
    latency: float,
    max_tokens: int,
    budget_seconds: float,
) -> RewardComponents:
    """Compute the exact transition reward and its independently logged terms."""
    correctness = correctness_reward(success) if done else 0.0
    distance = oracle_distance_reward(distance_before, distance_after, distance_scale)
    token_cost = token_penalty(tokens, max_tokens)
    latency_cost = latency_penalty(latency, budget_seconds)
    return RewardComponents(
        correctness=correctness,
        distance=distance,
        token_penalty=token_cost,
        latency_penalty=latency_cost,
        total=correctness + distance - token_cost - latency_cost,
    )
