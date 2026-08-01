"""Reward shaping for the Budget-Aware DQN controller.

The correctness reward is sparse: only the final transition of an episode
carries the exact one-hot success signal.  Earlier transitions keep a zero
correctness base.  Token and latency penalties are monotonic non-decreasing
in consumption and bounded to [0, 1].
"""

from __future__ import annotations


def correctness_reward(is_success: bool) -> float:
    return 1.0 if is_success else 0.0


def token_penalty(tokens: int, max_tokens: int = 1000) -> float:
    if max_tokens <= 0:
        return 0.0
    return min(1.0, max(0.0, tokens / max_tokens))


def latency_penalty(seconds: float, budget_seconds: float = 10.0) -> float:
    if budget_seconds <= 0:
        return 0.0
    return min(1.0, max(0.0, seconds / budget_seconds))


def rewards_for_episode(
    transitions,
    terminal_success: bool,
    max_tokens: int = 1000,
    budget_seconds: float = 10.0,
) -> list[float]:
    rewards: list[float] = []
    last_index = len(transitions) - 1
    for index, transition in enumerate(transitions):
        base = correctness_reward(terminal_success) if index == last_index else 0.0
        base -= token_penalty(transition["tokens"], max_tokens)
        base -= latency_penalty(transition["latency"], budget_seconds)
        rewards.append(base)
    return rewards
