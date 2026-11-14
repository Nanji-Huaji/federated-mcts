"""Exact remaining-distance labels for Blocksworld DQN collection."""

from __future__ import annotations

from federated_mcts.tasks.blocksworld_engine import parse_x, replay_state
from federated_mcts.tasks.blocksworld_oracle import SOLVED, shortest_plan


def min_remaining_distance(x: str, trajectories: list[str]) -> int | None:
    """Return the shortest finite remaining distance among valid trajectories."""
    parsed = parse_x(x)
    distances: list[int] = []
    for trajectory in trajectories:
        state = replay_state(parsed, trajectory)
        if state is None:
            continue
        result = shortest_plan({**parsed, "init": state})
        if result.status == SOLVED:
            distances.append(len(result.plan))
    return min(distances) if distances else None
