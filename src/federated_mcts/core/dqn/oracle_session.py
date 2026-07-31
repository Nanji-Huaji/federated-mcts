"""Blocksworld-only distance hooks for DQN session lifecycle."""

from __future__ import annotations

from federated_mcts.core.dqn.oracle_distance import min_remaining_distance


def oracle_distance_enabled(task: object, enabled: bool) -> bool:
    """Return whether a session may attach exact Blocksworld distance rewards."""
    if not enabled:
        return False
    from federated_mcts.tasks.blocksworld import BlocksworldTask

    return isinstance(task, BlocksworldTask)


def initial_distance(x: str) -> int | None:
    """Return the initial exact remaining distance for a Blocksworld input."""
    return min_remaining_distance(x, [""])


def selected_distance(x: str, selected: list[str], stopped: bool) -> int | None:
    """Return zero at success or the best remaining distance of the selected beam."""
    return 0 if stopped else min_remaining_distance(x, selected)
