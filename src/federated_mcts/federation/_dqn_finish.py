"""DQN episode finalization shared by the standard and federated solvers."""

from __future__ import annotations


def build_dqn_session(args, *, evaluator_cache_for=None):
    """Load optional Torch dependencies only when a DQN run is requested."""
    from federated_mcts.core.dqn.factory import build_dqn_session as build
    return build(args, evaluator_cache_for=evaluator_cache_for)


def finish_dqn_session(dqn_session, infos, *, flush_terminal: bool = True) -> None:
    """Flush final DQN transitions into the latest step information.

    ``flush_terminal=False`` keeps the terminal record pending so the caller
    can publish it once with the exact verification reward.
    """
    if dqn_session is not None:
        dqn_session.finish_episode(flush_terminal=flush_terminal)
        remaining = dqn_session.drain_transitions()
        if remaining and infos:
            infos[-1].setdefault("dqn_transitions", []).extend(remaining)
