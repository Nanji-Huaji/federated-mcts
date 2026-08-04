"""DQN episode finalization shared by the standard solver."""

from __future__ import annotations


def finish_dqn_session(dqn_session, infos) -> None:
    """Flush final DQN transitions into the latest step information."""
    if dqn_session is not None:
        dqn_session.finish_episode()
        remaining = dqn_session.drain_transitions()
        if remaining and infos:
            infos[-1].setdefault("dqn_transitions", []).extend(remaining)
