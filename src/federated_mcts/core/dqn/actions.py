"""Action space for the Budget-Aware DQN controller.

Actions are the Cartesian product of beam width {2, 3, 4, 5} and joint-rank
usage {False, True}, enumerated beam-major:
action = 2 * beam_index + joint_rank_index.
"""

from __future__ import annotations

ACTION_BEAMS: tuple[int, ...] = (2, 3, 4, 5)
ACTION_JOINT_RANKS: tuple[bool, ...] = (False, True)
ACTION_COUNT: int = len(ACTION_BEAMS) * len(ACTION_JOINT_RANKS)


def action_for(beam: int, joint_rank: bool) -> int:
    if beam not in ACTION_BEAMS:
        raise ValueError(f"beam must be one of {ACTION_BEAMS}, got {beam!r}")
    if joint_rank not in ACTION_JOINT_RANKS:
        raise ValueError(f"joint_rank must be one of {ACTION_JOINT_RANKS}, got {joint_rank!r}")
    beam_index = ACTION_BEAMS.index(beam)
    joint_rank_index = ACTION_JOINT_RANKS.index(joint_rank)
    return beam_index * len(ACTION_JOINT_RANKS) + joint_rank_index


def beam_and_joint_rank(action: int) -> tuple[int, bool]:
    if not isinstance(action, int) or not 0 <= action < ACTION_COUNT:
        raise ValueError(f"action must be an int in [0, {ACTION_COUNT}), got {action!r}")
    return (
        ACTION_BEAMS[action // len(ACTION_JOINT_RANKS)],
        ACTION_JOINT_RANKS[action % len(ACTION_JOINT_RANKS)],
    )
