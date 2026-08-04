"""Shared helpers for Game24 step processing: state parsing, validation,
disposition detection.  Extracted from ``game24_task_recorder.py`` to stay
under the 250 pure-LOC ceiling.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Protocol as _Protocol

from federated_mcts.evaluation.game24_metrics_types import DispositionLiteral


class _SolverTask(_Protocol):
    steps: int

    def get_input(self, idx: int) -> str: ...

    @staticmethod
    def canonical_state_key(x: str, y: str) -> tuple[Fraction, ...]: ...

    @staticmethod
    def is_success_state(x: str, y: str) -> bool: ...


def parse_canonical_numbers(y: str) -> tuple[Fraction, ...]:
    import re

    match = re.search(r"left:\s*([^)]+)", y)
    parts = match.group(1).strip().split() if match else y.strip().split()
    nums: list[Fraction] = []
    for p in parts:
        try:
            nums.append(Fraction(p))
        except (ValueError, ZeroDivisionError):
            pass
    return tuple(sorted(nums))


def _is_valid_candidate(y: str) -> bool:
    import re

    return bool(re.search(r"\d+\s*[\+\-\*\/]\s*\d+\s*=\s*\d+", y))


def detect_disposition(
    *,
    ci: int,
    candidate: str,
    selected_indices: frozenset[int],
    is_last_step: bool,
    task: _SolverTask,
    x: str,
) -> tuple[DispositionLiteral, bool]:
    selected = ci in selected_indices
    try:
        key = task.canonical_state_key(x, candidate)
    except Exception:
        key = parse_canonical_numbers(candidate)
    is_terminal = len(key) <= 1
    try:
        success = task.is_success_state(x, candidate)
    except Exception:
        success = is_terminal and len(key) == 1 and list(key)[0] == Fraction(24)
    if success:
        return ("terminal_solved", True)
    if is_terminal or is_last_step:
        if is_terminal and not _is_valid_candidate(candidate):
            return ("terminal_invalid", True)
        return ("terminal_unsolved", True)
    if selected:
        return ("selected", False)
    return ("none", False)
