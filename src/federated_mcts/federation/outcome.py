"""Task verdicts are resolved before a search outcome is published."""
from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TaskOutcome:
    candidates: tuple[str, ...]
    verdicts: tuple[dict, ...]
    trace: dict


def evaluate_outputs(task, index: int, candidates: list[str]) -> list[dict]:
    """Apply the task's exact verifier and the runner's duplicate-answer rule."""
    seen: set[str] = set()
    verdicts = []
    for candidate in candidates:
        if hasattr(task, "test_output_modify"):
            verdict, normalized = task.test_output_modify(index, candidate)
        else:
            verdict, normalized = task.test_output(index, candidate), candidate
        if normalized in seen:
            verdict = {"r": 0}
        seen.add(normalized)
        verdicts.append(dict(verdict))
    return verdicts


def finish_task(solver, task, index: int, *, method: str, **kwargs) -> TaskOutcome:
    """Own solve → exact verification → episode completion as one operation.

    The task's exact verifier runs before any outcome is returned, and a DQN
    terminal transition is published once, only after that verdict is known.
    Callers never touch the solver's session, and never mutate a returned
    trace or rewrite a terminal record to correct ``success``.  Legacy
    per-policy methods remain internal seams.
    """
    methods = {
        "naive": solver.naive_solve, "tot": solver.solve,
        "speculative_solve": solver.speculative_solve,
        "federated_solve": solver.federated_solve,
    }
    defer = getattr(solver, "deferred_dqn_terminal_flush", None)
    scope = defer() if defer is not None else nullcontext()
    with scope:
        candidates, trace = methods[method](task, index, **kwargs)
    verdicts = evaluate_outputs(task, index, candidates)
    session = getattr(solver, "dqn_session", None)
    if session is not None and method in ("tot", "federated_solve"):
        # Only the search methods that can own a DQN episode complete one;
        # naive/speculative never create a session, so a stale one is left
        # untouched.
        session.finalize(any(verdict["r"] for verdict in verdicts))
    return TaskOutcome(tuple(candidates), tuple(deepcopy(verdicts)), deepcopy(trace))
