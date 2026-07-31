"""Pure deterministic Blocksworld symbolic oracle.

Built on the engine's pure functions: candidate actions are enumerated in
family order (pick-up, put-down, stack, unstack) with blocks sorted, kept only
when ``step_state`` accepts them; ``shortest_plan`` runs an exact breadth-first
search over the reachable state graph (visited on enqueue); and
``evaluate_trajectory`` replays a submission and stops at the first malformed
or illegal line.  No I/O, no randomness, no LLM calls.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Literal

from federated_mcts.tasks.blocksworld_engine import (
    action_lines,
    is_goal_satisfied,
    parse_action_line,
    step_state,
)

SearchStatus = Literal["solved", "unreachable", "cutoff"]
TrajectoryStatus = Literal["valid", "malformed", "illegal"]

SOLVED: SearchStatus = "solved"
UNREACHABLE: SearchStatus = "unreachable"
CUTOFF: SearchStatus = "cutoff"
VALID: TrajectoryStatus = "valid"
MALFORMED: TrajectoryStatus = "malformed"
ILLEGAL: TrajectoryStatus = "illegal"

Action = tuple[str, tuple[str, ...]]


@dataclass(frozen=True)
class ShortestPlanResult:
    """Exact shortest-path search outcome over the Blocksworld state graph."""

    status: SearchStatus
    plan: tuple[Action, ...]
    explored: int


@dataclass(frozen=True)
class TrajectoryMetrics:
    """Action-by-action evaluation of a submitted trajectory.

    ``plan_length`` is the actual legal action count for a fully valid
    submission and ``None`` for a malformed or illegal one.  ``optimal_length``
    is the exact optimum when one exists and is known (``None`` for an
    unreachable goal).  ``excess_steps`` and ``optimality_ratio`` are only
    populated for a fully valid trajectory that reaches the goal with a finite
    optimum.
    """

    status: TrajectoryStatus
    submitted_length: int
    legal_count: int
    legal_rate: float
    first_failure_index: int | None
    success: bool
    plan_length: int | None
    optimal_length: int | None
    excess_steps: int | None
    optimality_ratio: float | None


def legal_actions(parsed: dict, state: frozenset | None) -> tuple[Action, ...]:
    """Deterministic, duplicate-free actions legal in ``state``.

    Candidate families in pick-up, put-down, stack, unstack order, blocks and
    ordered pairs sorted; only candidates ``step_state`` accepts survive.
    """
    if state is None:
        return ()
    blocks = sorted(parsed["blocks"])
    candidates: list[Action] = [
        ("pick-up", (block,)) for block in blocks
    ] + [
        ("put-down", (block,)) for block in blocks
    ] + [
        ("stack", (block, target))
        for block in blocks
        for target in blocks
        if block != target
    ] + [
        ("unstack", (block, below))
        for block in blocks
        for below in blocks
        if block != below
    ]
    return tuple(c for c in candidates if step_state(state, c) is not None)


def shortest_plan(parsed: dict, max_depth: int | None = None) -> ShortestPlanResult:
    """Exact BFS shortest plan; reports solved / unreachable / cutoff.

    Visited-on-enqueue; ``explored`` counts the distinct states actually
    explored (popped from the frontier, start included).  With ``max_depth``,
    a state at the bound reports ``CUTOFF`` only when the bound suppresses at
    least one unvisited legal successor; a bound frontier whose successors are
    all already visited contributes nothing and the search falls through to
    ``UNREACHABLE`` once the whole reachable graph is exhausted.
    """
    start = parsed["init"]
    visited = {start}
    queue = deque([(start, (), 0)])
    explored = 0
    reached_bound = False
    while queue:
        state, plan, depth = queue.popleft()
        explored += 1
        if is_goal_satisfied(parsed, state):
            return ShortestPlanResult(SOLVED, plan, explored)
        if max_depth is not None and depth >= max_depth:
            if any(
                step_state(state, action) not in visited
                for action in legal_actions(parsed, state)
            ):
                reached_bound = True
            continue
        for action in legal_actions(parsed, state):
            nxt = step_state(state, action)
            if nxt is not None and nxt not in visited:
                visited.add(nxt)
                queue.append((nxt, plan + (action,), depth + 1))
    if reached_bound:
        return ShortestPlanResult(CUTOFF, (), explored)
    return ShortestPlanResult(UNREACHABLE, (), explored)


def evaluate_trajectory(
    parsed: dict,
    trajectory: str,
    optimal: int | None = None,
) -> TrajectoryMetrics:
    """Replay a trajectory, stopping at the first malformed or illegal line.

    ``success`` is True only for a fully valid trajectory whose final state
    satisfies the goal.  The exact optimum is computed once when ``optimal``
    is omitted: ``optimal_length`` is the shortest-plan length for a reachable
    goal and ``None`` for an unreachable one.  ``excess_steps`` and
    ``optimality_ratio`` are populated only on success with a finite optimum;
    the ratio is the actual successful plan length over the optimum (1.0
    optimal, >1 suboptimal), with a zero optimum yielding 1.0 for a zero-step
    success and ``inf`` for a positive successful detour.
    """
    if optimal is None:
        result = shortest_plan(parsed)
        optimal_length: int | None = (
            len(result.plan) if result.status == SOLVED else None
        )
    else:
        optimal_length = optimal
    state = parsed["init"]
    lines = action_lines(trajectory)
    submitted = len(lines)
    status: TrajectoryStatus = VALID
    first_failure: int | None = None
    legal_count = 0
    for index, line in enumerate(lines):
        action = parse_action_line(line, parsed["blocks"])
        if action is None:
            status, first_failure = MALFORMED, index
            break
        nxt = step_state(state, action)
        if nxt is None:
            status, first_failure = ILLEGAL, index
            break
        state = nxt
        legal_count += 1
    plan_length = legal_count if status == VALID else None
    success = status == VALID and is_goal_satisfied(parsed, state)
    if success and optimal_length is not None:
        excess = legal_count - optimal_length
        if optimal_length == 0:
            ratio = 1.0 if legal_count == 0 else float("inf")
        else:
            ratio = legal_count / optimal_length
    else:
        excess, ratio = None, None
    return TrajectoryMetrics(
        status=status,
        submitted_length=submitted,
        legal_count=legal_count,
        legal_rate=legal_count / submitted if submitted else 1.0,
        first_failure_index=first_failure,
        success=success,
        plan_length=plan_length,
        optimal_length=optimal_length,
        excess_steps=excess,
        optimality_ratio=ratio,
    )
