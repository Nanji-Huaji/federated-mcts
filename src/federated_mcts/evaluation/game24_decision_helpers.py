"""Decision-recording helpers extracted from TaskSessionRecorder.

Keeps `game24_task_recorder.py` under 250 pure LOC by housing:
- Step disposition detection and state parsing
- Audit hash computation for record_decision steps
- Expanded disposition application for parent-lineage back-propagation
- Failed-step event construction from generation recorder state
"""

from __future__ import annotations

import hashlib
from fractions import Fraction
from typing import Protocol as _Protocol

from federated_mcts.evaluation.game24_metrics_types import (
    CandidateEvent,
    DispositionLiteral,
    RejectionReasonLiteral,
)
from federated_mcts.evaluation.game24_instrumentation import (
    GenerationRecorder,
)
from federated_mcts.evaluation.game24_metrics import (
    make_candidate_event,
)


class _SolverTask(_Protocol):
    steps: int
    def get_input(self, idx: int) -> str: ...
    @staticmethod
    def canonical_state_key(x: str, y: str) -> tuple[Fraction, ...]: ...
    @staticmethod
    def is_success_state(x: str, y: str) -> bool: ...


def _parse_canonical_numbers(y: str) -> tuple[Fraction, ...]:
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
    return bool(re.search(r'\d+\s*[\+\-\*\/]\s*\d+\s*=\s*\d+', y))


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
        key = _parse_canonical_numbers(candidate)
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


def compute_decision_audit(
    *,
    enable_audit: bool,
    task_id: str,
    task_input: str,
    depth: int,
    candidates: list[str],
    values: list[float],
    selected_indices: list[int],
    events: list[CandidateEvent],
    is_stopping: bool,
    decision_policy: str,
) -> str:
    """Run post-decision oracle audit and return deterministic snapshot hash."""
    if not enable_audit:
        return ""

    try:
        from federated_mcts.evaluation.game24_oracle_audit import (
            PostDecisionCandidate,
            PostDecisionSnapshot,
            run_audit,
            serialize_snapshot,
        )

        sel_set = frozenset(selected_indices)
        sc = [
            PostDecisionCandidate(
                idx=ci,
                raw_text=c,
                canonical_numbers=_parse_canonical_numbers(c),
                evaluator_score=float(values[ci]) if ci < len(values) else None,
                evaluator_mode="vote",
                is_selected=ci in sel_set,
                is_terminal=events[ci]["is_terminal"],
            )
            for ci, c in enumerate(candidates)
        ]
        snap = PostDecisionSnapshot(
            schema_version=1,
            task_id=task_id,
            task_input=task_input,
            depth=depth,
            candidates=tuple(sc),
            selected_indices=tuple(selected_indices),
            prompt_call_identity=f"step_{depth}",
            is_stopping=is_stopping,
            decision_policy=decision_policy,
        )
        _ = run_audit(snap)
        return hashlib.sha256(serialize_snapshot(snap)).hexdigest()
    except Exception:
        return ""


def apply_expanded_disposition(
    step_records: list[dict],
    parent_lid_set: frozenset[str],
) -> None:
    """Retroactively mark previous-step parents as 'expanded'.

    Mutates ``step_records[-1]["candidates"]`` in place.
    """
    if not parent_lid_set or not step_records:
        return
    prev_events = step_records[-1].get("candidates", [])
    if not prev_events:
        return
    rebuilt: list[dict] = []
    for ev in prev_events:
        lid = str(ev.get("lineage_id", ""))
        disp = str(ev.get("disposition", ""))
        if lid in parent_lid_set and disp in ("selected", "none"):
            new_ev = dict(ev)
            new_ev["disposition"] = "expanded"
            rebuilt.append(new_ev)
        else:
            rebuilt.append(ev)
    step_records[-1]["candidates"] = rebuilt


_KNOWN_FAILED_REASONS: dict[str, RejectionReasonLiteral] = {
    "empty_content": "empty_content",
    "malformed_format": "malformed_format",
}


def build_failed_step_events(
    gen_rec: GenerationRecorder,
    task_id: str,
    depth: int,
) -> list[CandidateEvent]:
    """Convert a GenerationRecorder's proposal records into CandidateEvents."""
    events: list[CandidateEvent] = []
    for step_rec in gen_rec.steps:
        for pi, prop in enumerate(step_rec["proposals"]):
            events.append(make_candidate_event(
                task_id=task_id, depth=depth,
                parent_lineage_id=str(depth - 1) if depth > 0 else "",
                generation_ordinal=pi, stage="proposed", disposition="none",
                raw_content=prop["raw_line"],
                rejection_reason=_KNOWN_FAILED_REASONS.get(prop["reason"]),
                is_selected=False, is_terminal=False))
    return events
