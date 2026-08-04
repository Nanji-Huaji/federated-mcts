"""Passive telemetry recorders for Game24 generation and vote evaluation.

Thread an optional recorder instance through ``get_proposals_with_check``,
``get_proposals_without_check``, ``get_proposals``, and ``get_votes``.
When ``recorder=None`` (the default), behaviour is byte-identical to the
uninstrumented path.  When a recorder is supplied, it captures raw
completions, parse results, rejection reasons, raw vote text, parsed vote
indices, parse-failure reasons, out-of-range votes, and evaluator identity —
without mutating the candidate/vote values returned to the caller.

All record types are plain ``TypedDict`` so serialisation is trivial.
"""

from __future__ import annotations

import re
from typing import TypedDict


# ── Proposal record ───────────────────────────────────────────────

class CandidateProposalRecord(TypedDict):
    """One raw proposal line and its processing outcome."""

    depth: int
    """Search depth (step index)."""

    parent_index: int
    """Index of the parent ``y`` in the current ``ys`` list."""

    retry_ordinal: int
    """Which retry loop this call belongs to (0-indexed)."""

    completion_index: int
    """Index of this completion in the multi-completion response."""

    raw_line: str
    """Exact line text from the model (may be empty)."""

    is_correct: bool
    """Outcome of ``process_generate_result`` / format check."""

    updated_text: str
    """The updated proposal string (or original if rejected)."""

    reason: str
    """Short rejection reason when ``is_correct`` is False."""


# ── Proposal aggregate ────────────────────────────────────────────

class ProposalStepRecord(TypedDict):
    """Aggregate for one ``get_proposals`` call: raw completions + proposals."""

    depth: int
    parent_index: int
    parent_state: str
    raw_completions: list[str]
    """Full completion strings returned by the model client."""

    retry_count: int
    """Total number of API calls in the retry loop."""

    proposals: list[CandidateProposalRecord]
    """Every line processed (including empty/rejected)."""


class GenerationRecorder:
    """Collects raw proposal events without affecting candidate lists.

    Usage::

        recorder = GenerationRecorder()
        candidates = get_proposals(args, step, task, x, y, recorder=recorder)
        # candidates is identical to the uninstrumented path
        print(recorder.steps)
    """

    def __init__(self) -> None:
        self.steps: list[ProposalStepRecord] = []

    def record_step(self, rec: ProposalStepRecord) -> None:
        self.steps.append(rec)

    @property
    def total_proposed(self) -> int:
        return sum(len(s["proposals"]) for s in self.steps)

    @property
    def total_accepted(self) -> int:
        return sum(1 for s in self.steps for p in s["proposals"] if p["is_correct"])


# ── Vote record ───────────────────────────────────────────────────

class VoteCallRecord(TypedDict):
    """One evaluator invocation and its parsed results."""

    call_index: int
    """0-based index of this call across all vote calls for the step."""

    evaluator_mode: str
    """``"vote"`` or ``"value"`` (the effective method_evaluate)."""

    evaluator_identity: str
    """Client identity (e.g. ``"remote_client"``)."""

    raw_output: str
    """Full raw text from this evaluator call."""

    parsed_index: int | None
    """Parsed candidate index or ``None`` when unparseable."""

    parse_success: bool
    """Whether the regex matched a candidate index."""

    failure_reason: str | None
    """``"no_match"`` or ``"out_of_range"`` when the vote was not counted."""

    in_range: bool
    """Whether ``parsed_index`` is within the valid candidate range."""


class VoteAggregateRecord(TypedDict):
    """Aggregate for one ``get_votes`` call."""

    depth: int
    n_candidates: int
    evaluator_mode: str
    evaluator_identity: str
    call_count: int
    vote_totals: list[int]
    """Per-candidate final vote counts (identical to the returned list)."""

    calls: list[VoteCallRecord]
    """Details of every raw evaluator call."""


class VoteRecorder:
    """Collects raw vote events without affecting vote totals.

    Usage::

        recorder = VoteRecorder()
        values = get_votes(args, task, x, ys, n_eval, recorder=recorder, client=gpt)
        # values is identical to the uninstrumented path
        print(recorder.steps)
    """

    def __init__(self) -> None:
        self.steps: list[VoteAggregateRecord] = []

    def record_step(self, rec: VoteAggregateRecord) -> None:
        self.steps.append(rec)

    @property
    def total_calls(self) -> int:
        return sum(s["call_count"] for s in self.steps)

    @property
    def total_parse_failures(self) -> int:
        result = 0
        for s in self.steps:
            for c in s["calls"]:
                if not c["parse_success"]:
                    result += 1
                elif not c["in_range"]:
                    result += 1
        return result


# ── Vote parse helper ─────────────────────────────────────────────

_VOTE_PARSE_PATTERN: re.Pattern[str] = re.compile(
    r".*best choice is.*?(?<!-)(\d+)", re.DOTALL | re.IGNORECASE
)


def parse_vote_response(raw: str) -> tuple[int | None, bool, str | None]:
    """Parse one evaluator vote response for telemetry.

    Returns ``(parsed_index, parse_success, failure_reason)`` where
    ``failure_reason`` is ``"no_match"`` when the regex fails to match,
    or ``None`` otherwise.  Does NOT check whether the index is within
    the valid candidate range — callers must perform that check.

    The regex is byte-identical to the decision parser in
    ``Game24Task.vote_outputs_unwrap`` so telemetry never reports parse
    success where the decision parser does not, and vice versa.
    """
    match = _VOTE_PARSE_PATTERN.search(raw)
    if not match:
        return (None, False, "no_match")
    return (int(match.group(1)), True, None)
