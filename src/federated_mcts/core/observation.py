"""Low-level, task-agnostic search-observation seam.

Core search code (generation and evaluation) only needs to *report* what the
model produced.  It must not import Game24 evaluation instrumentation: that
would invert the dependency direction (core -> evaluation) and drag run-time
telemetry concerns into the search primitives.

This module therefore owns the small, typed vocabulary the search primitives
speak:

* plain ``TypedDict`` records — trivially serialised, field names are the
  baseline telemetry JSON contract;
* two narrow sink protocols (:class:`GenerationObservationSink`,
  :class:`VoteObservationSink`) with a single ``record_step`` method;
* a no-op sink (:data:`NULL_OBSERVATION_SINK`) so callers can always call
  through without a ``None`` check at the observation site;
* one vote-choice parser (:func:`parse_vote_choice`) shared by the decision
  path and by telemetry, so the two can never disagree.

Record construction lives here too (:func:`record_proposal_step`,
:func:`record_unchecked_proposals`, :func:`record_vote_step`) so there is a
single implementation of the record shapes.  Task-specific collectors
(``federated_mcts.evaluation.game24_instrumentation``) are adapters: they
implement the sink protocols and keep the historical class names.

Observation is passive: a sink sees records after the fact and must not
influence candidates, scores, or client call counts.
"""

from __future__ import annotations

import re
from typing import NamedTuple, Protocol, TypedDict


# ── Typed records (baseline telemetry JSON field names) ──────────────

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


# ── Sink protocols ───────────────────────────────────────────────────

class GenerationObservationSink(Protocol):
    """Collects generation telemetry without affecting candidates."""

    def record_step(self, record: ProposalStepRecord) -> None:
        ...


class VoteObservationSink(Protocol):
    """Collects vote telemetry without affecting vote totals."""

    def record_step(self, record: VoteAggregateRecord) -> None:
        ...


class NullObservationSink:
    """Does nothing; usable where a sink is required but not wanted."""

    def record_step(self, record: object) -> None:
        return None


NULL_OBSERVATION_SINK = NullObservationSink()


# ── Record construction ──────────────────────────────────────────────

def record_proposal_step(
    sink: GenerationObservationSink,
    depth: int,
    parent_index: int,
    parent_state: str,
    raw_completions: list[str],
    retry_count: int,
    proposals: list[CandidateProposalRecord],
) -> None:
    """Append one generation call's passive telemetry."""
    sink.record_step(ProposalStepRecord(
        depth=depth,
        parent_index=parent_index,
        parent_state=parent_state,
        raw_completions=raw_completions,
        retry_count=retry_count,
        proposals=proposals,
    ))


def record_unchecked_proposals(
    sink: GenerationObservationSink,
    depth: int,
    parent_index: int,
    parent_state: str,
    proposals: list[str],
    candidates: list[str],
) -> None:
    """Append successful telemetry for unchecked proposal lines."""
    recorded_proposals = [CandidateProposalRecord(
        depth=depth,
        parent_index=parent_index,
        retry_ordinal=0,
        completion_index=0,
        raw_line=proposal,
        is_correct=True,
        updated_text=candidate,
        reason="accepted",
    ) for proposal, candidate in zip(proposals, candidates)]
    record_proposal_step(
        sink, depth, parent_index, parent_state, ["\n".join(proposals)],
        1, recorded_proposals)


def record_vote_step(
    sink: VoteObservationSink,
    depth: int,
    evaluator_mode: str,
    evaluator_identity: str,
    n_candidates: int,
    vote_totals: list[int],
    calls: list[VoteCallRecord],
) -> None:
    """Append one evaluator call aggregate without altering vote totals."""
    sink.record_step(VoteAggregateRecord(
        depth=depth,
        n_candidates=n_candidates,
        evaluator_mode=evaluator_mode,
        evaluator_identity=evaluator_identity,
        call_count=len(calls),
        vote_totals=vote_totals,
        calls=calls,
    ))


# ── Single vote-choice parser ────────────────────────────────────────

VOTE_CHOICE_PATTERN: re.Pattern[str] = re.compile(
    r".*best choice is.*?(?<!-)(\d+)", re.DOTALL | re.IGNORECASE
)


class VoteChoice(NamedTuple):
    """Parsed vote response: ``(index, matched, failure_reason)``."""

    vote_index: int | None
    matched: bool
    failure_reason: str | None


def parse_vote_choice(raw: str) -> VoteChoice:
    """Parse one evaluator vote response.

    This is the single parser used by both the decision path
    (``Game24Task.vote_outputs_unwrap``) and by telemetry, so the two can
    never disagree about whether a response names a candidate.
    Range checking against the candidate list is the caller's job.
    """
    match = VOTE_CHOICE_PATTERN.search(raw)
    if match is None:
        return VoteChoice(None, False, "no_match")
    return VoteChoice(int(match.group(1)), True, None)
