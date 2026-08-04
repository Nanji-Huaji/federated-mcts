"""Helpers for persisting passive Game24 generation telemetry."""

from __future__ import annotations

from federated_mcts.evaluation.game24_instrumentation import (
    CandidateProposalRecord,
    GenerationRecorder,
    ProposalStepRecord,
)


def record_proposal_step(
    recorder: GenerationRecorder,
    depth: int,
    parent_index: int,
    parent_state: str,
    raw_completions: list[str],
    retry_count: int,
    proposals: list[CandidateProposalRecord],
) -> None:
    """Append one generation call's passive telemetry."""
    recorder.record_step(ProposalStepRecord(
        depth=depth,
        parent_index=parent_index,
        parent_state=parent_state,
        raw_completions=raw_completions,
        retry_count=retry_count,
        proposals=proposals,
    ))


def record_unchecked_proposals(
    recorder: GenerationRecorder,
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
        recorder, depth, parent_index, parent_state, ["\n".join(proposals)],
        1, recorded_proposals)
