"""Helpers for persisting passive Game24 vote telemetry."""

from __future__ import annotations

from federated_mcts.evaluation.game24_instrumentation import (
    VoteAggregateRecord,
    VoteCallRecord,
    VoteRecorder,
)


def record_vote_step(
    recorder: VoteRecorder,
    depth: int,
    evaluator_mode: str,
    evaluator_identity: str,
    n_candidates: int,
    vote_totals: list[int],
    calls: list[VoteCallRecord],
) -> None:
    """Append one evaluator call aggregate without altering vote totals."""
    recorder.record_step(VoteAggregateRecord(
        depth=depth,
        n_candidates=n_candidates,
        evaluator_mode=evaluator_mode,
        evaluator_identity=evaluator_identity,
        call_count=len(calls),
        vote_totals=vote_totals,
        calls=calls,
    ))
