"""Compatibility shim for Game24 generation telemetry.

The record construction now lives in :mod:`federated_mcts.core.observation`
(the core search primitives must not depend on evaluation code).  This module
keeps its historical import path and function signatures for existing callers.
"""

from __future__ import annotations

from federated_mcts.core.observation import (
    CandidateProposalRecord,
    GenerationObservationSink,
    ProposalStepRecord,
    record_proposal_step,
    record_unchecked_proposals,
)

__all__ = [
    "CandidateProposalRecord",
    "GenerationObservationSink",
    "ProposalStepRecord",
    "record_proposal_step",
    "record_unchecked_proposals",
]
