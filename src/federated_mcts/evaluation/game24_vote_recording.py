"""Compatibility shim for Game24 vote telemetry.

The record construction now lives in :mod:`federated_mcts.core.observation`
(the core search primitives must not depend on evaluation code).  This module
keeps its historical import path and function signature for existing callers.
"""

from __future__ import annotations

from federated_mcts.core.observation import (
    VoteAggregateRecord,
    VoteCallRecord,
    VoteObservationSink,
    record_vote_step,
)

__all__ = [
    "VoteAggregateRecord",
    "VoteCallRecord",
    "VoteObservationSink",
    "record_vote_step",
]
