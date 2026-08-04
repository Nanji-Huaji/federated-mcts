"""Shared fixtures and helpers for Game24 oracle audit tests.

Underscore-prefixed so unittest discovery skips this module (no Test* classes).
Import from here in both test_game24_oracle_audit.py and its split sibling.
"""

from __future__ import annotations

import json
import os

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "game24")
FIXTURE_PATH = os.path.join(FIXTURE_DIR, "audit_recording.json")


def load_fixture() -> dict:
    with open(FIXTURE_PATH) as f:
        return json.load(f)


def build_snapshot(**kw):
    from fractions import Fraction as F
    from federated_mcts.evaluation.game24_oracle_audit import (
        PostDecisionCandidate, PostDecisionSnapshot, SNAPSHOT_SCHEMA_VERSION,
    )
    defaults: dict = dict(
        schema_version=SNAPSHOT_SCHEMA_VERSION, task_id="99",
        task_input="4 4 6 8", depth=1,
        candidates=(
            PostDecisionCandidate(idx=0, raw_text="4 * 6 = 24 (left: 24)",
                canonical_numbers=(F(24),), evaluator_score=3.0,
                evaluator_mode="value", is_selected=True, is_terminal=True),
            PostDecisionCandidate(idx=1, raw_text="4 + 4 = 8 (left: 6 8)",
                canonical_numbers=(F(6), F(8)), evaluator_score=2.5,
                evaluator_mode="vote", is_selected=False, is_terminal=False),
        ),
        selected_indices=(0,), prompt_call_identity="sha256:test-v1",
        is_stopping=False, decision_policy="local_validity",
    )
    defaults.update(kw)
    return PostDecisionSnapshot(**defaults)
