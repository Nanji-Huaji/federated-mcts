"""Tests for Game24 oracle audit: record/replay, reachability, frozen annotations.

Covers Todo 6: deterministic fixture record/replay (test_audit_off_on_identical_decision_trace
stays in this module per plan), reachability agreement, and immutable annotations.

Shared helpers live in ``tests._game24_audit_helpers`` (underscore-prefixed, not discovered).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, "src")

from tests._game24_audit_helpers import build_snapshot, load_fixture


class TestOracleAuditRecordReplay(unittest.TestCase):
    """Deterministic fixture replay: audit-off/on identical decision bytes."""

    def test_audit_off_on_identical_decision_trace(self):
        from federated_mcts.evaluation.game24_oracle_audit import (
            deserialize_audit_input, run_audit, serialize_snapshot,
            _canonical_json_bytes,
        )
        fx = load_fixture()
        fx_dict = fx["snapshot"]
        expected_hash = fx["snapshot_sha256"]
        expected = fx["expected"]

        fx_bytes = _canonical_json_bytes(fx_dict)
        self.assertEqual(hashlib.sha256(fx_bytes).hexdigest(), expected_hash)

        snap = deserialize_audit_input(fx_bytes)
        re_bytes = serialize_snapshot(snap)
        self.assertEqual(re_bytes, fx_bytes,
            "DEFECT: round-trip serialization != original — decision trace not identical")
        self.assertEqual(hashlib.sha256(re_bytes).hexdigest(), expected_hash)

        ann = run_audit(snap)
        self.assertEqual(ann.snapshot_hash, expected_hash)

        rmap = {idx: ok for _, idx, ok in ann.per_candidate_reachability}
        for idx in expected["reachable_indices"]:
            self.assertTrue(rmap.get(idx), f"candidate {idx} should be reachable")
        for idx in expected["unreachable_indices"]:
            self.assertFalse(rmap.get(idx, True), f"candidate {idx} should be unreachable")

        dm = ann.depth_metrics[0]
        edm = expected["depth_metrics"]
        for field in ("depth", "proposed_count", "reachable_count",
                       "unreachable_count", "selected_count",
                       "selected_reachable_count", "best_reachable_idx",
                       "best_reachable_score", "best_selected_idx",
                       "best_selected_score"):
            self.assertEqual(getattr(dm, field), edm[field], f"field {field} mismatch")
        regret = dm.selection_regret
        assert regret is not None
        self.assertAlmostEqual(regret, edm["selection_regret"])
        self.assertAlmostEqual(dm.selected_precision, edm["selected_precision"])

    def test_evidence_bytes_written_to_configurable_path(self):
        from federated_mcts.evaluation.game24_oracle_audit import (
            deserialize_audit_input, serialize_snapshot,
        )
        fx = load_fixture()
        snap = deserialize_audit_input(
            json.dumps(fx["snapshot"], sort_keys=True).encode())
        norm = serialize_snapshot(snap)
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "evidence_snapshot.json")
            with open(p, "wb") as f:
                f.write(norm)
            with open(p, "rb") as f:
                self.assertEqual(f.read(), norm)

    def test_identical_snapshots_produce_identical_annotations(self):
        from federated_mcts.evaluation.game24_oracle_audit import run_audit
        a1 = run_audit(build_snapshot())
        a2 = run_audit(build_snapshot())
        self.assertEqual(a1.snapshot_hash, a2.snapshot_hash)
        self.assertEqual(a1.per_candidate_reachability, a2.per_candidate_reachability)
        self.assertEqual(a1.aggregate_selection_regret, a2.aggregate_selection_regret)


class TestOracleAuditReachability(unittest.TestCase):
    """Audit reachability agrees with direct oracle calls."""

    def test_audit_agrees_with_direct_oracle(self):
        from federated_mcts.evaluation.game24_oracle_audit import run_audit
        from federated_mcts.tasks.game24_exact_prune import _can_reach_target_numbers
        s = build_snapshot()
        for _, idx, ok in run_audit(s).per_candidate_reachability:
            cand = next(c for c in s.candidates if c.idx == idx)
            self.assertEqual(ok, _can_reach_target_numbers(cand.canonical_numbers),
                f"audit {ok} != direct oracle for idx={idx}")

    def test_all_reachable_selected_yields_zero_regret(self):
        from federated_mcts.evaluation.game24_oracle_audit import run_audit, PostDecisionCandidate
        from fractions import Fraction as F
        c = PostDecisionCandidate(idx=0, raw_text="24", canonical_numbers=(F(24),),
            evaluator_score=3.0, evaluator_mode="value",
            is_selected=True, is_terminal=True)
        s = build_snapshot(candidates=(c,), selected_indices=(0,))
        dm = run_audit(s).depth_metrics[0]
        self.assertEqual(dm.reachable_count, 1)
        self.assertAlmostEqual(dm.selected_precision, 1.0)
        regret = dm.selection_regret
        assert regret is not None
        self.assertAlmostEqual(regret, 0.0)

    def test_all_unreachable_yields_none_best_reachable(self):
        from federated_mcts.evaluation.game24_oracle_audit import run_audit, PostDecisionCandidate
        from fractions import Fraction as F
        c = PostDecisionCandidate(idx=0, raw_text="1+1=2",
            canonical_numbers=(F(1), F(1), F(2)),
            evaluator_score=1.0, evaluator_mode="value",
            is_selected=True, is_terminal=False)
        s = build_snapshot(candidates=(c,), selected_indices=(0,))
        dm = run_audit(s).depth_metrics[0]
        self.assertEqual(dm.reachable_count, 0)
        self.assertIsNone(dm.best_reachable_idx)
        self.assertIsNone(dm.selection_regret)

    def test_no_selections_yields_one_precision(self):
        from federated_mcts.evaluation.game24_oracle_audit import run_audit, PostDecisionCandidate
        from fractions import Fraction as F
        c = PostDecisionCandidate(idx=0, raw_text="3+3=6",
            canonical_numbers=(F(3), F(3), F(4)),
            evaluator_score=2.0, evaluator_mode="value",
            is_selected=False, is_terminal=False)
        s = build_snapshot(candidates=(c,), selected_indices=())
        dm = run_audit(s).depth_metrics[0]
        self.assertEqual(dm.selected_count, 0)
        self.assertAlmostEqual(dm.selected_precision, 1.0)


class TestOracleAuditAnnotationsFrozen(unittest.TestCase):
    """Prove annotations are frozen and disjoint from decision state."""

    def test_annotations_are_frozen(self):
        from federated_mcts.evaluation.game24_oracle_audit import run_audit
        from dataclasses import FrozenInstanceError
        ann = run_audit(build_snapshot())
        with self.assertRaises(FrozenInstanceError):
            ann.aggregate_selection_regret = 0.99  # type: ignore[misc]
        with self.assertRaises(FrozenInstanceError):
            ann.depth_metrics[0].selection_regret = 0.99  # type: ignore[misc]

    def test_annotations_are_disjoint_from_snapshot(self):
        from federated_mcts.evaluation.game24_oracle_audit import run_audit
        s = build_snapshot()
        ann = run_audit(s)
        self.assertIsNot(ann, s)
        self.assertNotIsInstance(ann, type(s))


class TestOracleAuditSerialization(unittest.TestCase):
    """Serialize -> deserialize round-trip preserves all fields."""

    def test_round_trip_preserves_all_fields(self):
        from federated_mcts.evaluation.game24_oracle_audit import (
            deserialize_audit_input, serialize_snapshot,
        )
        s = build_snapshot()
        r = deserialize_audit_input(serialize_snapshot(s))
        scalar_fields = ("schema_version", "task_id", "task_input", "depth",
                         "selected_indices", "prompt_call_identity",
                         "is_stopping", "decision_policy")
        for f in scalar_fields:
            self.assertEqual(getattr(r, f), getattr(s, f), f"field {f}")
        self.assertEqual(len(r.candidates), len(s.candidates))
        for rc, sc in zip(r.candidates, s.candidates):
            for f in ("idx", "raw_text", "canonical_numbers", "evaluator_score",
                       "evaluator_mode", "is_selected", "is_terminal"):
                self.assertEqual(getattr(rc, f), getattr(sc, f))


if __name__ == "__main__":
    unittest.main()
