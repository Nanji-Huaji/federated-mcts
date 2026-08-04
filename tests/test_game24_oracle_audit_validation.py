"""Boundary validation and preflight/mutation tests for Game24 oracle audit.

Covers: NaN/Inf/bool rejection, exact types, duplicate idx, consistency
checks, preflight compatibility, serialization mutation.  All tests live
here to keep ``test_game24_oracle_audit.py`` under 250 pure LOC.

Shared helpers from ``tests._game24_audit_helpers`` (underscore-prefixed,
not discovered as tests).
"""

from __future__ import annotations

import json
import sys
import unittest

sys.path.insert(0, "src")

from tests._game24_audit_helpers import build_snapshot, load_fixture


class TestOracleAuditMutation(unittest.TestCase):
    """Tampered bytes -> deserialization / hash mismatch."""

    def test_empty_and_non_utf8_and_non_bytes_rejected(self):
        from federated_mcts.evaluation.game24_oracle_audit import deserialize_audit_input
        for bad_input in [b"", b"\xff\xfe\x00\x00", "not bytes"]:
            with self.assertRaises(ValueError):
                deserialize_audit_input(bad_input)  # type: ignore[arg-type]

    def test_wrong_schema_version_rejected(self):
        from federated_mcts.evaluation.game24_oracle_audit import (
            deserialize_audit_input, serialize_snapshot, _canonical_json_bytes,
        )
        raw = serialize_snapshot(build_snapshot())
        d = json.loads(raw.decode())
        d["schema_version"] = 999
        with self.assertRaises(ValueError):
            deserialize_audit_input(_canonical_json_bytes(d))

    def test_tampered_candidate_changes_serialization_hash(self):
        from federated_mcts.evaluation.game24_oracle_audit import (
            PostDecisionCandidate, serialize_snapshot,
        )
        s1 = build_snapshot()
        orig = s1.candidates[1]
        mod = PostDecisionCandidate(idx=1, raw_text="different text",
            canonical_numbers=orig.canonical_numbers,
            evaluator_score=orig.evaluator_score,
            evaluator_mode=orig.evaluator_mode,
            is_selected=orig.is_selected, is_terminal=orig.is_terminal)
        s2 = build_snapshot(candidates=(s1.candidates[0], mod), selected_indices=s1.selected_indices)
        self.assertNotEqual(serialize_snapshot(s1), serialize_snapshot(s2))


class TestOracleAuditPreflight(unittest.TestCase):
    """Preflight condition helpers: incompatible oracle-assisted policy fails before client."""

    def test_model_only_policies_are_compatible(self):
        from federated_mcts.evaluation.game24_oracle_audit import preflight_audit_compatible
        for policy in ("local_validity", "heuristic_rules"):
            ok, msg = preflight_audit_compatible(policy)
            self.assertTrue(ok, f"{policy}: {msg}")

    def test_oracle_exact_pruned_is_incompatible(self):
        from federated_mcts.evaluation.game24_oracle_audit import preflight_audit_compatible
        ok, msg = preflight_audit_compatible("oracle_exact_pruned")
        self.assertFalse(ok)
        self.assertIn("oracle-assisted", msg.lower())

    def test_invalid_inputs_rejected(self):
        from federated_mcts.evaluation.game24_oracle_audit import preflight_audit_compatible
        for bad in ["", 42, None]:
            ok, _ = preflight_audit_compatible(bad)  # type: ignore[arg-type]
            self.assertFalse(ok, f"should reject {bad!r}")

    def test_is_oracle_assisted_policy_correct(self):
        from federated_mcts.evaluation.game24_oracle_audit import is_oracle_assisted_policy
        self.assertTrue(is_oracle_assisted_policy("oracle_exact_pruned"))
        for p in ("local_validity", "heuristic_rules", "unknown"):
            self.assertFalse(is_oracle_assisted_policy(p))


class TestOracleAuditBoundaryValidation(unittest.TestCase):
    """Each defect rejected before frozen construction: NaN, bool-as-num, dup idx, etc."""

    def _snapshot_bytes(self, **overrides):
        from federated_mcts.evaluation.game24_oracle_audit import _canonical_json_bytes
        base = load_fixture()["snapshot"]
        return _canonical_json_bytes({**base, **overrides})

    def _assert_rejected(self, data: bytes, pattern: str):
        from federated_mcts.evaluation.game24_oracle_audit import deserialize_audit_input
        with self.assertRaises(ValueError, msg=f"expected rejection matching {pattern!r}") as ctx:
            deserialize_audit_input(data)
        self.assertIn(pattern, str(ctx.exception).lower())

    def test_evaluator_score_bool_rejected(self):
        b = self._snapshot_bytes(candidates=[{
            "idx": 0, "raw_text": "x", "canonical_numbers": ["24"],
            "evaluator_score": True, "evaluator_mode": None,
            "is_selected": True, "is_terminal": False,
        }], selected_indices=[0])
        self._assert_rejected(b, "not bool")

    def test_evaluator_score_nan_rejected_in_json(self):
        raw = (b'{"candidates":[{"idx":0,"raw_text":"x","canonical_numbers":["24"],'
               b'"evaluator_score":NaN,"evaluator_mode":null,'
               b'"is_selected":true,"is_terminal":false}],'
               b'"selected_indices":[0],"schema_version":1,"task_id":"t",'
               b'"task_input":"x","depth":0,"prompt_call_identity":"p",'
               b'"is_stopping":false,"decision_policy":"local_validity"}')
        self._assert_rejected(raw, "nan")

    def test_evaluator_score_inf_rejected_in_json(self):
        raw = (b'{"candidates":[{"idx":0,"raw_text":"x","canonical_numbers":["24"],'
               b'"evaluator_score":Infinity,"evaluator_mode":null,'
               b'"is_selected":true,"is_terminal":false}],'
               b'"selected_indices":[0],"schema_version":1,"task_id":"t",'
               b'"task_input":"x","depth":0,"prompt_call_identity":"p",'
               b'"is_stopping":false,"decision_policy":"local_validity"}')
        self._assert_rejected(raw, "infinity")

    def test_evaluator_score_negative_inf_rejected(self):
        raw = (b'{"candidates":[{"idx":0,"raw_text":"x","canonical_numbers":["24"],'
               b'"evaluator_score":-Infinity,"evaluator_mode":null,'
               b'"is_selected":true,"is_terminal":false}],'
               b'"selected_indices":[0],"schema_version":1,"task_id":"t",'
               b'"task_input":"x","depth":0,"prompt_call_identity":"p",'
               b'"is_stopping":false,"decision_policy":"local_validity"}')
        self._assert_rejected(raw, "infinity")

    def test_allow_nan_false_serialization_rejects_nan(self):
        import json as _json
        with self.assertRaises(ValueError):
            _json.dumps({"x": float("nan")}, allow_nan=False)

    def test_duplicate_candidate_idx_rejected(self):
        b = self._snapshot_bytes(candidates=[
            {"idx": 0, "raw_text": "a", "canonical_numbers": ["24"],
             "evaluator_score": 1.0, "evaluator_mode": "value",
             "is_selected": False, "is_terminal": False},
            {"idx": 0, "raw_text": "b", "canonical_numbers": ["1","1","2"],
             "evaluator_score": 1.0, "evaluator_mode": "value",
             "is_selected": False, "is_terminal": False},
        ], selected_indices=[])
        self._assert_rejected(b, "duplicate")

    def test_is_selected_inconsistent_with_selected_indices(self):
        b = self._snapshot_bytes(candidates=[{
            "idx": 0, "raw_text": "x", "canonical_numbers": ["24"],
            "evaluator_score": 1.0, "evaluator_mode": "value",
            "is_selected": True, "is_terminal": False,
        }], selected_indices=[])
        self._assert_rejected(b, "inconsistent")

    def test_selected_indices_refers_unknown_candidate(self):
        b = self._snapshot_bytes(candidates=[{
            "idx": 0, "raw_text": "x", "canonical_numbers": ["24"],
            "evaluator_score": 1.0, "evaluator_mode": "value",
            "is_selected": False, "is_terminal": False,
        }], selected_indices=[99])
        self._assert_rejected(b, "unknown")

    def test_bad_evaluator_mode_rejected(self):
        b = self._snapshot_bytes(candidates=[{
            "idx": 0, "raw_text": "x", "canonical_numbers": ["24"],
            "evaluator_score": 1.0, "evaluator_mode": "invalid_mode",
            "is_selected": False, "is_terminal": False,
        }], selected_indices=[])
        self._assert_rejected(b, "evaluator_mode")

    def test_idx_is_bool_rejected(self):
        b = self._snapshot_bytes(candidates=[{
            "idx": True, "raw_text": "x", "canonical_numbers": ["24"],
            "evaluator_score": 1.0, "evaluator_mode": "value",
            "is_selected": False, "is_terminal": False,
        }], selected_indices=[])
        self._assert_rejected(b, "idx not int")

    def test_is_terminal_not_bool_rejected(self):
        b = self._snapshot_bytes(candidates=[{
            "idx": 0, "raw_text": "x", "canonical_numbers": ["24"],
            "evaluator_score": 1.0, "evaluator_mode": "value",
            "is_selected": False, "is_terminal": 1,
        }], selected_indices=[])
        self._assert_rejected(b, "is_terminal must be bool")

    def test_canonical_numbers_non_string_rejected(self):
        b = self._snapshot_bytes(candidates=[{
            "idx": 0, "raw_text": "x", "canonical_numbers": [1, 2, 3],
            "evaluator_score": 1.0, "evaluator_mode": "value",
            "is_selected": False, "is_terminal": False,
        }], selected_indices=[])
        self._assert_rejected(b, "must be non-empty str")

    def test_fixture_roundtrip_survives_all_checks(self):
        from federated_mcts.evaluation.game24_oracle_audit import (
            _canonical_json_bytes, deserialize_audit_input,
        )
        snap = deserialize_audit_input(_canonical_json_bytes(load_fixture()["snapshot"]))
        self.assertEqual(len(snap.candidates), 3)
        self.assertEqual(snap.selected_indices, (1,))


if __name__ == "__main__":
    unittest.main()
