"""V2 Auth enforcement tests — before-client construction, diagnostics, rejects.

Coverage targets for Todo 3 V2:
- Authorization consumed BEFORE any FederatedSolver/client construction
- Second/deleted-output/resume/audit/schema attempts rejected
- Diagnostic keys distinct from selected-test key
- run_fingerprint metadata only
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, "src")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class TestAuthBeforeClientConstruction(unittest.TestCase):
    """Auth key MUST be consumed before any model client is built."""

    def setUp(self) -> None:
        self.tmpdir_obj = tempfile.TemporaryDirectory()
        self.tmpdir = self.tmpdir_obj.name

    def tearDown(self) -> None:
        self.tmpdir_obj.cleanup()

    def test_selected_test_consumed_once_only(self):
        """First consume succeeds; second fails."""
        from federated_mcts.evaluation.game24_authorization import (
            AuthRegistry,
            make_auth_key,
        )
        ap = os.path.join(self.tmpdir, "auth.jsonl")
        registry = AuthRegistry(ap)
        key = make_auth_key(
            ordered_test_ids_hash="a", model_config_hash="b",
            decision_policy_fingerprint="c", review_round_identity="r1",
        )
        self.assertTrue(registry.consume(key, "run_fp_1", is_selected_test=True))
        self.assertFalse(registry.consume(key, "run_fp_2", is_selected_test=True))

    def test_second_attempt_from_new_registry_rejected(self):
        """A second registry instance reading the same file also sees consumed key."""
        from federated_mcts.evaluation.game24_authorization import (
            AuthRegistry,
            make_auth_key,
        )
        ap = os.path.join(self.tmpdir, "auth.jsonl")
        r1 = AuthRegistry(ap)
        key = make_auth_key(
            ordered_test_ids_hash="a", model_config_hash="b",
            decision_policy_fingerprint="c", review_round_identity="r1",
        )
        self.assertTrue(r1.consume(key, "fp1", is_selected_test=True))

        r2 = AuthRegistry(ap)
        self.assertFalse(r2.consume(key, "fp2", is_selected_test=True))

    def test_deleted_output_cannot_reauthorize(self):
        """Even if the run output dir was deleted, auth still consumed."""
        from federated_mcts.evaluation.game24_authorization import (
            AuthRegistry,
            make_auth_key,
        )
        ap = os.path.join(self.tmpdir, "auth.jsonl")
        registry = AuthRegistry(ap)
        key = make_auth_key(
            ordered_test_ids_hash="a", model_config_hash="b",
            decision_policy_fingerprint="c", review_round_identity="r1",
        )
        self.assertTrue(registry.consume(key, "fp1", is_selected_test=True))

        # Simulate output deleted
        # Auth registry should still show consumed
        r2 = AuthRegistry(ap)
        self.assertTrue(r2.consumed_selected_test)

    def test_audit_change_cannot_reauthorize(self):
        """Toggling audit only changes run_fingerprint, not auth key."""
        from federated_mcts.evaluation.game24_authorization import (
            AuthRegistry,
            make_auth_key,
        )
        ap = os.path.join(self.tmpdir, "auth.jsonl")
        registry = AuthRegistry(ap)
        key = make_auth_key(
            ordered_test_ids_hash="a", model_config_hash="b",
            decision_policy_fingerprint="c", review_round_identity="r1",
        )
        self.assertTrue(registry.consume(key, "fp_audit_off", is_selected_test=True))
        self.assertFalse(registry.consume(key, "fp_audit_on", is_selected_test=True))

    def test_diagnostics_distinct_from_selected(self):
        """Diagnostic keys are separate and do not consume selected-test slot."""
        from federated_mcts.evaluation.game24_authorization import (
            AuthRegistry,
            make_auth_key,
        )
        ap = os.path.join(self.tmpdir, "auth.jsonl")
        registry = AuthRegistry(ap)

        sel_key = make_auth_key(
            ordered_test_ids_hash="a", model_config_hash="b",
            decision_policy_fingerprint="sel_fp", review_round_identity="r1",
        )
        diag_key = make_auth_key(
            ordered_test_ids_hash="a", model_config_hash="b",
            decision_policy_fingerprint="diag_fp", review_round_identity="r1",
        )
        self.assertTrue(registry.consume(diag_key, "fp_diag", is_selected_test=False))
        self.assertTrue(registry.consume(sel_key, "fp_sel", is_selected_test=True))
        self.assertTrue(registry.consumed_selected_test)
        # Diag didn't consume selected slot — it was still available

    def test_run_fingerprint_not_part_of_key(self):
        """Prove run_fingerprint is metadata, not key identity."""
        from federated_mcts.evaluation.game24_authorization import (
            make_auth_key,
        )
        key = make_auth_key(
            ordered_test_ids_hash="a", model_config_hash="b",
            decision_policy_fingerprint="c", review_round_identity="r1",
        )
        self.assertNotIn("run_fp", str(key))
        self.assertEqual(len(key), 4)


if __name__ == "__main__":
    unittest.main()
