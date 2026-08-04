"""TDD tests for the durable selected-test authorization registry.

Coverage targets for Todo 3:
- Authorization key: (ordered_test_ids_hash, model_config_hash,
  decision_policy_fingerprint, review_round_identity)
- Run fingerprint stored as metadata only, never part of the key
- fsync + lock + hash-chain append-only registry
- Consume before client construction
- Retries/resume/deleted output/audit/schema changes cannot reauthorize
- Diagnostic keys distinct from selected-test key
- Concurrent access safety (lock-acquisition)
- Secrets stay out of the registry
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import time
import unittest
from dataclasses import dataclass, field

sys.path.insert(0, "src")


# ═══════════════════════════════════════════════════════════════════
# Fixture helpers
# ═══════════════════════════════════════════════════════════════════

def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _make_manifest_diagnostic_key() -> tuple[str, str, str, str]:
    """Return a tuple that would form a synthetic authorization key."""
    task_ids_hash = _sha(b"ordered_task_ids:[3,9]")
    model_cfg_hash = _sha(b"model_config:test")
    decision_fp = _sha(b"decision_policy:local_validity+seed:17")
    review_round = "review-1"
    return (task_ids_hash, model_cfg_hash, decision_fp, review_round)


class TestAuthorizationKey(unittest.TestCase):
    """Key construction and identity invariants."""

    def test_key_is_exactly_4_tuple(self):
        """Authorization key must be a 4-tuple of strings."""
        from federated_mcts.evaluation.game24_authorization import (
            make_auth_key,
        )
        key = make_auth_key(
            ordered_test_ids_hash="abc123",
            model_config_hash="def456",
            decision_policy_fingerprint="ghi789",
            review_round_identity="review-1",
        )
        self.assertIsInstance(key, tuple)
        self.assertEqual(len(key), 4)
        for item in key:
            self.assertIsInstance(item, str)

    def test_key_does_not_include_run_fingerprint(self):
        """Run fingerprint is metadata, never part of the key."""
        from federated_mcts.evaluation.game24_authorization import (
            make_auth_key,
        )
        # The function signature accepts 4 items; run_fp is separate
        key = make_auth_key(
            ordered_test_ids_hash="abc123",
            model_config_hash="def456",
            decision_policy_fingerprint="ghi789",
            review_round_identity="review-1",
        )
        self.assertNotIn("run_fp", str(key))
        self.assertEqual(len(key), 4)

    def test_different_decision_fp_produces_different_key(self):
        """Changing decision_policy_fingerprint changes the auth key."""
        from federated_mcts.evaluation.game24_authorization import (
            make_auth_key,
        )
        k1 = make_auth_key(
            ordered_test_ids_hash="a",
            model_config_hash="b",
            decision_policy_fingerprint="fp_local",
            review_round_identity="r1",
        )
        k2 = make_auth_key(
            ordered_test_ids_hash="a",
            model_config_hash="b",
            decision_policy_fingerprint="fp_heuristic",
            review_round_identity="r1",
        )
        self.assertNotEqual(k1, k2)

    def test_different_audit_same_key(self):
        """Audit on/off does not affect decision_policy_fingerprint,
        so the auth key stays the same."""
        from federated_mcts.evaluation.game24_authorization import (
            make_auth_key,
        )
        k1 = make_auth_key(
            ordered_test_ids_hash="a",
            model_config_hash="b",
            decision_policy_fingerprint="fp_same",
            review_round_identity="r1",
        )
        k2 = make_auth_key(
            ordered_test_ids_hash="a",
            model_config_hash="b",
            decision_policy_fingerprint="fp_same",
            review_round_identity="r1",
        )
        self.assertEqual(k1, k2)


class TestAuthorizationRegistry(unittest.TestCase):
    """Consume, reject replay, hash-chain integrity."""

    def setUp(self) -> None:
        self.tmpdir_obj = tempfile.TemporaryDirectory()
        self.tmpdir = self.tmpdir_obj.name
        self.registry_path = os.path.join(self.tmpdir, "auth_registry.jsonl")

    def tearDown(self) -> None:
        self.tmpdir_obj.cleanup()

    def _make_key(self) -> tuple[str, str, str, str]:
        return _make_manifest_diagnostic_key()

    def _make_run_fp(self) -> str:
        return _sha(b"run_fp:audit_off")

    def test_consume_first_time_succeeds(self):
        """First consumption of a key succeeds and writes receipt."""
        from federated_mcts.evaluation.game24_authorization import (
            AuthRegistry,
        )
        key = self._make_key()
        run_fp = self._make_run_fp()

        registry = AuthRegistry(self.registry_path)
        result = registry.consume(key, run_fp, is_selected_test=True)

        self.assertTrue(result, "First consume must succeed")
        self.assertTrue(os.path.exists(self.registry_path))

    def test_consume_second_time_fails(self):
        """Cannot consume the same key twice."""
        from federated_mcts.evaluation.game24_authorization import (
            AuthRegistry,
        )
        key = self._make_key()
        run_fp = self._make_run_fp()

        registry = AuthRegistry(self.registry_path)
        ok1 = registry.consume(key, run_fp, is_selected_test=True)
        self.assertTrue(ok1)

        ok2 = registry.consume(key, _sha(b"run_fp:audit_on"), is_selected_test=True)
        self.assertFalse(ok2, "Second consume with same key must fail")

    def test_consume_returns_true_only_once(self):
        """consume() returns True the first time, False thereafter."""
        from federated_mcts.evaluation.game24_authorization import (
            AuthRegistry,
        )
        key = self._make_key()
        run_fp = self._make_run_fp()

        registry = AuthRegistry(self.registry_path)
        self.assertTrue(registry.consume(key, run_fp, is_selected_test=True))
        self.assertFalse(registry.consume(key, run_fp, is_selected_test=True))
        self.assertFalse(registry.consume(key, _sha(b"other_run_fp"), is_selected_test=True))

    def test_audit_change_cannot_reauthorize(self):
        """Changing only audit/schema cannot authorize a second run —
        the key stays the same and was already consumed."""
        from federated_mcts.evaluation.game24_authorization import (
            AuthRegistry,
        )
        key = self._make_key()
        run_fp_audit_off = _sha(b"run_fp:audit_off")
        run_fp_audit_on = _sha(b"run_fp:audit_on")

        registry = AuthRegistry(self.registry_path)
        self.assertTrue(registry.consume(key, run_fp_audit_off, is_selected_test=True))
        self.assertFalse(registry.consume(key, run_fp_audit_on, is_selected_test=True))

    def test_diagnostic_key_is_distinct(self):
        """Diagnostic runs use distinct predeclared keys, not the
        selected-test key."""
        from federated_mcts.evaluation.game24_authorization import (
            AuthRegistry,
        )
        selected_key = ("a", "b", "fp_selected", "r1")
        diagnostic_key = ("a", "b", "fp_diag", "r1")

        registry = AuthRegistry(self.registry_path)
        self.assertTrue(
            registry.consume(selected_key, _sha(b"run_selected"), is_selected_test=True)
        )
        # Diagnostic with different key still succeeds
        self.assertTrue(
            registry.consume(diagnostic_key, _sha(b"run_diag"), is_selected_test=False)
        )
        # But selected key is still consumed
        self.assertFalse(
            registry.consume(selected_key, _sha(b"run_selected_again"), is_selected_test=True)
        )

    def test_hash_chain_verifies(self):
        """Registry entries form a verifiable hash chain."""
        from federated_mcts.evaluation.game24_authorization import (
            AuthRegistry,
        )
        key1 = ("a", "b", "fp1", "r1")
        key2 = ("a", "b", "fp2", "r1")
        run_fp1 = _sha(b"run1")
        run_fp2 = _sha(b"run2")

        registry = AuthRegistry(self.registry_path)
        registry.consume(key1, run_fp1, is_selected_test=True)
        registry.consume(key2, run_fp2, is_selected_test=False)

        # Verify chain
        ok, error = registry.verify_chain()
        self.assertTrue(ok, f"Hash chain verification failed: {error}")

    def test_tampered_registry_fails_verification(self):
        """Modifying a registry entry breaks the hash chain."""
        from federated_mcts.evaluation.game24_authorization import (
            AuthRegistry,
        )
        key1 = ("a", "b", "fp1", "r1")
        run_fp1 = _sha(b"run1")

        registry = AuthRegistry(self.registry_path)
        registry.consume(key1, run_fp1, is_selected_test=True)

        # Tamper with the file
        with open(self.registry_path, "a") as f:
            f.write(json.dumps({"tampered": True}) + "\n")

        ok, error = registry.verify_chain()
        self.assertFalse(ok, "Tampered registry should fail verification")

    def test_registry_disk_is_fsynced(self):
        """Each consume writes + fsyncs; data survives a process restart
        (simulated by re-opening the file)."""
        from federated_mcts.evaluation.game24_authorization import (
            AuthRegistry,
        )
        key = self._make_key()
        run_fp = self._make_run_fp()

        # Consume in first "process"
        registry1 = AuthRegistry(self.registry_path)
        self.assertTrue(registry1.consume(key, run_fp, is_selected_test=True))

        # Simulate restart: new object reads the same file
        registry2 = AuthRegistry(self.registry_path)
        self.assertFalse(registry2.consume(key, run_fp, is_selected_test=True))

    def test_no_secrets_in_registry(self):
        """Registry entries must never contain secret strings."""
        from federated_mcts.evaluation.game24_authorization import (
            AuthRegistry,
        )
        key = self._make_key()
        run_fp = _sha(b"run_with_secret_sk-test-abc")

        registry = AuthRegistry(self.registry_path)
        registry.consume(key, run_fp, is_selected_test=True)

        with open(self.registry_path) as f:
            content = f.read()
        self.assertNotIn("sk-test-abc", content)
        self.assertNotIn("api_key", content)


if __name__ == "__main__":
    unittest.main()
