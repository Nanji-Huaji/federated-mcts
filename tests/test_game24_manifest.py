"""TDD tests for immutable Game24 experiment manifests, dual fingerprints,
seed handling, and task-list resolution.

Coverage targets for Todo 3:
- Immutable manifest dataclass
- decision_policy_fingerprint (excludes audit/observation)
- run_fingerprint (includes audit + observation)
- Same inputs → same fingerprints
- Audit-only change preserves decision fingerprint, changes run
- Decision-affecting field change changes both
- Seed → deterministic Python/NumPy state; record nondeterminism notice
- Task IDs resolved and ordered before execution
- SHA-256 of split file and task IDs
- Model config SHA-256 (model names, never API keys)
- Secrets absent from manifest
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, "src")


# ── Fixture helpers ────────────────────────────────────────────────

def _make_model_config(api_key_env: str | None = None) -> list[dict]:
    cfg = [
        {
            "client_name": "local_client",
            "model": "TestModel-1B",
            "type": "mock",
            "api_base": "http://localhost:9999/v1",
            "temperature": 0.3,
            "max_tokens": 256,
        },
        {
            "client_name": "remote_client",
            "model": "TestModel-7B",
            "api_base": "https://api.example.com/v1",
            "temperature": 0.7,
            "max_tokens": 512,
        },
    ]
    if api_key_env:
        cfg[1]["api_key_environment_variable"] = api_key_env
        os.environ[api_key_env] = "sk-test-secret-12345"
    return cfg


def _make_split_file(
    tmpdir: str,
    name: str = "game24",
    splits: dict | None = None,
) -> str:
    if splits is None:
        splits = {"train": [0, 1, 2], "val": [3, 4], "test": [5, 6, 7]}
    path = os.path.join(tmpdir, f"{name}.json")
    with open(path, "w") as f:
        json.dump(splits, f)
    return path


class TestImmutableManifest(unittest.TestCase):
    """Manifest creation, immutability, and basic field coverage."""

    def setUp(self) -> None:
        self.tmpdir_obj = tempfile.TemporaryDirectory()
        self.tmpdir = self.tmpdir_obj.name

    def tearDown(self) -> None:
        self.tmpdir_obj.cleanup()

    def test_manifest_created_with_required_fields(self):
        """Manifest dataclass has all required fields, not smuggled via dict."""
        from federated_mcts.evaluation.game24_manifest import (
            Game24Manifest,
        )

        cfg = _make_model_config()
        split_path = _make_split_file(self.tmpdir)

        manifest = Game24Manifest(
            task="game24",
            task_ids=[3, 9],
            task_split_name="test",
            split_file_path=split_path,
            model_config_raw=cfg,
            decision_policy="local_validity",
            oracle_audit=False,
            seed=17,
            temperature=0.9,
            method_generate="propose",
            method_evaluate="vote",
            method_select="greedy",
            n_generate_sample=2,
            n_evaluate_sample=3,
            n_select_sample=2,
            game24_exact_prune=False,
            max_call_limit=None,
            max_token_limit=None,
            max_cost_usd=None,
            max_raw_unit_limit=19.99,
            runner_path="scripts/merged_run.py",
        )

        self.assertEqual(manifest.task, "game24")
        self.assertEqual(manifest.task_ids, [3, 9])
        self.assertEqual(manifest.decision_policy, "local_validity")
        self.assertFalse(manifest.oracle_audit)
        self.assertEqual(manifest.seed, 17)
        self.assertEqual(manifest.max_raw_unit_limit, 19.99)

    def test_manifest_is_frozen(self):
        """Manifest must be frozen — mutation raises FrozenInstanceError."""
        from federated_mcts.evaluation.game24_manifest import (
            Game24Manifest,
        )

        cfg = _make_model_config()
        split_path = _make_split_file(self.tmpdir)

        manifest = Game24Manifest(
            task="game24",
            task_ids=[3, 9],
            task_split_name="test",
            split_file_path=split_path,
            model_config_raw=cfg,
            decision_policy="local_validity",
            oracle_audit=False,
            seed=17,
            temperature=0.9,
            method_generate="propose",
            method_evaluate="vote",
            method_select="greedy",
            n_generate_sample=2,
            n_evaluate_sample=3,
            n_select_sample=2,
            game24_exact_prune=False,
            runner_path="scripts/merged_run.py",
        )

        # Python 3.10+ @dataclass(frozen=True) raises FrozenInstanceError
        from dataclasses import FrozenInstanceError

        with self.assertRaises(FrozenInstanceError):
            manifest.task_ids = [99]  # type: ignore[misc]

    def test_manifest_serializes_without_secrets(self):
        """JSON serialization must exclude API keys and env-var values."""
        from federated_mcts.evaluation.game24_manifest import Game24Manifest
        env_key = "TEST_MANIFEST_API_KEY"
        cfg = _make_model_config(api_key_env=env_key)
        split_path = _make_split_file(self.tmpdir)
        manifest = Game24Manifest(
            task="game24", task_ids=[3, 9], task_split_name="test",
            split_file_path=split_path, model_config_raw=cfg,
            decision_policy="local_validity", oracle_audit=False,
            seed=17, temperature=0.9,
            method_generate="propose", method_evaluate="vote", method_select="greedy",
            n_generate_sample=2, n_evaluate_sample=3, n_select_sample=2,
            game24_exact_prune=False, runner_path="scripts/merged_run.py")
        try:
            serialized = manifest.to_json()
            self.assertNotIn("sk-test-secret", serialized)
            self.assertNotIn(env_key, serialized)
            self.assertIn("TestModel-7B", serialized)
        finally:
            os.environ.pop(env_key, None)


class TestDualFingerprints(unittest.TestCase):
    """decision_policy_fingerprint vs run_fingerprint invariants."""

    def setUp(self) -> None:
        self.tmpdir_obj = tempfile.TemporaryDirectory()
        self.tmpdir = self.tmpdir_obj.name

    def tearDown(self) -> None:
        self.tmpdir_obj.cleanup()

    def _make_manifest(self, **overrides) -> "Game24Manifest":
        from federated_mcts.evaluation.game24_manifest import (
            Game24Manifest,
        )
        cfg = _make_model_config()
        split_path = _make_split_file(self.tmpdir)

        kwargs: dict = dict(
            task="game24",
            task_ids=[3, 9],
            task_split_name="test",
            split_file_path=split_path,
            model_config_raw=cfg,
            decision_policy="local_validity",
            oracle_audit=False,
            seed=17,
            temperature=0.9,
            method_generate="propose",
            method_evaluate="vote",
            method_select="greedy",
            n_generate_sample=2,
            n_evaluate_sample=3,
            n_select_sample=2,
            game24_exact_prune=False,
            runner_path="scripts/merged_run.py",
        )
        kwargs.update(overrides)
        return Game24Manifest(**kwargs)

    def test_identical_manifests_produce_identical_fingerprints(self):
        """Same inputs → same fingerprints (both kinds)."""
        m1 = self._make_manifest()
        m2 = self._make_manifest()

        self.assertEqual(
            m1.decision_policy_fingerprint,
            m2.decision_policy_fingerprint,
        )
        self.assertEqual(m1.run_fingerprint, m2.run_fingerprint)

    def test_decision_policy_fingerprint_excludes_audit(self):
        """audit off vs on: decision fingerprint identical, run different."""
        m_off = self._make_manifest(oracle_audit=False)
        m_on = self._make_manifest(oracle_audit=True)

        self.assertEqual(
            m_off.decision_policy_fingerprint,
            m_on.decision_policy_fingerprint,
            "DEFECT: decision_policy_fingerprint changed when only audit toggled",
        )
        self.assertNotEqual(
            m_off.run_fingerprint,
            m_on.run_fingerprint,
            "DEFECT: run_fingerprint should differ when audit toggled",
        )

    def test_decision_affecting_change_changes_both(self):
        """Changing decision_policy from local_validity → heuristic_rules
        must change both fingerprints."""
        m_local = self._make_manifest(decision_policy="local_validity")
        m_heur = self._make_manifest(decision_policy="heuristic_rules")

        self.assertNotEqual(
            m_local.decision_policy_fingerprint,
            m_heur.decision_policy_fingerprint,
        )
        self.assertNotEqual(
            m_local.run_fingerprint,
            m_heur.run_fingerprint,
        )

    def test_seed_change_changes_both(self):
        """Seed affects reproducibility → must change both fingerprints."""
        m17 = self._make_manifest(seed=17)
        m42 = self._make_manifest(seed=42)

        self.assertNotEqual(
            m17.decision_policy_fingerprint, m42.decision_policy_fingerprint
        )
        self.assertNotEqual(m17.run_fingerprint, m42.run_fingerprint)

    def test_fingerprint_is_stable_sha256_string(self):
        """Fingerprints are deterministic SHA-256 hex strings of fixed length."""
        m = self._make_manifest()
        d_fp = m.decision_policy_fingerprint
        r_fp = m.run_fingerprint
        self.assertIsInstance(d_fp, str)
        self.assertIsInstance(r_fp, str)
        self.assertEqual(len(d_fp), 64)
        self.assertEqual(len(r_fp), 64)
        int(d_fp, 16)
        int(r_fp, 16)


if __name__ == "__main__":
    unittest.main()
