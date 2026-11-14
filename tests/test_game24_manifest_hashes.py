"""SHA-256 tests for split files, task IDs, and model config (secrets stripped).

Extracted from test_game24_manifest.py to keep ≤250 pure LOC per file.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, "src")


# ── Shared fixtures ────────────────────────────────────────────────

def _make_model_config(api_key_env: str | None = None) -> list[dict]:
    cfg = [
        {"client_name": "local_client", "model": "TestModel-1B", "type": "mock",
         "api_base": "http://localhost:9999/v1", "temperature": 0.3, "max_tokens": 256},
        {"client_name": "remote_client", "model": "TestModel-7B",
         "api_base": "https://api.example.com/v1", "temperature": 0.7, "max_tokens": 512},
    ]
    if api_key_env:
        cfg[1]["api_key_environment_variable"] = api_key_env
        os.environ[api_key_env] = "sk-test-secret-12345"
    return cfg


def _make_split_file(tmpdir: str, splits: dict | None = None) -> str:
    if splits is None:
        splits = {"train": [0, 1, 2], "val": [3, 4], "test": [5, 6, 7]}
    path = os.path.join(tmpdir, "game24.json")
    with open(path, "w") as f:
        json.dump(splits, f)
    return path


class TestManifestSplitHashes(unittest.TestCase):
    def setUp(self):
        self.tmpdir_obj = tempfile.TemporaryDirectory()
        self.tmpdir = self.tmpdir_obj.name
    def tearDown(self):
        self.tmpdir_obj.cleanup()

    def test_split_file_sha256_matches_known(self):
        from federated_mcts.evaluation.game24_manifest import Game24Manifest
        split_path = _make_split_file(self.tmpdir)
        cfg = _make_model_config()
        manifest = Game24Manifest(
            task="game24", task_ids=[5, 6, 7], task_split_name="test",
            split_file_path=split_path, model_config_raw=cfg,
            decision_policy="local_validity", oracle_audit=False,
            seed=17, temperature=0.9,
            method_generate="propose", method_evaluate="vote", method_select="greedy",
            n_generate_sample=2, n_evaluate_sample=3, n_select_sample=2,
            game24_exact_prune=False, runner_path="scripts/merged_run.py",
        )
        expected_sha = hashlib.sha256(
            json.dumps({"train": [0, 1, 2], "val": [3, 4], "test": [5, 6, 7]},
                       sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.assertEqual(manifest.split_file_sha256, expected_sha)

    def test_task_ids_sha256_is_stable(self):
        from federated_mcts.evaluation.game24_manifest import Game24Manifest
        split_path = _make_split_file(self.tmpdir)
        cfg = _make_model_config()
        mk = lambda ids: Game24Manifest(
            task="game24", task_ids=ids, task_split_name="test",
            split_file_path=split_path, model_config_raw=cfg,
            decision_policy="local_validity", oracle_audit=False,
            seed=17, temperature=0.9,
            method_generate="propose", method_evaluate="vote", method_select="greedy",
            n_generate_sample=2, n_evaluate_sample=3, n_select_sample=2,
            game24_exact_prune=False, runner_path="scripts/merged_run.py",
        )
        m1, m2 = mk([1, 2, 3]), mk([1, 2, 3])
        self.assertEqual(m1.ordered_task_ids_sha256, m2.ordered_task_ids_sha256)
        self.assertNotEqual(m1.ordered_task_ids_sha256, mk([3, 2, 1]).ordered_task_ids_sha256)


class TestManifestModelConfigHash(unittest.TestCase):
    def setUp(self):
        self.tmpdir_obj = tempfile.TemporaryDirectory()
        self.tmpdir = self.tmpdir_obj.name
    def tearDown(self):
        self.tmpdir_obj.cleanup()

    def test_model_config_hash_excludes_api_keys(self):
        from federated_mcts.evaluation.game24_manifest import Game24Manifest
        env_key = "TEST_HASH_NOLEAK2"
        os.environ[env_key] = "sk-leaked-key-999"
        cfg_with = _make_model_config(api_key_env=env_key)
        cfg_without = [
            {"client_name": "local_client", "model": "TestModel-1B", "type": "mock",
             "api_base": "http://localhost:9999/v1", "temperature": 0.3, "max_tokens": 256},
            {"client_name": "remote_client", "model": "TestModel-7B",
             "api_base": "https://api.example.com/v1", "temperature": 0.7, "max_tokens": 512},
        ]
        split_path = _make_split_file(self.tmpdir)
        mk = lambda cfg: Game24Manifest(
            task="game24", task_ids=[3, 9], task_split_name="test",
            split_file_path=split_path, model_config_raw=cfg,
            decision_policy="local_validity", oracle_audit=False,
            seed=17, temperature=0.9,
            method_generate="propose", method_evaluate="vote", method_select="greedy",
            n_generate_sample=2, n_evaluate_sample=3, n_select_sample=2,
            game24_exact_prune=False, runner_path="scripts/merged_run.py",
        )
        self.assertEqual(mk(cfg_with).model_config_sha256, mk(cfg_without).model_config_sha256,
                         "DEFECT: model_config_sha256 leaked API key/env-var info")
        os.environ.pop(env_key, None)


if __name__ == "__main__":
    unittest.main()
