"""TDD tests for seed handling, task-list resolution, and code revision markers.

Coverage for Todo 3:
- Seed → deterministic Python/NumPy state
- Task IDs resolved and ordered before execution
- Code commit hash and dirty flag in manifest
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

sys.path.insert(0, "src")


# ── Shared fixtures (copied from test_game24_manifest.py) ──────────

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
    import json
    path = os.path.join(tmpdir, f"{name}.json")
    with open(path, "w") as f:
        json.dump(splits, f)
    return path


class TestSeedHandling(unittest.TestCase):
    """Seed application to Python random and NumPy."""

    def setUp(self) -> None:
        self.tmpdir_obj = tempfile.TemporaryDirectory()
        self.tmpdir = self.tmpdir_obj.name

    def tearDown(self) -> None:
        self.tmpdir_obj.cleanup()

    def test_seed_applied_before_solver(self):
        import random
        import numpy as np
        from federated_mcts.evaluation.game24_manifest import apply_seed

        apply_seed(42)
        random_seq1 = [random.random() for _ in range(5)]
        np_seq1 = list(np.random.rand(5))

        apply_seed(42)
        random_seq2 = [random.random() for _ in range(5)]
        np_seq2 = list(np.random.rand(5))

        self.assertEqual(random_seq1, random_seq2,
                         "Python random not deterministic after same seed")
        for a, b in zip(np_seq1, np_seq2):
            self.assertAlmostEqual(a, b, places=12)

    def test_seed_records_nondeterminism_notice(self):
        from federated_mcts.evaluation.game24_manifest import Game24Manifest
        cfg = _make_model_config()
        split_path = _make_split_file(self.tmpdir)

        manifest = Game24Manifest(
            task="game24", task_ids=[3, 9], task_split_name="test",
            split_file_path=split_path, model_config_raw=cfg,
            decision_policy="local_validity", oracle_audit=False,
            seed=17, temperature=0.9,
            method_generate="propose", method_evaluate="vote", method_select="greedy",
            n_generate_sample=2, n_evaluate_sample=3, n_select_sample=2,
            game24_exact_prune=False, runner_path="scripts/merged_run.py",
        )
        self.assertTrue(manifest.remote_nondeterminism_notice)

    def test_different_seeds_produce_different_sequences(self):
        import random
        from federated_mcts.evaluation.game24_manifest import apply_seed

        apply_seed(17)
        r17 = random.random()
        apply_seed(42)
        r42 = random.random()
        self.assertNotEqual(r17, r42)


class TestTaskListResolution(unittest.TestCase):
    """Task IDs resolved from split/range before execution."""

    def setUp(self) -> None:
        self.tmpdir_obj = tempfile.TemporaryDirectory()
        self.tmpdir = self.tmpdir_obj.name

    def tearDown(self) -> None:
        self.tmpdir_obj.cleanup()

    def test_task_split_overrides_range(self):
        from federated_mcts.evaluation.game24_manifest import resolve_task_ids
        split_path = _make_split_file(
            self.tmpdir,
            splits={"val": [10, 20, 30], "test": [5, 6, 7]},
        )
        ids = resolve_task_ids(
            task="game24", split_file_path=split_path,
            task_split="val", task_start_index=0, task_end_index=100,
        )
        self.assertEqual(ids, [10, 20, 30])

    def test_no_split_uses_range(self):
        from federated_mcts.evaluation.game24_manifest import resolve_task_ids
        split_path = _make_split_file(self.tmpdir)
        ids = resolve_task_ids(
            task="game24", split_file_path=split_path,
            task_split=None, task_start_index=5, task_end_index=8,
        )
        self.assertEqual(ids, [5, 6, 7])

    def test_task_ids_are_ordered(self):
        from federated_mcts.evaluation.game24_manifest import resolve_task_ids
        split_path = _make_split_file(self.tmpdir, splits={"val": [30, 10, 20]})
        ids = resolve_task_ids(
            task="game24", split_file_path=split_path,
            task_split="val", task_start_index=0, task_end_index=100,
        )
        self.assertEqual(ids, [10, 20, 30])


class TestManifestCodeRevision(unittest.TestCase):
    """Code commit hash and dirty marker in manifest."""

    def setUp(self) -> None:
        self.tmpdir_obj = tempfile.TemporaryDirectory()
        self.tmpdir = self.tmpdir_obj.name

    def tearDown(self) -> None:
        self.tmpdir_obj.cleanup()

    def test_dirty_marker_is_present(self):
        from federated_mcts.evaluation.game24_manifest import Game24Manifest
        cfg = _make_model_config()
        split_path = _make_split_file(self.tmpdir)
        manifest = Game24Manifest(
            task="game24", task_ids=[3, 9], task_split_name="test",
            split_file_path=split_path, model_config_raw=cfg,
            decision_policy="local_validity", oracle_audit=False,
            seed=17, temperature=0.9,
            method_generate="propose", method_evaluate="vote", method_select="greedy",
            n_generate_sample=2, n_evaluate_sample=3, n_select_sample=2,
            game24_exact_prune=False, runner_path="scripts/merged_run.py",
        )
        self.assertIsInstance(manifest.git_commit, str)
        self.assertIsInstance(manifest.worktree_dirty, bool)
        self.assertTrue(len(manifest.git_commit) >= 7)
        int(manifest.git_commit[:7], 16)


if __name__ == "__main__":
    unittest.main()
