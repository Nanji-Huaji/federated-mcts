"""Regression test: distinct task sets must never share an output prefix.

PRODUCTION DEFECT (observed, destroyed real evidence):

``compute_file_prefix`` encodes ``task_start_index`` / ``task_end_index`` in the
output filename, but those fields are NOT derived from the runs' actual task IDs.
A ``--pilot_ids`` run and a full ``--task_split val`` run therefore produced the
**byte-identical prefix**:

    0.9_propose_n_generate_sample_2_..._start900_end1000_..._idx_0

Consequence, measured on 2026-08-13: launching the 200-ID full validation
silently overwrote the completed 20-ID pilot's artifacts in place — its
``manifest.json`` (20 IDs replaced by 200), its results ``.json`` (destroyed), and
its ``journal.json`` ``run_fingerprint`` (rewritten to the full run's value while
the surviving ``performance.json`` still carried the pilot's). The B0 evidence set
became internally inconsistent and had to be regenerated, costing a full pilot
re-run.

Expected contract: two runs whose resolved task-ID sets differ in size must
produce different prefixes, so that starting one can never clobber the other's
evidence.
"""

from __future__ import annotations

import sys
import unittest

sys.path.insert(0, "src")

from federated_mcts.evaluation.game24_runner_persistence import compute_file_prefix


class _Args:
    """Plain args holder.

    A real class (not MagicMock) so that an attribute genuinely absent — notably
    ``model_config`` — is absent to ``hasattr`` too, matching real argparse
    behaviour.
    """

    def __init__(self, **overrides: object) -> None:
        base: dict[str, object] = {
            "naive_run": False,
            "task": "game24",
            "solve_method": "tot",
            "remotebackend": "qwen3.7-plus",
            "temperature": 0.9,
            "method_generate": "propose",
            "n_generate_sample": 2,
            "method_evaluate": "vote",
            "n_evaluate_sample": 3,
            "method_select": "greedy",
            "n_select_sample": 2,
            "search_policy": "baseline",
            "task_start_index": 900,
            "task_end_index": 1000,
            "slm_generate": False,
            "slm_eval": False,
            "check_format": True,
            "eval_rule": False,
            "warm_start": False,
            "last_lm": False,
            "inference_idx": 0,
            "game24_exact_prune": False,
        }
        base.update(overrides)
        for k, v in base.items():
            setattr(self, k, v)


def _args(**overrides: object) -> _Args:
    """Args carrying the same CLI decision fields for both runs.

    Only the resolved task-ID count differs, which is exactly the situation that
    collided: identical CLI, different task set.
    """
    return _Args(**overrides)


class TestFilePrefixTaskSetDistinctness(unittest.TestCase):
    """Given two runs with different task sets, Then prefixes must differ."""

    def test_pilot_and_full_validation_prefixes_differ(self) -> None:
        """A 20-ID pilot and its 200-ID scale-up must not share a prefix."""
        pilot = compute_file_prefix(_args(), task_count=20)
        full = compute_file_prefix(_args(), task_count=200)

        self.assertNotEqual(
            pilot, full,
            "DEFECT: the 20-ID pilot and the 200-ID full validation computed the "
            "SAME output prefix, so starting the full run overwrote the pilot's "
            "manifest, results, and journal fingerprints in place.",
        )

    def test_task_id_count_is_encoded_in_prefix(self) -> None:
        """The resolved task-ID count must be visible in the filename."""
        self.assertIn("ntasks20", compute_file_prefix(_args(), task_count=20))
        self.assertIn("ntasks200", compute_file_prefix(_args(), task_count=200))

    def test_differing_pilot_sizes_do_not_collide(self) -> None:
        """Even two different pilot sizes must be distinguishable."""
        a = compute_file_prefix(_args(), task_count=20)
        b = compute_file_prefix(_args(), task_count=21)
        self.assertNotEqual(a, b)

    def test_same_task_set_is_stable(self) -> None:
        """Recomputing for the same task set must be deterministic."""
        self.assertEqual(
            compute_file_prefix(_args(), task_count=4),
            compute_file_prefix(_args(), task_count=4),
        )

    def test_omitting_task_count_keeps_legacy_prefix_shape(self) -> None:
        """Legacy callers that pass no count keep the previous naming."""
        prefix = compute_file_prefix(_args())
        self.assertNotIn("ntasks", prefix)


class TestRunnerNamingUsesResolvedCount(unittest.TestCase):
    """Given the canonical prepare seam, Then the resolved IDs size the prefix.

    Encoding a count is only useful if the runner feeds it the *resolved* task
    IDs rather than the CLI's start/end range.
    """

    def setUp(self) -> None:
        import json
        import os
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.tmpdir = self._tmp.name
        self._cwd = os.getcwd()
        os.chdir(self.tmpdir)
        os.makedirs("configs")
        os.makedirs("data/splits")
        with open("configs/model_config.json", "w", encoding="utf-8") as stream:
            json.dump(
                [
                    {
                        "client_name": "local_client",
                        "model": "Qwen/Qwen2.5-32B-Instruct",
                        "max_tokens": 256,
                        "api_key": "k",
                        "api_base": "http://127.0.0.1:9/v1",
                    },
                    {
                        "client_name": "remote_client",
                        "model": "deepseek-v4-flash",
                        "max_tokens": 512,
                        "api_key": "k",
                        "api_base": "http://127.0.0.1:9/v1",
                    },
                ],
                stream,
            )
        with open("data/splits/game24.json", "w", encoding="utf-8") as stream:
            json.dump({"train": list(range(1000, 1200)),
                       "val": list(range(200)),
                       "test": list(range(2000, 2200))}, stream)

    def tearDown(self) -> None:
        import os

        os.chdir(self._cwd)
        self._tmp.cleanup()

    def _prepare(self, extra: list[str]):
        from unittest.mock import patch

        from federated_mcts.evaluation.game24_runner_args import parse_args
        from federated_mcts.experiments import runner

        args = parse_args(
            [
                "--task",
                "game24",
                "--model_config",
                "configs/model_config.json",
                *extra,
            ]
        )
        # Projection construction has its own tests; this test owns the
        # resolution → naming integration and must not depend on it.
        with patch.object(runner, "build_preflight", return_value=None):
            return runner.prepare(args, "tot")

    def test_pilot_prefix_encodes_20_resolved_ids(self) -> None:
        from federated_mcts.evaluation.game24_pilot import VALIDATION_PILOT_IDS

        plan = self._prepare(["--pilot_ids", *map(str, VALIDATION_PILOT_IDS)])
        self.assertIsNotNone(plan)
        self.assertEqual(len(plan.setup.task_ids), 20)
        self.assertIn("_ntasks20", plan.setup.file_name)

    def test_full_split_prefix_encodes_200_resolved_ids(self) -> None:
        plan = self._prepare(["--task_split", "val"])
        self.assertIsNotNone(plan)
        self.assertEqual(len(plan.setup.task_ids), 200)
        self.assertIn("_ntasks200", plan.setup.file_name)

    def test_pilot_and_full_validation_prefixes_differ_at_the_seam(self) -> None:
        from federated_mcts.evaluation.game24_pilot import VALIDATION_PILOT_IDS

        pilot = self._prepare(["--pilot_ids", *map(str, VALIDATION_PILOT_IDS)])
        full = self._prepare(["--task_split", "val"])
        self.assertNotEqual(pilot.setup.file_name, full.setup.file_name)


if __name__ == "__main__":
    unittest.main()
