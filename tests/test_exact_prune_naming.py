"""Naming contract for the canonical runner's output prefix.

The retired root ``merged_run.py`` and ``scripts/merged_run.py`` each carried a
copy of ``file_name_generater``.  Both are gone; the single implementation now
lives in ``federated_mcts.evaluation.game24_runner_persistence`` and is
exercised here through its public function.

The contract: ``--game24_exact_prune`` appends exactly one
``_exactprune_True`` marker, after any search-policy marker, and the resolved
task-count marker comes after it.
"""

from __future__ import annotations

import sys
import unittest

sys.path.insert(0, "src")

from federated_mcts.evaluation.game24_runner_persistence import (
    compute_file_prefix,
)


class _Args:
    """Attributes a real argparse namespace exposes to naming."""

    def __init__(self, **overrides: object) -> None:
        base: dict[str, object] = {
            "naive_run": False,
            "task": "game24",
            "solve_method": "tot",
            "localbackend": "local-model",
            "remotebackend": "remote-model",
            "temperature": 0.7,
            "method_generate": "sample",
            "n_generate_sample": 5,
            "method_evaluate": "vote",
            "n_evaluate_sample": 3,
            "method_select": "greedy",
            "n_select_sample": 1,
            "search_policy": "baseline",
            "task_start_index": 0,
            "task_end_index": 10,
            "slm_generate": False,
            "slm_eval": False,
            "check_format": False,
            "eval_rule": "standard",
            "warm_start": False,
            "last_lm": "none",
            "inference_idx": 0,
            "prompt_sample": "standard",
        }
        base.update(overrides)
        for key, value in base.items():
            setattr(self, key, value)


def _prefix(**overrides: object) -> str:
    return compute_file_prefix(_Args(**overrides))


class TestExactPruneNaming(unittest.TestCase):
    def test_default_has_no_suffix(self) -> None:
        prefix = _prefix(game24_exact_prune=False)
        self.assertNotIn("_exactprune_True", prefix)
        self.assertIn("game24", prefix)

    def test_flag_adds_exactly_one_trailing_suffix(self) -> None:
        prefix = _prefix(game24_exact_prune=True)
        self.assertTrue(prefix.endswith("_exactprune_True"), prefix)
        self.assertEqual(prefix.count("_exactprune_True"), 1)

    def test_absent_attribute_adds_no_suffix(self) -> None:
        args = _Args()
        self.assertFalse(hasattr(args, "game24_exact_prune"))
        self.assertNotIn("_exactprune_True", compute_file_prefix(args))

    def test_naive_run_keeps_the_suffix(self) -> None:
        prefix = _prefix(
            naive_run=True,
            prompt_sample="cot",
            game24_exact_prune=True,
        )
        self.assertTrue(prefix.startswith("./logs/game24/local-model/remote-model/"))
        self.assertTrue(prefix.endswith("_exactprune_True"), prefix)

    def test_policy_marker_precedes_exact_prune_marker(self) -> None:
        prefix = _prefix(game24_exact_prune=True, search_policy="diverse")
        self.assertIn("_policy_diverse", prefix)
        self.assertTrue(prefix.endswith("_exactprune_True"), prefix)
        self.assertLess(
            prefix.index("_policy_diverse"),
            prefix.index("_exactprune_True"),
        )

    def test_task_count_marker_follows_exact_prune_marker(self) -> None:
        prefix = compute_file_prefix(
            _Args(game24_exact_prune=True, search_policy="dqn"),
            task_count=20,
        )
        self.assertTrue(prefix.endswith("_exactprune_True_ntasks20"), prefix)
        self.assertEqual(prefix.count("_ntasks20"), 1)


if __name__ == "__main__":
    unittest.main()
