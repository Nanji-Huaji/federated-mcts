"""Tests for paired analysis, bootstrap CI, and oracle rejection.

Covers: paired delta, task-ID mismatch rejection, incomplete rejection,
bootstrap determinism, and oracle-mislabeling rejection.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

sys.path.insert(0, "src")

from tests.test_game24_analysis_shared import (
    make_manifest_json,
    make_performance_json,
    make_tasks_json,
)


class TestPairedAnalysis(unittest.TestCase):
    """Given two loaded results, When run_paired_analysis() is called,
       Then it produces correct delta, CI, and validation."""

    def setUp(self) -> None:
        self.tmpdir_obj = tempfile.TemporaryDirectory()
        self.tmpdir = self.tmpdir_obj.name

    def tearDown(self) -> None:
        self.tmpdir_obj.cleanup()

    def _write_experiment(
        self, prefix: str, *,
        task_ids=None, solved_ids=None,
        decision_policy: str = "local_validity",
        complete: bool = True, acc=None,
    ) -> tuple[str, str, str]:
        m_path = os.path.join(self.tmpdir, f"{prefix}_manifest.json")
        p_path = os.path.join(self.tmpdir, f"{prefix}_performance.json")
        t_path = os.path.join(self.tmpdir, f"{prefix}.json")
        if task_ids is None:
            task_ids = [3, 9, 16, 19]
        if solved_ids is None:
            solved_ids = set()
        n = len(task_ids)
        s = len(solved_ids)
        computed_acc = s / n if n else 0.0
        with open(m_path, "w") as f:
            f.write(make_manifest_json(
                task_ids=task_ids, decision_policy=decision_policy,
                ordered_task_ids_sha256="b" * 64,
            ))
        with open(p_path, "w") as f:
            f.write(make_performance_json(
                acc=acc if acc is not None else computed_acc,
                solved_count=s, complete=complete, started_tasks=n,
            ))
        with open(t_path, "w") as f:
            f.write(make_tasks_json(task_ids=task_ids, solved_ids=solved_ids))
        return m_path, p_path, t_path

    def test_matching_tasks_paired_delta(self):
        from federated_mcts.evaluation.game24_analysis import (
            load_result, run_paired_analysis,
        )
        b_mf, b_pf, b_tf = self._write_experiment("baseline", solved_ids={3, 9})
        c_mf, c_pf, c_tf = self._write_experiment("candidate", solved_ids={3, 9, 16})
        baseline = load_result(b_mf, b_pf, b_tf)
        candidate = load_result(c_mf, c_pf, c_tf)
        a = run_paired_analysis(baseline, candidate, seed=42)
        self.assertAlmostEqual(a.baseline_accuracy, 0.5)
        self.assertAlmostEqual(a.candidate_accuracy, 0.75)
        self.assertAlmostEqual(a.delta, 0.25)

    def test_mismatched_tasks_rejected(self):
        from federated_mcts.evaluation.game24_analysis import (
            load_result, run_paired_analysis,
        )
        b_mf, b_pf, b_tf = self._write_experiment(
            "baseline", task_ids=[3, 9], solved_ids={3},
        )
        c_mf, c_pf, c_tf = self._write_experiment(
            "candidate", task_ids=[16, 19], solved_ids={16},
        )
        baseline = load_result(b_mf, b_pf, b_tf)
        candidate = load_result(c_mf, c_pf, c_tf)
        with self.assertRaises(ValueError) as ctx:
            run_paired_analysis(baseline, candidate)
        self.assertIn("mismatch", str(ctx.exception).lower())

    def test_incomplete_candidate_rejected(self):
        from federated_mcts.evaluation.game24_analysis import load_result
        _, _, _ = self._write_experiment("base", complete=True)
        c_mf, c_pf, c_tf = self._write_experiment("cand", complete=False)
        with self.assertRaises(ValueError):
            load_result(c_mf, c_pf, c_tf)

    def test_delta_zero_when_same_accuracy(self):
        from federated_mcts.evaluation.game24_analysis import (
            load_result, run_paired_analysis,
        )
        b_mf, b_pf, b_tf = self._write_experiment("b", solved_ids={3, 9})
        c_mf, c_pf, c_tf = self._write_experiment("c", solved_ids={3, 9})
        a = run_paired_analysis(
            load_result(b_mf, b_pf, b_tf),
            load_result(c_mf, c_pf, c_tf),
            seed=42,
        )
        self.assertAlmostEqual(a.delta, 0.0)

    def test_oracle_mislabeled_model_only_rejected(self):
        from federated_mcts.evaluation.game24_analysis import (
            load_result, run_paired_analysis,
        )
        b_mf, b_pf, b_tf = self._write_experiment(
            "b", decision_policy="local_validity",
        )
        c_mf, c_pf, c_tf = self._write_experiment(
            "c", decision_policy="oracle_exact_pruned",
        )
        baseline = load_result(b_mf, b_pf, b_tf)
        candidate = load_result(c_mf, c_pf, c_tf)
        with self.assertRaises(ValueError) as ctx:
            run_paired_analysis(baseline, candidate)
        self.assertIn("oracle", str(ctx.exception).lower())


class TestBootstrapCI(unittest.TestCase):
    """Given known paired data, When bootstrap is computed,
       Then the CI is reproducible with the same seed."""

    def test_deterministic_bootstrap(self):
        from federated_mcts.evaluation.game24_analysis import compute_paired_bootstrap_ci
        paired: list[tuple[int, int]] = (
            [(1, 1)] * 3 + [(1, 0)] * 1 + [(0, 1)] * 4 + [(0, 0)] * 2
        )
        ci1 = compute_paired_bootstrap_ci(paired, seed=42)
        ci2 = compute_paired_bootstrap_ci(paired, seed=42)
        self.assertEqual(ci1, ci2)

    def test_bootstrap_ci_bounded(self):
        from federated_mcts.evaluation.game24_analysis import compute_paired_bootstrap_ci
        paired: list[tuple[int, int]] = [(1, 1)] * 3 + [(0, 1)] * 4 + [(0, 0)] * 3
        ci1 = compute_paired_bootstrap_ci(paired, seed=42)
        ci2 = compute_paired_bootstrap_ci(paired, seed=99)
        for ci in (ci1, ci2):
            self.assertGreaterEqual(ci[0], -1.0)
            self.assertLessEqual(ci[1], 1.0)

    def test_all_same_ci_is_delta(self):
        from federated_mcts.evaluation.game24_analysis import compute_paired_bootstrap_ci
        paired: list[tuple[int, int]] = [(1, 1)] * 20
        ci = compute_paired_bootstrap_ci(paired, seed=42)
        self.assertAlmostEqual(ci[0], 0.0)
        self.assertAlmostEqual(ci[1], 0.0)


if __name__ == "__main__":
    unittest.main()
