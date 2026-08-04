"""Tests for ``load_result`` boundary validation and result integrity.

Covers: valid loading, solved/unsolved extraction, parse-failure handling,
rejection of missing/incomplete/non-finite/mislabeled inputs,
and result data format checks.
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


class TestLoadResult(unittest.TestCase):
    """Given valid/invalid result files, When load_result() is called,
       Then it returns a validated ExperimentResult or raises."""

    def setUp(self) -> None:
        self.tmpdir_obj = tempfile.TemporaryDirectory()
        self.tmpdir = self.tmpdir_obj.name

    def tearDown(self) -> None:
        self.tmpdir_obj.cleanup()

    def _write_fixtures(
        self, prefix: str,
        *, manifest_overrides=None, perf_overrides=None, tasks_overrides=None,
    ) -> tuple[str, str, str]:
        m_path = os.path.join(self.tmpdir, f"{prefix}_manifest.json")
        p_path = os.path.join(self.tmpdir, f"{prefix}_performance.json")
        t_path = os.path.join(self.tmpdir, f"{prefix}.json")
        with open(m_path, "w") as f:
            f.write(make_manifest_json(**(manifest_overrides or {})))
        with open(p_path, "w") as f:
            f.write(make_performance_json(**(perf_overrides or {})))
        with open(t_path, "w") as f:
            f.write(make_tasks_json(**(tasks_overrides or {})))
        return m_path, p_path, t_path

    def test_load_valid_result(self):
        from federated_mcts.evaluation.game24_analysis import load_result
        mf, pf, tf = self._write_fixtures(
            "valid", tasks_overrides={"solved_ids": {3, 9}},
            perf_overrides={"acc": 0.5, "solved_count": 2},
        )
        r = load_result(mf, pf, tf)
        self.assertEqual(r.total_tasks, 4)
        self.assertEqual(r.accuracy, 0.5)

    def test_load_solved_and_unsolved(self):
        from federated_mcts.evaluation.game24_analysis import load_result
        mf, pf, tf = self._write_fixtures(
            "sol", tasks_overrides={"solved_ids": {3, 16}},
        )
        r = load_result(mf, pf, tf)
        self.assertAlmostEqual(r.accuracy, 0.5)

    def test_parse_failures_are_unsolved(self):
        from federated_mcts.evaluation.game24_analysis import load_result
        mf, pf, tf = self._write_fixtures(
            "pf", tasks_overrides={"parse_failure_ids": {3, 9, 16, 19}},
            perf_overrides={"acc": 0.0, "solved_count": 0},
        )
        r = load_result(mf, pf, tf)
        self.assertAlmostEqual(r.accuracy, 0.0)

    def test_missing_manifest_rejected(self):
        from federated_mcts.evaluation.game24_analysis import load_result
        mf, pf, tf = self._write_fixtures("miss")
        os.remove(mf)
        with self.assertRaises(FileNotFoundError):
            load_result(mf, pf, tf)

    def test_incomplete_run_rejected(self):
        from federated_mcts.evaluation.game24_analysis import load_result
        mf, pf, tf = self._write_fixtures(
            "inc", perf_overrides={"complete": False, "stop_reason": "max_total_calls"},
        )
        with self.assertRaises(ValueError) as ctx:
            load_result(mf, pf, tf)
        self.assertIn("incomplete", str(ctx.exception).lower())

    def test_non_finite_cost_rejected(self):
        from federated_mcts.evaluation.game24_analysis import load_result
        mf, pf, tf = self._write_fixtures(
            "nan", perf_overrides={"estimated_cost_usd": float("nan")},
        )
        with self.assertRaises(ValueError):
            load_result(mf, pf, tf)

    def test_non_finite_accuracy_rejected(self):
        from federated_mcts.evaluation.game24_analysis import load_result
        mf, pf, tf = self._write_fixtures(
            "nan2", perf_overrides={"acc": float("nan")},
        )
        with self.assertRaises(ValueError):
            load_result(mf, pf, tf)

    def test_bool_accuracy_rejected(self):
        from federated_mcts.evaluation.game24_analysis import load_result
        mf, pf, tf = self._write_fixtures(
            "bool", perf_overrides={"acc": True},
        )
        with self.assertRaises(ValueError):
            load_result(mf, pf, tf)

    def test_zero_task_result_loads(self):
        from federated_mcts.evaluation.game24_analysis import load_result
        mf, pf, tf = self._write_fixtures(
            "zero", tasks_overrides={"task_ids": []},
            perf_overrides={"acc": 0.0, "avg_sum": 0.0, "started_tasks": 0},
        )
        result = load_result(mf, pf, tf)
        self.assertEqual(result.total_tasks, 0)


class TestReportRejections(unittest.TestCase):
    """Given invalid inputs, When load_result is called,
       Then it raises clear errors."""

    def setUp(self) -> None:
        self.tmpdir_obj = tempfile.TemporaryDirectory()
        self.tmpdir = self.tmpdir_obj.name

    def tearDown(self) -> None:
        self.tmpdir_obj.cleanup()

    def test_report_rejects_missing_manifest_file(self):
        from federated_mcts.evaluation.game24_analysis import load_result
        with self.assertRaises(FileNotFoundError):
            load_result(
                os.path.join(self.tmpdir, "ghost_manifest.json"),
                os.path.join(self.tmpdir, "ghost_perf.json"),
                os.path.join(self.tmpdir, "ghost_tasks.json"),
            )


class TestResultFormat(unittest.TestCase):
    """Result data integrity checks."""

    def setUp(self) -> None:
        self.tmpdir_obj = tempfile.TemporaryDirectory()
        self.tmpdir = self.tmpdir_obj.name

    def tearDown(self) -> None:
        self.tmpdir_obj.cleanup()

    def _make_and_load(self, prefix: str, **task_kwargs):  # type: ignore[no-untyped-def]
        from federated_mcts.evaluation.game24_analysis import load_result
        m_path = os.path.join(self.tmpdir, f"{prefix}_manifest.json")
        p_path = os.path.join(self.tmpdir, f"{prefix}_performance.json")
        t_path = os.path.join(self.tmpdir, f"{prefix}.json")
        task_ids = task_kwargs.pop("task_ids", [3, 9, 16, 19])
        solved_ids = task_kwargs.pop("solved_ids", {3, 9})
        s = len(solved_ids)
        n = len(task_ids)
        with open(m_path, "w") as f:
            f.write(make_manifest_json(task_ids=task_ids))
        with open(p_path, "w") as f:
            f.write(make_performance_json(
                acc=s / n, solved_count=s, started_tasks=n, decision_fp="d" * 64,
            ))
        with open(t_path, "w") as f:
            f.write(make_tasks_json(task_ids=task_ids, solved_ids=solved_ids, **task_kwargs))
        return load_result(m_path, p_path, t_path)

    def test_total_tasks_matches_task_ids(self):
        result = self._make_and_load("t1")  # type: ignore[arg-type]
        self.assertEqual(result.total_tasks, 4)  # type: ignore[union-attr]

    def test_solved_count_matches(self):
        result = self._make_and_load(  # type: ignore[arg-type]
            "t2", task_ids=[3, 9, 16, 19], solved_ids={3, 9, 16},
        )
        self.assertEqual(result.solved_count, 3)  # type: ignore[union-attr]

    def test_accuracy_is_strictly_correct(self):
        result = self._make_and_load(  # type: ignore[arg-type]
            "t3", task_ids=[3, 9], solved_ids={3},
        )
        self.assertAlmostEqual(result.accuracy, 0.5)  # type: ignore[union-attr]


if __name__ == "__main__":
    unittest.main()
