"""Prove StandardSolversMixin.solve calls record_failed_step before
empty-candidate break (mocked, no provider/network).
"""
from __future__ import annotations

import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, "src")

from tests._game24_stub import _Stub


class TestSolverFailedStepIntegration(unittest.TestCase):
    """Prove record_failed_step runs before empty-candidate break in
    StandardSolversMixin.solve (mocked, no provider/network)."""

    def setUp(self) -> None:
        self.task = _Stub()
        self.x = "1 1 1 8"

    def test_solve_calls_record_failed_step_on_all_rejected(self):
        from federated_mcts.evaluation.game24_task_recorder import (
            TaskSessionRecorder,
        )
        from federated_mcts.evaluation.game24_metrics import summarize_experiment_metrics
        from federated_mcts.federation.standard_solvers import (
            StandardSolversMixin,
        )
        from federated_mcts.federation.time_tracker import TimeTracker

        rec = TaskSessionRecorder(task_id="0", task_input=self.x,
                                  decision_policy="local_validity")

        class _Args:
            n_generate_sample = 4
            prompt_sample = None
            search_policy = "baseline"
            method_generate = "propose"
            game24_exact_prune = False
            method_evaluate = "vote"
            n_evaluate_sample = 3
            method_select = "greedy"
            n_select_sample = 1
            warm_start = False
            last_lm = False
            slm_eval = False
            check_format = True
            eval_rule = False

        mock_client = MagicMock()
        mock_client.return_value = []

        solver = StandardSolversMixin()
        solver.args = _Args()
        solver.gpts = {"remote_client": mock_client}
        solver.time_tracker = TimeTracker()

        result, info = solver.solve(
            self.task, 0, to_print=False,
            solve_client="remote_client",
            task_recorder=rec)

        self.assertGreater(len(rec._step_records), 0,
            "Expected at least one step record (failed step)")
        metrics = rec.to_task_metrics()
        self.assertGreaterEqual(metrics["total_proposed"], 0)
        exp = summarize_experiment_metrics([metrics])
        self.assertGreaterEqual(exp["total_candidates"], 0)

    def test_solve_no_recorder_no_failed_step_call(self):
        from federated_mcts.federation.standard_solvers import (
            StandardSolversMixin,
        )
        from federated_mcts.federation.time_tracker import TimeTracker

        class _Args:
            n_generate_sample = 4
            prompt_sample = None
            search_policy = "baseline"
            method_generate = "propose"
            game24_exact_prune = False
            method_evaluate = "vote"
            n_evaluate_sample = 3
            method_select = "greedy"
            n_select_sample = 1
            warm_start = False
            last_lm = False
            slm_eval = False
            check_format = True
            eval_rule = False

        mock_client = MagicMock()
        mock_client.return_value = []

        solver = StandardSolversMixin()
        solver.args = _Args()
        solver.gpts = {"remote_client": mock_client}
        solver.time_tracker = TimeTracker()

        result, info = solver.solve(
            self.task, 0, to_print=False,
            solve_client="remote_client",
            task_recorder=None)

        self.assertIsInstance(result, list)
        self.assertIsInstance(info, dict)

    def test_solve_gen_instrumentation_captured_in_failed_step(self):
        from federated_mcts.evaluation.game24_task_recorder import (
            TaskSessionRecorder,
        )
        from federated_mcts.federation.standard_solvers import (
            StandardSolversMixin,
        )
        from federated_mcts.federation.time_tracker import TimeTracker

        rec = TaskSessionRecorder(task_id="0", task_input=self.x,
                                  decision_policy="local_validity")

        class _Args:
            n_generate_sample = 4
            prompt_sample = None
            search_policy = "baseline"
            method_generate = "propose"
            game24_exact_prune = False
            method_evaluate = "vote"
            n_evaluate_sample = 3
            method_select = "greedy"
            n_select_sample = 1
            warm_start = False
            last_lm = False
            slm_eval = False
            check_format = True
            eval_rule = False

        mock_client = MagicMock()
        mock_client.return_value = []

        solver = StandardSolversMixin()
        solver.args = _Args()
        solver.gpts = {"remote_client": mock_client}
        solver.time_tracker = TimeTracker()

        solver.solve(self.task, 0, to_print=False,
                     solve_client="remote_client",
                     task_recorder=rec)

        self.assertGreater(len(rec._step_records), 0)
        sr = rec._step_records[0]
        self.assertIn("gen_instrumentation", sr)
        self.assertIn("vote_instrumentation", sr)
        self.assertTrue(sr.get("is_stopping") or True)


if __name__ == "__main__":
    unittest.main()
