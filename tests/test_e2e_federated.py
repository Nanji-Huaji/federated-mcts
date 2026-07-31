"""Offline end-to-end test for FederatedSolver.federated_solve.

Runs a minimal multi-step MCTS solve with fake task and fake model clients.
Exercises: root replication, _run_assignments, proposal budget passing,
evaluation/selection, strategy cache, and timing — all without network.
"""

import sys
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, "src")

from federated_mcts.federation.orchestrator import FederatedSolver


# ── Fake multi-step task ──────────────────────────────────────────────

class _FakeTask:
    steps = 2
    stops = [None, None]
    value_cache = {}

    @staticmethod
    def get_input(idx):
        return "fake-problem"

    @staticmethod
    def pre_generate_check(y):
        return True

    @staticmethod
    def propose_prompt_wrap(x, y):
        return "propose-prompt"

    @staticmethod
    def value_prompt_wrap(x, y):
        return "value-prompt"

    @staticmethod
    def value_outputs_unwrap(x, y, value_outputs):
        return 0.5

    @staticmethod
    def process_generate_result(pro, x, y, check):
        return True, pro.strip()


# ── Fake deterministic model clients ──────────────────────────────────

def _fake_gpt_fn(return_lines):
    def fn(*args, **kwargs):
        return return_lines
    return fn


# ── Args ──────────────────────────────────────────────────────────────

class _E2EArgs:
    model_config = None
    temperature = 0.7
    method_generate = "propose"
    method_evaluate = "value"
    method_select = "greedy"
    n_generate_sample = 4
    n_evaluate_sample = 1
    n_select_sample = 2
    check_format = True
    prompt_sample = "standard"


class TestE2EFederatedSolve(unittest.TestCase):

    def setUp(self):
        self.solver = FederatedSolver.__new__(FederatedSolver)
        self.solver.args = _E2EArgs()
        self.solver.model_config = []
        self.solver.time_tracker = MagicMock()
        self.solver._strategy_cache = {}
        self.solver.args.model_config = None

        self.task = _FakeTask()

    def _build_fake_gpts(self, local_lines, remote_lines):
        return {
            "local": _fake_gpt_fn(local_lines),
            "remote": _fake_gpt_fn(remote_lines),
        }

    def test_full_two_step_federated_solve(self):
        self.solver.gpts = self._build_fake_gpts(
            local_lines=["step0-A\nstep0-B", "step1-local-A\nstep1-local-B"],
            remote_lines=["step0-C\nstep0-D", "step1-remote-C"],
        )

        with patch(
            "federated_mcts.core.evaluation.get_values",
            return_value=[0.9, 0.3, 0.8, 0.1],
        ), patch(
            "federated_mcts.federation.task_assign.get_strategy",
        ) as mock_gs:
            ys, info = self.solver.federated_solve(
                self.task, 0, to_print=False, assign_strategy="round_robin",
            )

        self.assertIsInstance(ys, list)
        self.assertGreater(len(ys), 0)
        self.assertEqual(len(info["steps"]), 2)

        step0 = info["steps"][0]
        self.assertIn("task_assignments", step0)
        self.assertIn("client_infos", step0)
        self.assertIn("step_client_times", step0)

        self.assertGreaterEqual(len(step0["new_ys"]), 1)
        self.assertGreaterEqual(len(step0["values"]), 1)

    def test_strategy_cache_populated_after_solve(self):
        self.solver.gpts = self._build_fake_gpts(
            ["a\nb", "c"], ["d\ne", "f"],
        )

        with patch(
            "federated_mcts.core.evaluation.get_values",
            return_value=[0.5, 0.5, 0.5, 0.5],
        ), patch(
            "federated_mcts.federation.task_assign.get_strategy",
        ):
            self.solver.federated_solve(
                self.task, 0, to_print=False, assign_strategy="round_robin",
            )

        expected_key = ("round_robin", "_FakeTask", self.task.steps)
        self.assertIn(
            expected_key, self.solver._strategy_cache,
            "Strategy was not cached with task-identity key after federated_solve.",
        )

    def test_solve_with_explicit_strategy_instance(self):
        from federated_mcts.federation.task_assign import RoundRobinStrategy
        self.solver.gpts = self._build_fake_gpts(
            ["a"], ["b"],
        )
        strategy = RoundRobinStrategy(eval_client="remote")

        with patch(
            "federated_mcts.core.evaluation.get_values",
            return_value=[0.6, 0.4],
        ):
            ys, info = self.solver.federated_solve(
                self.task, 0, to_print=False, assign_strategy=strategy,
            )

        self.assertIsInstance(ys, list)

    def test_single_client_federated_solve(self):
        self.solver.gpts = {"only": _fake_gpt_fn(["a\nb", "c"])}

        with patch(
            "federated_mcts.core.evaluation.get_values",
            return_value=[0.7, 0.3, 0.5],
        ):
            ys, info = self.solver.federated_solve(
                self.task, 0, to_print=False, assign_strategy="round_robin",
            )

        self.assertGreaterEqual(len(ys), 1)


if __name__ == "__main__":
    unittest.main()
