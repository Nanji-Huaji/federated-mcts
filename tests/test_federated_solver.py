"""Contract tests for FederatedSolver strategy reuse.

PRODUCTION DEFECT: federated_solve calls get_strategy(name, ...) inside
the method body on every invocation, creating a brand-new strategy object
each time.  Stateful strategies (e.g. ContextualBanditStrategy) lose
accumulated learning across calls.

Expected contract: one named strategy instance is reused across
federated_solve calls for the same solver.  The cache key must include
task identity (class name + steps) so different task types do not
accidentally share a strategy.
"""

import sys
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, "src")

from federated_mcts.federation.orchestrator import FederatedSolver


class _TaskStub:
    steps = 3

    @staticmethod
    def get_input(idx):
        return "mock-x"

    @staticmethod
    def pre_generate_check(y):
        return True

    @staticmethod
    def propose_prompt_wrap(x, y):
        return "prompt"

    @staticmethod
    def process_generate_result(pro, x, y, check):
        return True, pro.strip()


class _OtherTaskStub(_TaskStub):
    steps = 4


class _ArgsStub:
    model_config = None
    temperature = 0.7
    method_generate = "propose"
    method_evaluate = "value"
    method_select = "greedy"
    n_generate_sample = 4
    n_evaluate_sample = 1
    n_select_sample = 2
    check_format = True


def _build_solver():
    solver = object.__new__(FederatedSolver)
    solver.args = _ArgsStub()
    solver.gpts = {"local": MagicMock(), "remote": MagicMock()}
    solver.model_config = []
    solver.time_tracker = MagicMock()
    solver.args.model_config = None
    return solver


def _fake_client_step(task, x, client_ys, step, solve_name, eval_name,
                      to_print=False, n_generate_sample=None, **kw):
    return ([], [], {}, {"generation": 0.0, "evaluation": 0.0})


class TestFederatedSolverStrategyReuse(unittest.TestCase):

    def test_strategy_instance_reused_across_calls(self):
        solver = _build_solver()
        task = _TaskStub()
        solver._client_step = _fake_client_step

        instances = []

        def tracking_get_strategy(name, **kw):
            from federated_mcts.federation.task_assign import RoundRobinStrategy
            inst = RoundRobinStrategy(**kw)
            instances.append(inst)
            return inst

        with patch(
            "federated_mcts.federation.task_assign.get_strategy",
            side_effect=tracking_get_strategy,
        ):
            solver.federated_solve(task, 0, to_print=False, assign_strategy="round_robin")
            solver.federated_solve(task, 1, to_print=False, assign_strategy="round_robin")

        self.assertGreaterEqual(len(instances), 1)
        expected_key = ("round_robin", "_TaskStub", task.steps)
        self.assertIn(
            expected_key, solver._strategy_cache,
            "DEFECT: strategy cache key does not include task class name.",
        )

    def test_different_task_classes_get_different_strategies(self):
        solver = _build_solver()
        task_a = _TaskStub()
        task_b = _OtherTaskStub()
        solver._client_step = _fake_client_step

        strategy_ids = set()

        def tracking_get_strategy(name, **kw):
            from federated_mcts.federation.task_assign import RoundRobinStrategy
            inst = RoundRobinStrategy(**kw)
            strategy_ids.add(id(inst))
            return inst

        with patch(
            "federated_mcts.federation.task_assign.get_strategy",
            side_effect=tracking_get_strategy,
        ):
            solver.federated_solve(task_a, 0, to_print=False, assign_strategy="round_robin")
            solver.federated_solve(task_b, 0, to_print=False, assign_strategy="round_robin")

        self.assertEqual(
            len(strategy_ids), 2,
            "DEFECT: different task classes (3 vs 4 steps) get the same "
            "cached strategy instance because task identity is not in the key.",
        )


if __name__ == "__main__":
    unittest.main()
