"""Contract test for n_generate_sample budget splitting in federated_solve.

Expected contract: federated_solve computes per-assignment generation budget
as ceil(total_n_generate_sample / active_assignment_count), then passes
the per-assignment n_gen to _client_step.  Non-divisible budgets round up.
"""

import sys
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, "src")

from federated_mcts.federation.orchestrator import FederatedSolver
from federated_mcts.federation.task_assign import RoundRobinStrategy


class _TaskStub:
    steps = 1

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


class _ArgsStub:
    model_config = None
    temperature = 0.7
    method_generate = "propose"
    method_evaluate = "value"
    method_select = "greedy"
    n_generate_sample = 7   # non-divisible by 2
    n_evaluate_sample = 1
    n_select_sample = 2
    check_format = True


def _build_solver(n_generate_sample=7):
    solver = object.__new__(FederatedSolver)
    solver.args = _ArgsStub()
    solver.args.n_generate_sample = n_generate_sample
    solver.gpts = {"local": MagicMock(), "remote": MagicMock()}
    solver.model_config = []
    solver.time_tracker = MagicMock()
    solver.args.model_config = None
    return solver


class TestBudgetCeilSplitting(unittest.TestCase):

    def test_ceil_split_two_active_assignments(self):
        solver = _build_solver(n_generate_sample=7)
        task = _TaskStub()
        received_n_gens = []

        def fake_client_step(task, x, client_ys, step, solve_name, eval_name,
                             to_print=False, n_generate_sample=None, **kw):
            received_n_gens.append(n_generate_sample)
            return ([], [], {}, {"generation": 0.0, "evaluation": 0.0})

        solver._client_step = fake_client_step

        solver.federated_solve(
            task, 0, to_print=False,
            assign_strategy=RoundRobinStrategy(eval_client="remote"),
        )

        self.assertEqual(len(received_n_gens), 2)
        expected = 4   # ceil(7 / 2) = 4
        self.assertEqual(
            received_n_gens[0], expected,
            "DEFECT: per-assignment n_gen for first active assignment is not ceil(7/2)=4.",
        )
        self.assertEqual(
            received_n_gens[1], expected,
            "DEFECT: per-assignment n_gen for second active assignment is not ceil(7/2)=4.",
        )

    def test_exact_division(self):
        solver = _build_solver(n_generate_sample=8)
        task = _TaskStub()
        received_n_gens = []

        def fake_client_step(task, x, client_ys, step, solve_name, eval_name,
                             to_print=False, n_generate_sample=None, **kw):
            received_n_gens.append(n_generate_sample)
            return ([], [], {}, {"generation": 0.0, "evaluation": 0.0})

        solver._client_step = fake_client_step

        solver.federated_solve(
            task, 0, to_print=False,
            assign_strategy=RoundRobinStrategy(eval_client="remote"),
        )

        expected = 4   # 8 / 2 = 4
        for n in received_n_gens:
            self.assertEqual(n, expected)

    def test_single_active_assignment_gets_full_budget(self):
        solver = _build_solver(n_generate_sample=5)
        task = _TaskStub()
        received_n_gens = []

        def fake_client_step(task, x, client_ys, step, solve_name, eval_name,
                             to_print=False, n_generate_sample=None, **kw):
            received_n_gens.append(n_generate_sample)
            return ([], [], {}, {"generation": 0.0, "evaluation": 0.0})

        solver._client_step = fake_client_step
        solver.gpts = {"local": MagicMock()}

        solver.federated_solve(
            task, 0, to_print=False,
            assign_strategy=RoundRobinStrategy(eval_client="local"),
        )

        self.assertEqual(received_n_gens[0], 5)


if __name__ == "__main__":
    unittest.main()
