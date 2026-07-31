"""Contract test for FederatedSolver._run_assignments concurrent-execution seam.

PRODUCTION DEFECT: The assignment loop inside federated_solve executes
each non-empty assignment sequentially.  Expected contract: a new method
`_run_assignments(task, x, step, task_assignments, to_print, n_gen)`
that receives prepared callable jobs (or equivalent minimal arguments),
executes non-empty assignments concurrently, and returns results in
original assignment order.

The seam does not exist yet, so every test here fails with
AttributeError — the missing-method documentation for the implementation
phase.
"""

import sys
import threading
import time
import unittest
from unittest.mock import MagicMock

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

    def process_generate_result(self, pro, x, y, check):
        return True, pro.strip()


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


def _make_fake_client_step(solver, delay=0.0):
    """Return a fake _client_step that records invocation order and
    optionally simulates compute time."""

    order = []
    lock = threading.Lock()

    def fake_step(task, x, client_ys, step, solve_name, eval_name,
                  to_print=False, n_generate_sample=None,
                  uncertainty_backoff=False, uncertainty_threshold=0.8):
        if delay:
            time.sleep(delay)
        with lock:
            order.append(solve_name)
        return ([f"{solve_name}-r"], [0.5], {"solve_client_name": solve_name}, {"generation": 0.01, "evaluation": 0.01})

    fake_step.order = order
    return fake_step


def _build_solver():
    solver = object.__new__(FederatedSolver)
    solver.args = _ArgsStub()
    solver.gpts = {"local": MagicMock(), "remote": MagicMock()}
    solver.model_config = []
    solver.time_tracker = MagicMock()
    return solver


class TestRunAssignmentsSeamExists(unittest.TestCase):

    def test_method_exists_on_solver(self):
        solver = _build_solver()
        self.assertTrue(
            callable(getattr(solver, "_run_assignments", None)),
            "DEFECT: FederatedSolver._run_assignments does not exist. "
            "Expected: a seam that executes non-empty assignments concurrently.",
        )


class TestRunAssignmentsConcurrentExecution(unittest.TestCase):

    def setUp(self):
        self.solver = _build_solver()
        self.task = _TaskStub()

    def _standard_assignments(self):
        return [
            {"solve_client": "local", "eval_client": "remote", "ys": ["a"]},
            {"solve_client": "remote", "eval_client": "remote", "ys": ["b"]},
        ]

    def test_executes_concurrently(self):
        """Non-empty assignments run in parallel — total time ~= max,
        not sum of per-assignment times."""
        fake_step = _make_fake_client_step(self.solver, delay=0.05)
        self.solver._client_step = fake_step

        assignments = self._standard_assignments()
        n_gen = self.solver.args.n_generate_sample // len(assignments)

        t0 = time.perf_counter()
        self.solver._run_assignments(
            self.task, "x", 0, assignments, to_print=False, n_gen=n_gen,
        )
        elapsed = time.perf_counter() - t0

        self.assertLess(
            elapsed, 0.09,
            f"DEFECT: _run_assignments is missing or serial.  "
            f"Elapsed: {elapsed:.3f}s.  Expected: < 0.09s for 2 x 0.05s concurrent.",
        )

    def test_results_preserve_assignment_order(self):
        """Results from _run_assignments must appear in original assignment
        order, even if concurrent execution finishes out of order."""
        fake_step = _make_fake_client_step(self.solver, delay=0.0)
        self.solver._client_step = fake_step

        assignments = self._standard_assignments()
        n_gen = self.solver.args.n_generate_sample // len(assignments)

        all_new_ys, all_values, step_infos, step_times = self.solver._run_assignments(
            self.task, "x", 0, assignments, to_print=False, n_gen=n_gen,
        )

        self.assertEqual(
            all_new_ys, ["local-r", "remote-r"],
            "DEFECT: results not in original assignment order.",
        )

    def test_empty_assignments_skipped(self):
        """Assignments with empty ys are skipped; no-op returns empty."""
        fake_step = _make_fake_client_step(self.solver)
        self.solver._client_step = fake_step

        assignments = [
            {"solve_client": "local", "eval_client": "local", "ys": []},
            {"solve_client": "remote", "eval_client": "remote", "ys": []},
        ]
        n_gen = 1

        all_new_ys, all_values, step_infos, step_times = self.solver._run_assignments(
            self.task, "x", 0, assignments, to_print=False, n_gen=n_gen,
        )

        self.assertEqual(all_new_ys, [])
        self.assertEqual(all_values, [])
        self.assertEqual(fake_step.order, [])

    def test_sparse_assignments_empty_active_empty_active(self):
        """Active assignments at non-contiguous original indices must
        not cause IndexError when results are stored by original position
        into a dense results list."""
        fake_step = _make_fake_client_step(self.solver, delay=0.0)
        self.solver._client_step = fake_step

        assignments = [
            {"solve_client": "a", "eval_client": "a", "ys": []},
            {"solve_client": "b", "eval_client": "b", "ys": ["b1"]},
            {"solve_client": "c", "eval_client": "c", "ys": []},
            {"solve_client": "d", "eval_client": "d", "ys": ["d1"]},
        ]
        n_gen = 1

        all_new_ys, all_values, step_infos, step_times = self.solver._run_assignments(
            self.task, "x", 0, assignments, to_print=False, n_gen=n_gen,
        )

        self.assertEqual(
            all_new_ys, ["b-r", "d-r"],
            "DEFECT: sparse active assignments (indices 1,3 with 2 jobs) "
            "cause IndexError because original index 3 >= len(results)=2.",
        )
        self.assertEqual(len(step_infos), 2)


if __name__ == "__main__":
    unittest.main()
