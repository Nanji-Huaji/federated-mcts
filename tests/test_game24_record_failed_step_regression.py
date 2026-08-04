"""Regression test: StandardSolversMixin.solve path triggers record_failed_step
when all generation proposals are invalid, and telemetry flows into task metrics.

Given a mock task that rejects every proposal line, a mock client that returns
format-invalid text, and a TaskSessionRecorder wired into solve, the recorder
must capture the failed step with raw proposal telemetry, and the resulting
task/experiment metrics must reflect the rejection counts.
"""

from __future__ import annotations

import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, "src")

# Prevent federated_mcts.federation.__init__ → openai import cascade
_patches = [
    patch.dict(sys.modules, {
        "openai": MagicMock(),
        "openai.api_resources": MagicMock(),
        "openai.api_resources.chat_completion": MagicMock(),
    }),
]
for _p in _patches:
    _p.start()

from federated_mcts.evaluation.game24_task_recorder import (
    TaskSessionRecorder,
)
from federated_mcts.evaluation.game24_metrics import (
    summarize_experiment_metrics,
)
from federated_mcts.federation.standard_solvers import (
    StandardSolversMixin,
)
from federated_mcts.federation.time_tracker import TimeTracker


class _AllRejectTask:
    """Task stub that rejects every proposal and returns empty on no candidates."""
    steps = 3

    @staticmethod
    def get_input(idx: int) -> str:
        return "1 1 1 8"

    @staticmethod
    def canonical_state_key(x: str, y: str) -> tuple:
        from fractions import Fraction
        return tuple(sorted(Fraction(p) for p in y.strip().split() if p.strip()))

    @staticmethod
    def is_success_state(x: str, y: str) -> bool:
        return False

    @staticmethod
    def propose_prompt_wrap(x: str, y: str) -> str:
        return "generate 24-point proposals"

    @staticmethod
    def process_generate_result(pro: str, x: str, y: str, check_format: bool) -> tuple[bool, str]:
        return (False, pro)

    @staticmethod
    def on_no_valid_candidates(y: str) -> list[str]:
        return []

    @staticmethod
    def pre_generate_check(y: str) -> bool:
        return True

    stops: list[str | None] = [None] * 10


def _make_all_reject_client():
    """Return a callable that satisfies GenerationClient: (args, prompt, n=1) -> list[str]."""
    def client(_args, _prompt, n: int = 1, stop=None, get_logprobs=False) -> list[str]:
        return ["not a valid line\ncannot parse\nx+y=z\n"] * n
    return client


class TestSolvePathAllRejected(unittest.TestCase):
    """Given a StandardSolversMixin with an all-reject task, When solve is called
    with a TaskSessionRecorder, Then record_failed_step captures reject telemetry."""

    def setUp(self) -> None:
        self.task = _AllRejectTask()

    def _make_args(self) -> MagicMock:
        args = MagicMock()
        args.n_generate_sample = 4
        args.prompt_sample = None
        args.method_generate = "propose"
        args.method_evaluate = "vote"
        args.method_select = "greedy"
        args.n_evaluate_sample = 1
        args.n_select_sample = 1
        args.check_format = True
        args.warm_start = False
        args.last_lm = False
        args.slm_eval = False
        args.game24_exact_prune = False
        args.search_policy = "baseline"
        return args

    def _make_solver(self, args, client) -> StandardSolversMixin:
        class _Solver(StandardSolversMixin):
            pass
        solver = _Solver()
        solver.args = args
        solver.gpts = {"remote_client": client}
        solver.time_tracker = TimeTracker(["remote_client"])
        solver.dqn_session = None
        return solver

    def test_solve_path_records_failed_step(self) -> None:
        """When: all proposals are rejected; Then: step_records contain a failed step."""
        args = self._make_args()
        client = _make_all_reject_client()
        solver = self._make_solver(args, client)
        recorder = TaskSessionRecorder(
            task_id="0", task_input="1 1 1 8",
            decision_policy="local_validity")

        ys, info = solver.solve(
            self.task, 0, to_print=False,
            solve_client="remote_client",
            task_recorder=recorder,
        )
        # After failed-step break, ys retains value from prior iteration ([""])
        self.assertGreater(len(recorder._step_records), 0,
                           "Expected at least one step record")
        last = recorder._step_records[-1]
        self.assertTrue(last["is_stopping"],
                        "Failed step should be stopping")
        self.assertGreater(len(last["candidates"]), 0,
                           "Failed step should carry candidate events")
        self.assertGreater(len(last["gen_instrumentation"]), 0,
                           "Failed step should carry gen instrumentation")

    def test_failed_step_rejection_flow_to_task_metrics(self) -> None:
        """When: all proposals are rejected; Then: task metrics contain rejection counts."""
        args = self._make_args()
        client = _make_all_reject_client()
        solver = self._make_solver(args, client)
        recorder = TaskSessionRecorder(
            task_id="0", task_input="1 1 1 8",
            decision_policy="local_validity")

        solver.solve(self.task, 0, to_print=False,
                     solve_client="remote_client",
                     task_recorder=recorder)
        metrics = recorder.to_task_metrics()
        step_recs = recorder._step_records
        # Sum up candidate count across all step records
        total_candidates = sum(len(sr["candidates"]) for sr in step_recs)
        self.assertEqual(
            metrics["total_proposed"], total_candidates,
            "Total proposed should match sum of all step candidates")
        self.assertEqual(
            metrics["total_candidates"], total_candidates,
            "Total candidates should match sum of all step candidates")
        rejection = metrics["rejection_counts"]
        self.assertGreater(
            rejection.get("malformed_format", 0) + rejection.get("empty_content", 0),
            0, "At least one rejection reason should be present")
        for sr in step_recs:
            for ev in sr["candidates"]:
                self.assertEqual(
                    ev["stage"], "proposed",
                    "All failed-step events should be at proposed stage")
                self.assertFalse(
                    ev["is_selected"],
                    "No failed-step candidate should be selected")

    def test_failed_step_flow_to_experiment_metrics(self) -> None:
        """When: all steps fail; Then: experiment aggregates reflect reject telemetry."""
        args = self._make_args()
        client = _make_all_reject_client()

        # Two tasks: one normal, one all-rejected
        # Task 1: normal (use Stub style task that accepts proposals)
        class _AcceptTask(_AllRejectTask):
            @staticmethod
            def process_generate_result(pro, x, y, check_format):
                return (True, pro)
            @staticmethod
            def on_no_valid_candidates(y):
                return [y]
            @staticmethod
            def test_output_modify(idx, output):
                return {"r": 0}, output
            @staticmethod
            def vote_prompt_wrap(x, ys):
                return "vote prompt"
            @staticmethod
            def vote_outputs_unwrap(outputs, n):
                return [1] * n

        accept_solver = self._make_solver(args, self._make_accept_client())
        accept_rec = TaskSessionRecorder(
            task_id="accept", task_input="1 1 1 8",
            decision_policy="local_validity")
        accept_solver.solve(_AcceptTask(), 0, to_print=False,
                            solve_client="remote_client",
                            task_recorder=accept_rec)

        reject_solver = self._make_solver(args, client)
        reject_rec = TaskSessionRecorder(
            task_id="reject", task_input="1 1 1 8",
            decision_policy="local_validity")
        reject_solver.solve(self.task, 0, to_print=False,
                             solve_client="remote_client",
                             task_recorder=reject_rec)

        exp = summarize_experiment_metrics([
            accept_rec.to_task_metrics(),
            reject_rec.to_task_metrics(),
        ])
        self.assertEqual(exp["task_count"], 2)
        self.assertEqual(exp["solved_task_count"], 0)
        self.assertGreater(exp["total_proposed"], 0)
        self.assertGreater(exp["total_candidates"], 0)
        # Rejection counts should aggregate from the failed task
        rc = exp["rejection_counts"]
        self.assertGreater(
            rc.get("malformed_format", 0) + rc.get("empty_content", 0),
            0, "Experiment-level rejection counts must include failed-step telemetry")

    def _make_accept_client(self):
        def client(_args, _prompt, n: int = 1, stop=None, get_logprobs=False) -> list[str]:
            return ["1 + 2 = 3 (left: 3 3 8)\n2 + 3 = 5 (left: 5 3)\n"] * n
        return client


if __name__ == "__main__":
    unittest.main()
