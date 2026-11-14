"""Todo 7 runner/solver-level 5-task resume test."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, "src")


class _Stub:
    steps = 3

    @staticmethod
    def get_input(idx: int) -> str:
        return ["1 1 1 8", "2 2 2 6", "3 3 3 3", "4 4 4 4", "5 5 5 5"][idx]

    @staticmethod
    def canonical_state_key(x: str, y: str) -> tuple:
        from fractions import Fraction
        import re
        lines = [l for l in y.strip().split("\n") if "left:" in l]
        tgt = lines[-1] if lines else y
        m = re.search(r"left:\s*([^)]+)", tgt)
        parts = (m.group(1).strip().split() if m else tgt.strip().split())
        return tuple(sorted(Fraction(p) for p in parts if p.strip()))

    @staticmethod
    def is_success_state(x: str, y: str) -> bool:
        k = _Stub.canonical_state_key(x, y)
        return len(k) == 1 and list(k)[0] == 24

    @staticmethod
    def propose_prompt_wrap(x: str, y: str) -> str:
        return "prompt"

    @staticmethod
    def test_output_modify(idx, output):
        return {"r": 0}, output


class TestRunnerSolverResume(unittest.TestCase):

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_solver_solve_with_task_recorder(self) -> None:
        from federated_mcts.federation.orchestrator import FederatedSolver
        from federated_mcts.evaluation.game24_task_recorder import (
            TaskSessionRecorder)

        task = _Stub()
        solver = object.__new__(FederatedSolver)
        solver.args = MagicMock()
        solver.args.n_generate_sample = 1; solver.args.prompt_sample = None
        solver.args.search_policy = "baseline"
        solver.args.method_generate = "propose"
        solver.args.method_evaluate = "vote"
        solver.args.method_select = "greedy"
        solver.args.n_evaluate_sample = 1; solver.args.n_select_sample = 1
        solver.args.game24_exact_prune = False
        solver.args.check_format = False

        out = ["1+1=2 (left: 2 8)", "2+8=10 (left: 10)",
               "10+14=24 (left: 24)"]
        cc = [0]

        def fc(a, p, n=1, stop=None, **kw):
            r = out[min(cc[0], len(out)-1)]; cc[0] += 1; return [r]

        solver.gpts = {"remote_client": fc}
        solver.time_tracker = MagicMock()

        rec = TaskSessionRecorder(task_id="0",
            task_input=task.get_input(0),
            decision_policy="local_validity")

        with patch("federated_mcts.federation.standard_solvers.get_votes",
                   side_effect=lambda a, t, x, ys, n, **kw: [5] * len(ys)):
            ys, info = solver.solve(task, 0, to_print=False,
                                    task_recorder=rec,
                                    solve_client="remote_client")

        self.assertEqual(len(ys), 1)
        m = rec.to_task_metrics()
        self.assertEqual(len(m["step_details"]), 3)
        lids = set()
        for sd in m["step_details"]:
            for c in sd["candidates"]:
                lids.add(c["lineage_id"])
        self.assertEqual(len(lids), 3)

    def test_five_task_journal_resume(self) -> None:
        from federated_mcts.federation.orchestrator import FederatedSolver
        from federated_mcts.evaluation.game24_task_recorder import (
            TaskSessionRecorder)
        from federated_mcts.evaluation.game24_persistence import (
            AtomicTaskJournal, ResumeValidator)

        task = _Stub()
        out = ["1+1=2", "2+8=10", "10+14=24"]

        def make_solver():
            sol = object.__new__(FederatedSolver)
            sol.args = MagicMock()
            sol.args.n_generate_sample = 1; sol.args.prompt_sample = None
            sol.args.search_policy = "baseline"
            sol.args.method_generate = "propose"
            sol.args.method_evaluate = "vote"
            sol.args.method_select = "greedy"
            sol.args.n_evaluate_sample = 1; sol.args.n_select_sample = 1
            sol.args.game24_exact_prune = False
            sol.args.check_format = False
            fc = [0]
            def fk(a, p, n=1, sm=None, **kw):
                r = out[min(fc[0], len(out)-1)]; fc[0] += 1; return [r]
            sol.gpts = {"remote_client": fk}
            sol.time_tracker = MagicMock()
            return sol

        jp = os.path.join(self.tmpdir, "j.json")
        journal = AtomicTaskJournal(jp)
        journal.write_meta(run_fingerprint="fp-resume")

        # Uninterrupted ref: 5 tasks
        ref = []
        for i in range(5):
            s = make_solver()
            rec = TaskSessionRecorder(task_id=str(i),
                task_input=task.get_input(i),
                decision_policy="local_validity")
            with patch("federated_mcts.federation.standard_solvers.get_votes",
                       side_effect=lambda a, t, x, ys, n, **kw: [5] * len(ys)):
                s.solve(task, i, to_print=False, task_recorder=rec,
                        solve_client="remote_client")
            ref.append(rec.to_task_metrics())

        # Persist first 2
        for i in range(2):
            journal.write_task(str(i),
                {"task_id": i, "_completed": True,
                 "task_metrics_json": json.dumps(ref[i])})

        # Resume
        v = ResumeValidator(journal, "fp-resume")
        can, skip, _ = v.validate_resume([0, 1, 2, 3, 4])
        self.assertTrue(can)
        self.assertEqual(skip, {"0", "1"})

        # Run remaining 3
        res = []
        for i in [2, 3, 4]:
            s = make_solver()
            rec = TaskSessionRecorder(task_id=str(i),
                task_input=task.get_input(i),
                decision_policy="local_validity")
            with patch("federated_mcts.federation.standard_solvers.get_votes",
                       side_effect=lambda a, t, x, ys, n, **kw: [5] * len(ys)):
                s.solve(task, i, to_print=False, task_recorder=rec,
                        solve_client="remote_client")
            res.append(rec.to_task_metrics())

        for i, (r, rr) in enumerate(zip(ref[2:], res)):
            self.assertEqual(r["total_proposed"], rr["total_proposed"])
            self.assertEqual(r["total_selected"], rr["total_selected"])


if __name__ == "__main__":
    unittest.main()
