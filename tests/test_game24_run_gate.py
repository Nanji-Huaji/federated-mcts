"""Real canonical-runner gate: commit interrupt, resume identity, fail-closed resume.

These tests drive ``federated_mcts.experiments.runner.prepare``/``execute``.
Clients are injected, but the solver is constructed through its normal
constructor and executes the real search path; no ``__init__`` bypass is used.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from dataclasses import replace
from unittest.mock import patch

sys.path.insert(0, "src")

from federated_mcts.evaluation.game24_persistence import (
    AtomicTaskJournal,
    TaskCommitter,
)
from federated_mcts.evaluation.game24_runner_args import parse_args
from federated_mcts.experiments import runner
from federated_mcts.federation import FederatedSolver


class _InjectedInterruption(BaseException):
    """Controlled interruption that propagates past ``except Exception``."""


class _StatelessClient:
    """Callable generation client with no cross-call state.

    Determinism across a resumed and an uninterrupted run is required to prove
    resume identity; a stateful mock would replay a different response sequence
    after the skipped prefix.
    """

    def __call__(self, args, prompt, n=1, stop=None, get_logprobs=False):
        if "vote" in (prompt or "").lower():
            return ["The best choice is 0"] * n
        return ["8 - 1 = 7 (left: 1 1 7)"] * n


class _RecordingSolver(FederatedSolver):
    """A real solver that records which task IDs actually reached ``solve``."""

    instances: list["_RecordingSolver"] = []

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.solved_ids: list[int] = []
        type(self).instances.append(self)

    def solve(self, task, idx, **kwargs):
        self.solved_ids.append(idx)
        return super().solve(task, idx, **kwargs)


class _InterruptingCommitter(TaskCommitter):
    """Settle two tasks, then interrupt after durably recording the third."""

    limit = 2
    calls = 0

    def _charge(self, *args, **kwargs):
        type(self).calls += 1
        if type(self).calls > type(self).limit:
            raise _InjectedInterruption("injected after pending journal")
        return super()._charge(*args, **kwargs)


class TestRealRunnerGate(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmpdir = self._tmp.name
        self._cwd = os.getcwd()
        os.chdir(self.tmpdir)
        os.makedirs("configs")
        os.makedirs("data/24")
        os.makedirs("data/splits")
        with open("configs/model_config.json", "w", encoding="utf-8") as stream:
            json.dump(
                [
                    {
                        "client_name": "local_client",
                        "model": "Qwen/Qwen2.5-32B-Instruct",
                        "max_tokens": 256,
                        "api_key": "k",
                        "api_base": "http://127.0.0.1:9/v1",
                    },
                    {
                        "client_name": "remote_client",
                        "model": "deepseek-v4-flash",
                        "max_tokens": 512,
                        "api_key": "k",
                        "api_base": "http://127.0.0.1:9/v1",
                    },
                ],
                stream,
            )
        with open("data/24/24.csv", "w", encoding="utf-8") as stream:
            stream.write("Puzzles\n1 1 1 8\n2 2 2 6\n3 3 3 3\n4 4 4 4\n5 5 5 5\n")
        with open("data/splits/game24.json", "w", encoding="utf-8") as stream:
            json.dump({}, stream)

        _RecordingSolver.instances = []
        _InterruptingCommitter.calls = 0

    def tearDown(self) -> None:
        os.chdir(self._cwd)
        self._tmp.cleanup()

    @staticmethod
    def _args(*, resume: bool = False):
        argv = [
            "--task", "game24",
            "--task_start_index", "0",
            "--task_end_index", "5",
            "--model_config", "configs/model_config.json",
            "--solve_method", "tot",
            "--method_generate", "propose",
            "--method_evaluate", "vote",
            "--method_select", "greedy",
            "--n_generate_sample", "1",
            "--n_evaluate_sample", "1",
            "--n_select_sample", "1",
            "--check_format",
        ]
        if resume:
            argv.append("--resume")
        return parse_args(argv)

    def _prepare(self):
        # Projection construction has dedicated tests; these tests own the
        # commit/resume gate.
        with patch.object(runner, "build_preflight", return_value=None):
            return runner.prepare(self._args(), "tot")

    def _factory(self, args, setup, solver_cls=None):
        solver = _RecordingSolver(
            args,
            clients={
                "remote_client": _StatelessClient(),
                "local_client": _StatelessClient(),
            },
        )
        self.assertIsInstance(solver, FederatedSolver)
        return solver

    def _execute(self, plan, committer_cls=TaskCommitter):
        with patch.object(runner, "setup_mock_solver", side_effect=self._factory):
            with patch.object(runner, "TaskCommitter", committer_cls):
                with contextlib.redirect_stdout(io.StringIO()):
                    return runner.execute(plan)

    def test_true_interrupt_resume_identity(self) -> None:
        plan = self._prepare()
        self.assertIsNotNone(plan)
        base = plan.setup.file_name

        try:
            self._execute(plan, _InterruptingCommitter)
        except _InjectedInterruption:
            pass
        else:
            self.fail("injected commit interruption did not propagate")

        self.assertEqual(_RecordingSolver.instances[0].solved_ids, [0, 1, 2])
        journal = AtomicTaskJournal(base + "_journal.json")
        self.assertEqual(journal.completed_ids(), {"0", "1"})
        self.assertFalse(os.path.exists(base + "_performance.json"))
        self.assertFalse(os.path.exists(base + ".json"))

        resumed = self._execute(replace(plan, resume=True))
        self.assertEqual(_RecordingSolver.instances[1].solved_ids, [3, 4])
        self.assertEqual(
            journal.completed_ids(), {"0", "1", "2", "3", "4"}
        )
        self.assertEqual(resumed["task_count"], 5)

        full_plan = replace(
            plan, setup=replace(plan.setup, file_name=base + "_full")
        )
        self._execute(full_plan)

        with open(base + ".json", encoding="utf-8") as stream:
            resumed_json = json.load(stream)
        with open(base + "_full.json", encoding="utf-8") as stream:
            full_json = json.load(stream)
        self.assertEqual(
            [record["idx"] for record in resumed_json], [0, 1, 2, 3, 4]
        )
        self.assertEqual(
            json.dumps(resumed_json, sort_keys=True, indent=2),
            json.dumps(full_json, sort_keys=True, indent=2),
            "resumed and uninterrupted results must be identical",
        )

        with open(base + "_performance.json", encoding="utf-8") as stream:
            resumed_perf = json.load(stream)
        with open(base + "_full_performance.json", encoding="utf-8") as stream:
            full_perf = json.load(stream)
        # Wall-clock-independent: the resumed run starts two tasks later.
        for payload in (resumed_perf, full_perf):
            for key in ("started_tasks", "budget_remaining", "budget_spent"):
                payload.pop(key, None)
        self.assertEqual(
            json.dumps(resumed_perf, sort_keys=True, indent=2),
            json.dumps(full_perf, sort_keys=True, indent=2),
            "resumed and uninterrupted performance records must match",
        )

        resumed_records = journal.read_all()
        full_records = AtomicTaskJournal(
            base + "_full_journal.json"
        ).read_all()
        self.assertEqual(
            resumed_records["_meta"]["run_fingerprint"],
            full_records["_meta"]["run_fingerprint"],
        )
        for task_id in ("0", "1", "2", "3", "4"):
            left = dict(resumed_records[task_id])
            right = dict(full_records[task_id])
            for record in (left, right):
                record.pop("elapsed_ms", None)
                record.pop("task_metrics_json", None)
            self.assertEqual(left, right, f"journal record {task_id} differs")

    def test_mismatched_resume_leaves_run_artifacts_untouched(self) -> None:
        plan = self._prepare()
        self.assertIsNotNone(plan)
        base = replace(plan.setup, file_name=plan.setup.file_name + "_mismatch")
        plan = replace(plan, setup=base)
        file_name = base.file_name

        journal = AtomicTaskJournal(file_name + "_journal.json")
        journal.write_meta(run_fingerprint="stored-fingerprint")
        journal.write_task("0", {"task_id": 0, "_completed": True})
        artifact_paths = [
            file_name + suffix
            for suffix in (
                "_journal.json",
                "_manifest.json",
                "_preflight.json",
                ".json",
                "_performance.json",
                "_budget.jsonl",
                "_authorization.jsonl",
            )
        ]
        for path in artifact_paths[1:]:
            with open(path, "wb") as artifact:
                artifact.write(b"sentinel")

        def read_artifacts() -> dict[str, bytes]:
            contents: dict[str, bytes] = {}
            for path in artifact_paths:
                with open(path, "rb") as artifact:
                    contents[path] = artifact.read()
            return contents

        before = read_artifacts()
        resume_plan = replace(plan, resume=True)
        with patch.object(runner, "setup_budget_ledger") as ledger:
            with patch.object(runner, "consume_authorization") as auth:
                with patch.object(runner, "setup_mock_solver") as solver:
                    with contextlib.redirect_stderr(io.StringIO()):
                        with self.assertRaises(SystemExit):
                            runner.execute(resume_plan)
        self.assertEqual(before, read_artifacts())
        ledger.assert_not_called()
        auth.assert_not_called()
        solver.assert_not_called()


if __name__ == "__main__":
    unittest.main()
