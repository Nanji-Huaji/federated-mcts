"""Real runner accounting across interruptions, using nonzero offline usage."""
from __future__ import annotations

from contextlib import ExitStack, redirect_stdout
from dataclasses import replace
from decimal import Decimal
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from federated_mcts.evaluation.game24_budget import read_verified_ledger
from federated_mcts.evaluation.game24_persistence import AtomicTaskJournal, TaskCommitter
from federated_mcts.evaluation.game24_runner_args import parse_args
from federated_mcts.experiments import runner
from federated_mcts.models.usage import get_usage_tracker


class Interrupted(BaseException):
    pass


class TinyTask:
    def get_input(self, index):
        return str(index)

    def standard_prompt_wrap(self, x, y):
        return x

    def test_output_modify(self, index, output):
        return {"r": int(output == str(index))}, output


class MeteredClient:
    def __init__(self):
        self.calls = []

    def __call__(self, args, prompt, n=1, **kwargs):
        self.calls.append(int(prompt))
        # Exactly 0.0125 raw units per task, no API/network activity.
        get_usage_tracker().record("gpt-4o", completion_tokens=1000, prompt_tokens=1000)
        return [prompt] * n


class TestMatchingResumeBudget(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cwd = os.getcwd()
        os.chdir(self.tmp.name)
        Path("models.json").write_text(json.dumps([
            {"client_name": "remote_client", "model": "gpt-4o"},
        ]))
        self.client = MeteredClient()

    def tearDown(self):
        os.chdir(self.cwd)
        self.tmp.cleanup()

    def plan(self, *, budget="0.025", calls=None, tokens=None):
        argv = ["--task", "text", "--task_start_index", "0", "--task_end_index", "3",
                "--model_config", "models.json", "--solve_method", "naive",
                "--prompt_sample", "standard", "--max_raw_unit_limit", budget]
        if calls is not None:
            argv += ["--max_total_calls", str(calls)]
        if tokens is not None:
            argv += ["--max_total_tokens", str(tokens)]
        plan = runner.prepare(parse_args(argv))
        self.assertIsNotNone(plan)
        return plan

    def execute(self, plan):
        with ExitStack() as stack:
            stack.enter_context(patch.object(runner, "get_task", return_value=TinyTask()))
            stack.enter_context(patch("federated_mcts.models.clients.create_clients",
                                      return_value={"remote_client": self.client}))
            stack.enter_context(redirect_stdout(io.StringIO()))
            return runner.execute(plan)

    def test_matching_resume_uses_persisted_remaining_budget(self):
        for phase in ("after_journal", "after_spend"):
            with self.subTest(phase=phase):
                plan = self.plan()
                plan = replace(plan, setup=replace(plan.setup, file_name=phase))
                self.client.calls.clear()
                fault = (patch.object(TaskCommitter, "_charge", side_effect=Interrupted())
                         if phase == "after_journal" else
                         patch.object(AtomicTaskJournal, "mark_settled", side_effect=Interrupted()))
                with fault, self.assertRaises(Interrupted):
                    self.execute(plan)
                journal = AtomicTaskJournal(phase + "_journal.json")
                self.assertEqual(journal.completed_ids(), set())
                self.assertEqual(journal.read_all()["0"]["cost_raw"], "0.0125")
                resumed = self.execute(replace(plan, resume=True))
                self.assertEqual(self.client.calls, [0, 1])
                self.assertEqual(journal.completed_ids(), {"0", "1"})
                self.assertEqual(read_verified_ledger(phase + "_budget.jsonl").spent, Decimal("0.025"))
                self.assertEqual(resumed["gpt-4o"]["total_calls"], 2)
                self.assertEqual(resumed["gpt-4o"]["total_tokens"], 4000)
                self.assertFalse(resumed["complete"])
                self.assertEqual(resumed["stop_reason"], "max_raw_unit_limit")
                full = replace(plan, setup=replace(plan.setup, file_name=phase + "_full"))
                self.client.calls.clear()
                uninterrupted = self.execute(full)
                self.assertEqual(self.client.calls, [0, 1])
                self.assertEqual(uninterrupted["gpt-4o"], resumed["gpt-4o"])
                self.assertEqual(uninterrupted["budget_spent"], resumed["budget_spent"])
                before = Path(phase + "_budget.jsonl").read_bytes()
                self.execute(replace(plan, resume=True))
                self.assertEqual(Path(phase + "_budget.jsonl").read_bytes(), before)

    def test_fresh_run_does_not_overwrite_an_existing_run(self):
        plan = self.plan()
        self.execute(plan)
        journal = Path(plan.setup.file_name + "_journal.json")
        before = journal.read_bytes()
        self.client.calls.clear()
        with self.assertRaises(SystemExit):
            self.execute(plan)
        self.assertEqual(journal.read_bytes(), before)
        self.assertEqual(self.client.calls, [])

    def test_in_flight_interruption_is_not_silently_rerun(self):
        plan = self.plan()
        with patch.object(MeteredClient, "__call__", side_effect=Interrupted()):
            with self.assertRaises(Interrupted):
                self.execute(plan)
        journal = AtomicTaskJournal(plan.setup.file_name + "_journal.json")
        self.assertEqual(journal.read_all()["0"]["execution_state"], "running")
        with self.assertRaisesRegex(ValueError, "in-flight"):
            self.execute(replace(plan, resume=True))
        self.assertEqual(self.client.calls, [])

    def test_runs_use_separate_usage_scopes(self):
        plan = self.plan(budget="1")
        outside_before = get_usage_tracker().get_summary()
        first = self.execute(plan)
        second = self.execute(replace(plan, setup=replace(plan.setup, file_name="other")))
        self.assertEqual(first["gpt-4o"], second["gpt-4o"])
        self.assertEqual(second["gpt-4o"]["total_calls"], 3)
        self.assertEqual(get_usage_tracker().get_summary(), outside_before)

    def test_last_task_cap_violation_is_not_marked_complete(self):
        plan = self.plan(budget="1", calls=2)
        result = self.execute(plan)
        self.assertEqual(self.client.calls, [0, 1, 2])
        self.assertFalse(result["complete"])
        self.assertEqual(result["stop_reason"], "max_total_calls")


if __name__ == "__main__":
    unittest.main()
