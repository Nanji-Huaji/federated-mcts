"""Runner-path matching-resume budget regression."""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, "src")


class _Solver:
    def __init__(self) -> None:
        self.solved_ids: list[int] = []
        self.latency_dict: dict[str, float] = {}

    def solve(self, task, idx, **kwargs):
        self.solved_ids.append(idx)
        return [f"result_{idx}"], {"steps": []}

    naive_solve = solve
    speculative_solve = solve
    federated_solve = solve


class _Task:
    @staticmethod
    def get_input(idx: int) -> str:
        return "1 1 1 8"

    @staticmethod
    def test_output_modify(idx: int, output: str) -> tuple[dict[str, int], str]:
        return {"r": 0}, output


class TestMatchingResumeBudget(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.cwd = os.getcwd()
        os.chdir(self.tmpdir)
        os.makedirs("configs", exist_ok=True)
        os.makedirs("data/24", exist_ok=True)
        os.makedirs("logs/game24/tot/remote-model", exist_ok=True)
        with open("configs/model_config.json", "w") as config:
            json.dump([], config)
        with open("data/24/24.csv", "w") as puzzles:
            puzzles.write("Puzzles\n1 1 1 8\n2 2 2 6\n3 3 3 3\n")

    def tearDown(self) -> None:
        os.chdir(self.cwd)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    @staticmethod
    def _args() -> MagicMock:
        args = MagicMock()
        values = {
            "task": "game24", "task_split": None, "task_start_index": 0,
            "task_end_index": 3, "naive_run": False, "solve_method": "tot",
            "model_config": "configs/model_config.json", "localbackend": "local",
            "remotebackend": "remote", "temperature": 0.7,
            "method_generate": "propose", "method_evaluate": "vote",
            "method_select": "greedy", "n_generate_sample": 1,
            "n_evaluate_sample": 1, "n_select_sample": 1,
            "search_policy": "baseline", "slm_generate": False,
            "slm_eval": False, "check_format": False, "eval_rule": False,
            "warm_start": False, "last_lm": False, "inference_idx": 0,
            "game24_exact_prune": False, "max_cost_usd": None,
            "assign_strategy": "round_robin", "prompt_sample": None,
            "selected_test_auth": None, "game24_decision_policy": "local_validity",
            "game24_oracle_audit": False, "seed": None, "max_total_calls": None,
            "max_total_tokens": None, "max_raw_unit_limit": 2.0,
            "review_round_identity": "review-1", "resume": True,
            "filter": False, "pilot_ids": None, "dry_run": False, "mock_client": False,
        }
        for name, value in values.items():
            setattr(args, name, value)
        return args

    @staticmethod
    def _manifest() -> MagicMock:
        manifest = MagicMock()
        manifest.to_json.return_value = "{}"
        manifest.run_fingerprint = "matching-fingerprint"
        manifest.decision_policy_fingerprint = "policy-fingerprint"
        return manifest

    @staticmethod
    def _runner():
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        spec = importlib.util.spec_from_file_location(
            "_matching_resume_runner", os.path.join(root, "scripts", "merged_run.py"))
        assert spec is not None and spec.loader is not None
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        return runner

    def test_matching_resume_uses_persisted_remaining_budget(self) -> None:
        from federated_mcts.evaluation.game24_budget import BudgetLedger
        from federated_mcts.evaluation.game24_persistence import AtomicTaskJournal

        file_name = "logs/game24/tot/remote-model/_matching"
        journal = AtomicTaskJournal(file_name + "_journal.json")
        journal.write_meta(run_fingerprint="matching-fingerprint")
        journal.write_task("0", {
            "task_id": 0, "_completed": True,
            "task_info": {"idx": 0, "infos": [{"r": 0}]},
        })
        ledger = BudgetLedger(file_name + "_budget.jsonl", total_limit=2.0)
        self.assertTrue(ledger.spend(1.0))
        solver = _Solver()
        runner = self._runner()
        with patch.object(runner, "compute_file_prefix", return_value=file_name):
            with patch.object(runner, "build_manifest", return_value=self._manifest()):
                with patch.object(runner, "consume_authorization", return_value=(True, "")):
                    with patch.object(runner, "FederatedSolver", return_value=solver):
                        with patch.object(runner, "get_task", return_value=_Task()):
                            with patch.object(runner, "compute_task_raw_unit_cost", return_value=1.0):
                                with patch.object(runner, "enrich_result"):
                                    with patch.object(runner, "merge_oracle_metrics"):
                                        runner.run(self._args(), "tot")
        self.assertEqual(solver.solved_ids, [1])
        self.assertEqual(journal.completed_ids(), {"0", "1"})
        self.assertEqual(
            BudgetLedger(file_name + "_budget.jsonl").remaining, 0.0)


if __name__ == "__main__":
    unittest.main()
