"""Todo 7 real runner gate: persist_task interrupt, resume identity."""
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


class _TrackerSolver:
    def __init__(self, args):
        self.args = args
        self.solved_ids: list[int] = []
        self.latency_dict: dict[str, float] = {}
    def naive_solve(self, task, idx, **kw):
        return self._do(task, idx)
    def solve(self, task, idx, **kw):
        return self._do(task, idx)
    def speculative_solve(self, task, idx, **kw):
        return self._do(task, idx)
    def federated_solve(self, task, idx, **kw):
        return self._do(task, idx)
    def _do(self, task, idx):
        self.solved_ids.append(idx)
        return [f"result_{idx}"], {"steps": []}


class TestRealRunnerGate(unittest.TestCase):

    _repo_root: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        cls._repo_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), ".."))

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self._cwd = os.getcwd()
        os.chdir(self.tmpdir)
        os.makedirs("configs", exist_ok=True)
        os.makedirs("data/24", exist_ok=True)
        os.makedirs("data/splits", exist_ok=True)
        os.makedirs("logs/game24/tot/remote-model", exist_ok=True)
        with open("configs/model_config.json", "w") as f:
            json.dump([{"client_name": "local_client", "model": "Qwen/Qwen2.5-32B-Instruct",
                "max_tokens": 256, "api_key": "k", "api_base": "http://t"},
                {"client_name": "remote_client", "model": "deepseek-v4-flash",
                "max_tokens": 512, "api_key": "k", "api_base": "http://t"}], f)
        with open("data/24/24.csv", "w") as f:
            f.write("Puzzles\n1 1 1 8\n2 2 2 6\n3 3 3 3\n4 4 4 4\n5 5 5 5\n")
        with open("data/splits/game24.json", "w") as f:
            json.dump({}, f)

    def tearDown(self) -> None:
        os.chdir(self._cwd)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    @staticmethod
    def _args(start=0, end=5, resume=False):
        a = MagicMock()
        for attr, val in {"task": "game24", "task_split": None,
            "task_start_index": start, "task_end_index": end,
            "naive_run": False, "solve_method": "tot",
            "model_config": "configs/model_config.json",
            "localbackend": "Qwen/Qwen2.5-32B-Instruct", "remotebackend": "deepseek-v4-flash",
            "temperature": 0.7, "method_generate": "propose",
            "method_evaluate": "vote", "method_select": "greedy",
            "n_generate_sample": 1, "n_evaluate_sample": 1,
            "n_select_sample": 1, "search_policy": "baseline",
            "slm_generate": False, "slm_eval": False,
            "check_format": False, "eval_rule": False,
            "warm_start": False, "last_lm": False,
            "inference_idx": 0, "game24_exact_prune": False,
            "max_cost_usd": None, "assign_strategy": "round_robin",
            "prompt_sample": None, "selected_test_auth": None,
            "game24_decision_policy": "local_validity",
            "game24_oracle_audit": False, "seed": None,
            "max_total_calls": None, "max_total_tokens": None,
            "max_raw_unit_limit": 19.99,
            "review_round_identity": "review-1",
            "resume": resume, "filter": False,
            "pilot_ids": None, "dry_run": False, "mock_client": False}.items():
            setattr(a, attr, val)
        return a

    def _load_mr(self):
        scripts_dir = os.path.join(self._repo_root, "scripts")
        spec = importlib.util.spec_from_file_location(
            "_mr_run", os.path.join(scripts_dir, "merged_run.py"))
        assert spec is not None and spec.loader is not None
        mr = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mr)
        return mr

    @staticmethod
    def _fake_manifest(fingerprint="fp-test"):
        m = MagicMock()
        m.to_json.return_value = "{}"
        m.run_fingerprint = fingerprint
        m.decision_policy_fingerprint = "dp-test"
        return m

    def _run(self, mr, file_name, solver, start, end, *,
             resume=False, interrupt_after=0):
        call_count = [0]

        def _intercept(journal_ref, *args, orig=None, **kw):
            if orig is not None:
                orig(journal_ref, *args, **kw)
            call_count[0] += 1
            if interrupt_after and call_count[0] >= interrupt_after:
                from federated_mcts.evaluation.game24_runner_persistence \
                    import InjectedInterruption
                raise InjectedInterruption(
                    f"Injected after {call_count[0]} tasks")

        with patch.object(mr, "compute_file_prefix", return_value=file_name):
            with patch.object(mr, "FederatedSolver", return_value=solver):
                with patch.object(mr, "consume_authorization",
                                  return_value=(True, "")):
                    with patch.object(mr, "build_manifest",
                                      return_value=self._fake_manifest()):
                        with patch.object(mr, "setup_budget_ledger"):
                            with patch.object(mr, "check_budget_caps",
                                              return_value=(True, None)):
                                with patch.object(mr, "enrich_result"):
                                    with patch.object(
                                            mr, "merge_oracle_metrics"):
                                        real_p = mr.persist_task

                                        def _wrapped(*a, **kw):
                                            return _intercept(
                                                *a, **kw, orig=real_p)
                                        with patch.object(
                                                mr, "persist_task",
                                                side_effect=_wrapped):
                                            from federated_mcts.evaluation.game24_runner_persistence \
                                                import InjectedInterruption
                                            try:
                                                mr.run(self._args(
                                                    start, end, resume),
                                                    "tot")
                                            except InjectedInterruption:
                                                pass

    def test_true_interrupt_resume_identity(self) -> None:
        from federated_mcts.evaluation.game24_persistence import (
            AtomicTaskJournal)

        mr = self._load_mr()
        fn_base = "logs/game24/tot/remote-model/_trg"
        jp = fn_base + "_journal.json"

        s1 = _TrackerSolver(None)
        self._run(mr, fn_base, s1, 0, 5, interrupt_after=2)
        self.assertEqual(s1.solved_ids, [0, 1])
        self.assertTrue(os.path.exists(jp))

        j1 = AtomicTaskJournal(jp)
        self.assertEqual(j1.completed_ids(), {"0", "1"})
        # No performance file from interrupted run
        self.assertFalse(os.path.exists(fn_base + "_performance.json"),
                         "No performance file after interrupt")

        s2 = _TrackerSolver(None)
        self._run(mr, fn_base, s2, 0, 5, resume=True)
        self.assertEqual(s2.solved_ids, [2, 3, 4])
        self.assertNotIn(0, s2.solved_ids)
        self.assertNotIn(1, s2.solved_ids)

        s3 = _TrackerSolver(None)
        fn_full = "logs/game24/tot/remote-model/_trg_full"
        self._run(mr, fn_full, s3, 0, 5)
        self.assertEqual(s3.solved_ids, [0, 1, 2, 3, 4])

        j_resume = AtomicTaskJournal(jp)
        j_full = AtomicTaskJournal(fn_full + "_journal.json")
        for tid in ("0", "1", "2", "3", "4"):
            self.assertEqual(
                json.dumps(j_resume.read_all().get(tid, {}), sort_keys=True),
                json.dumps(j_full.read_all().get(tid, {}), sort_keys=True))

        self.assertEqual(j_resume.read_all()["_meta"]["run_fingerprint"],
                         j_full.read_all()["_meta"]["run_fingerprint"])
        self.assertEqual(j_resume.completed_ids(), {"0", "1", "2", "3", "4"})

        # Verify ordinary output JSONs — resume must contain all 5 task IDs
        resumed_json_path = fn_base + ".json"
        self.assertTrue(os.path.exists(resumed_json_path),
                        f"Expected {resumed_json_path} to exist after resume")
        with open(resumed_json_path) as f:
            resumed_json = json.load(f)
        self.assertEqual(len(resumed_json), 5,
                         f"Resumed JSON must have 5 task records, got {len(resumed_json)}")
        resumed_ids = [r.get("idx") for r in resumed_json if isinstance(r, dict)]
        self.assertEqual(resumed_ids, [0, 1, 2, 3, 4],
                         f"Resumed JSON task order must be [0,1,2,3,4], got {resumed_ids}")

        full_json_path = fn_full + ".json"
        self.assertTrue(os.path.exists(full_json_path))
        with open(full_json_path) as f:
            full_json = json.load(f)

        # Normalize nondeterministic fields before comparison
        for records in (resumed_json, full_json):
            for r in records:
                if isinstance(r, dict):
                    r.pop("usage_so_far", None)
                    r.pop("usage_delta", None)
                    r.pop("time_consumption", None)
                    if "steps" in r.get("steps", {}) or "steps" in str(r):
                        pass  # keep step count

        self.assertEqual(
            json.dumps(resumed_json, sort_keys=True, indent=2),
            json.dumps(full_json, sort_keys=True, indent=2),
            "Resumed and uninterrupted ordinary JSON must be byte-identical")

        # Verify performance JSONs
        resumed_perf = fn_base + "_performance.json"
        full_perf = fn_full + "_performance.json"
        self.assertTrue(os.path.exists(resumed_perf))
        self.assertTrue(os.path.exists(full_perf))
        with open(resumed_perf) as f:
            rp = json.load(f)
        with open(full_perf) as f:
            fp = json.load(f)
        rp.pop("complete", None); fp.pop("complete", None)
        rp.pop("stop_reason", None); fp.pop("stop_reason", None)
        rp.pop("budget_remaining", None); fp.pop("budget_remaining", None)
        rp.pop("budget_spent", None); fp.pop("budget_spent", None)
        rp.pop("started_tasks", None); fp.pop("started_tasks", None)
        self.assertEqual(
            json.dumps(rp, sort_keys=True, indent=2),
            json.dumps(fp, sort_keys=True, indent=2),
            "Resumed and uninterrupted performance JSON must match")

    def test_mismatched_resume_leaves_run_artifacts_untouched(self) -> None:
        from federated_mcts.evaluation.game24_persistence import AtomicTaskJournal

        mr = self._load_mr()
        fn_base = "logs/game24/tot/remote-model/_mismatch"
        journal = AtomicTaskJournal(fn_base + "_journal.json")
        journal.write_meta(run_fingerprint="stored-fingerprint")
        journal.write_task("0", {"task_id": 0, "_completed": True})
        artifact_paths = [
            fn_base + suffix for suffix in (
                "_journal.json", "_manifest.json", "_preflight.json", ".json",
                "_performance.json", "_budget.jsonl", "_authorization.jsonl")
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
        with patch.object(mr, "compute_file_prefix", return_value=fn_base):
            with patch.object(mr, "build_manifest",
                              return_value=self._fake_manifest("new-fingerprint")):
                with patch.object(mr, "setup_budget_ledger") as ledger:
                    with patch.object(mr, "consume_authorization") as auth:
                        with patch.object(mr, "FederatedSolver") as solver:
                            with patch.object(mr, "setup_mock_solver") as client:
                                with self.assertRaises(SystemExit):
                                    mr.run(self._args(resume=True), "tot")
        self.assertEqual(before, read_artifacts())
        ledger.assert_not_called()
        auth.assert_not_called()
        solver.assert_not_called()
        client.assert_not_called()


if __name__ == "__main__":
    unittest.main()
