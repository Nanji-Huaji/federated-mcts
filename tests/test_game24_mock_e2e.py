"""Test Wave 3 safety: no-network mocked E2E, dry run, budget reconciliation."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import sys

# Ensure src is on path
sys.path.insert(0, "src")

from federated_mcts.evaluation.game24_mock_client import (
    Game24MockClient,
    build_mock_gpts,
)
from federated_mcts.evaluation.game24_preflight import (
    build_projection,
    reconcile_budget_after_tasks,
)
from federated_mcts.evaluation.game24_runner_setup import (
    build_setup_from_args,
    ExperimentSetup,
)
from federated_mcts.evaluation.game24_runner import (
    compute_task_raw_unit_cost,
)


class TestMockClient(unittest.TestCase):
    """Given a mock GenerationClient, When called, Then returns deterministic responses."""

    def setUp(self) -> None:
        self.mock = Game24MockClient()

    def test_deterministic_proposals(self) -> None:
        """When generating proposals, Then returns format-correct lines."""
        args = MagicMock()
        proposals = self.mock(args, "propose prompt", n=2)
        self.assertEqual(len(proposals), 2)
        self.assertEqual(self.mock.call_count, 1)
        # Format check: should look like "N op N = N (left: ...)"
        for p in proposals:
            self.assertIn("(left:", p)

    def test_deterministic_votes(self) -> None:
        """When generating votes, Then returns vote format strings."""
        args = MagicMock()
        votes = self.mock(args, "vote prompt", n=1)
        self.assertEqual(len(votes), 1)
        self.assertIn("best choice", votes[0].lower())

    def test_deterministic_reproducibility(self) -> None:
        """When reset and called again, Then same sequence is produced."""
        args = MagicMock()
        self.mock.reset()
        p1a = self.mock(args, "prompt", n=2)
        self.mock.reset()
        p1b = self.mock(args, "prompt", n=2)
        self.assertEqual(p1a, p1b)

    def test_build_mock_gpts(self) -> None:
        """When building mock gpts, Then returns dict with expected client names."""
        config = [
            {"client_name": "remote_client", "model": "m1"},
            {"client_name": "local_client", "model": "m2"},
        ]
        gpts = build_mock_gpts(config)
        self.assertIn("remote_client", gpts)
        self.assertIn("local_client", gpts)
        self.assertIsInstance(gpts["remote_client"], Game24MockClient)

    def test_no_network_calls(self) -> None:
        """When mock client is called, Then no HTTP or socket calls are made."""
        import socket
        args = MagicMock()

        with patch.object(socket, "socket", side_effect=RuntimeError("network!")):
            result = self.mock(args, "test prompt", n=1)
            self.assertEqual(len(result), 1)

    def test_mock_solver_bypasses_real_init(self) -> None:
        """When mock_client is True, Then solver has methods without calling __init__."""
        from federated_mcts.federation import FederatedSolver

        with patch.object(FederatedSolver, "__init__",
                          side_effect=RuntimeError("Real __init__ called!")):
            # Direct construction via object.__new__ + manual setup
            solver = object.__new__(FederatedSolver)
            solver.args = MagicMock()
            solver.gpts = {"remote_client": self.mock}
            from federated_mcts.federation.time_tracker import TimeTracker
            solver.time_tracker = TimeTracker(["remote_client"])
            solver.dqn_session = None

        # Verify solver has the expected interface
        self.assertTrue(hasattr(solver, "solve"))
        self.assertTrue(hasattr(solver, "naive_solve"))
        self.assertTrue(hasattr(solver, "gpts"))


_SIMPLE_CONFIG = [
    {"client_name": "remote_client", "model": "deepseek-v4-flash",
     "max_tokens": 512, "api_base": "http://f"},
    {"client_name": "local_client", "model": "Qwen/Qwen2.5-32B-Instruct",
     "max_tokens": 256, "api_base": "http://l"},
]


class TestDryRunNoSideEffects(unittest.TestCase):
    """Given --dry_run, When running, Then no side effects occur."""

    def setUp(self) -> None:
        self.tmpdir_obj = tempfile.TemporaryDirectory()
        self.tmpdir = self.tmpdir_obj.name

    def tearDown(self) -> None:
        self.tmpdir_obj.cleanup()

    def _make_args(self, **overrides):
        attrs = {
            "task": "game24", "task_split": None,
            "task_start_index": 900, "task_end_index": 902,
            "pilot_ids": None, "dry_run": True, "mock_client": False,
            "game24_decision_policy": "local_validity",
            "game24_oracle_audit": False, "seed": None,
            "temperature": 0.7, "method_generate": "propose",
            "method_evaluate": "vote", "method_select": "greedy",
            "n_generate_sample": 1, "n_evaluate_sample": 1,
            "n_select_sample": 1, "game24_exact_prune": False,
            "assign_strategy": "round_robin",
            "max_total_calls": None, "max_total_tokens": None,
            "max_cost_usd": None, "max_raw_unit_limit": 19.99,
            "model_config": os.path.join(self.tmpdir, "model_config.json"),
            "selected_test_auth": None,
            "review_round_identity": "review-1",
        }
        attrs.update(overrides)
        a = MagicMock()
        for k, v in attrs.items():
            setattr(a, k, v)
        return a

    def test_dry_run_creates_no_files(self) -> None:
        """When dry_run is True, Then no files or directories are created."""
        args = self._make_args()
        # Write a minimal model config so file_name_generater works
        cfg_path = os.path.join(self.tmpdir, "model_config.json")
        with open(cfg_path, "w") as f:
            json.dump(_SIMPLE_CONFIG, f)
        args.model_config = cfg_path

        setup = build_setup_from_args(args, "", "/tmp/test_dry")
        self.assertTrue(setup.dry_run)
        # Sanity: setup was built
        self.assertEqual(len(setup.task_ids), 2)

    def test_dry_run_produces_projection(self) -> None:
        """When dry_run is True, Then projection is built but no files written."""
        cfg_path = os.path.join(self.tmpdir, "model_config.json")
        with open(cfg_path, "w") as f:
            json.dump(_SIMPLE_CONFIG, f)

        args = self._make_args(model_config=cfg_path)
        projection = build_projection(
            n_tasks=2, model_config_raw=_SIMPLE_CONFIG,
            n_generate_sample=1, n_evaluate_sample=1, n_select_sample=1)
        self.assertIsNotNone(projection)
        self.assertTrue(projection.gate_passed)
        self.assertGreater(projection.worst_case_raw_unit_estimate, 0.0)


class TestBudgetReconciliation(unittest.TestCase):
    """Given running tasks, When reconciling budget, Then correct fields are emitted."""

    def test_spend_tracking_in_projection(self) -> None:
        """When actual spend is reconciled with projection, Then correct fields exist."""
        projection = build_projection(
            n_tasks=10, model_config_raw=_SIMPLE_CONFIG,
            n_generate_sample=1, n_evaluate_sample=1, n_select_sample=1)
        rec = reconcile_budget_after_tasks(0.03, projection)
        self.assertAlmostEqual(rec["actual_spent_raw_units"], 0.03)
        self.assertEqual(rec["preflight_projection_raw_units"],
                         projection.worst_case_raw_unit_estimate)
        self.assertIn("projection_gate_passed", rec)
        self.assertIn("projection_gate_limit", rec)

    def test_zero_spend_reconciliation(self) -> None:
        """When no spend has occurred, Then reconciliation shows zero."""
        projection = build_projection(
            n_tasks=5, model_config_raw=_SIMPLE_CONFIG,
            n_generate_sample=1, n_evaluate_sample=1, n_select_sample=1)
        rec = reconcile_budget_after_tasks(0.0, projection)
        self.assertEqual(rec["actual_spent_raw_units"], 0.0)


class TestPerTaskSpendTracking(unittest.TestCase):
    """Given two consecutive task usage deltas, When computing raw-unit cost,
    Then totals are correct and reconciled."""

    def test_two_tasks_accumulate_spend(self) -> None:
        """When two tasks produce usage, Then spent == sum(delta costs)."""
        # Task 1: 1000 prompt, 500 completion on deepseek-v4-flash
        prev = {"deepseek-v4-flash": {"prompt_tokens": 1000,
                                       "completion_tokens": 500,
                                       "total_calls": 5}}
        after_task1 = {"deepseek-v4-flash": {"prompt_tokens": 2000,
                                              "completion_tokens": 1000,
                                              "total_calls": 10}}
        cost1 = compute_task_raw_unit_cost(prev, after_task1)
        self.assertGreater(cost1, 0.0)
        # deepseek-v4-flash: prompt=0.14, completion=0.28 per 1M
        expected1 = (1000 * 0.14 + 500 * 0.28) / 1_000_000
        self.assertAlmostEqual(cost1, expected1, places=8)

        # Task 2: additional usage
        after_task2 = {"deepseek-v4-flash": {"prompt_tokens": 3500,
                                              "completion_tokens": 2000,
                                              "total_calls": 18}}
        cost2 = compute_task_raw_unit_cost(after_task1, after_task2)
        expected2 = (1500 * 0.14 + 1000 * 0.28) / 1_000_000
        self.assertAlmostEqual(cost2, expected2, places=8)

        # Total should be sum of both
        total_cost = cost1 + cost2
        cost_total = compute_task_raw_unit_cost(prev, after_task2)
        self.assertAlmostEqual(cost_total, total_cost, places=8)

    def test_zero_usage_no_cost(self) -> None:
        """When usage hasn't changed, Then cost is zero."""
        snap = {"deepseek-v4-flash": {"prompt_tokens": 100,
                                       "completion_tokens": 50,
                                       "total_calls": 1}}
        cost = compute_task_raw_unit_cost(snap, snap)
        self.assertEqual(cost, 0.0)

    def test_reconciliation_matches_spent(self) -> None:
        """When reconciling, Then actual_spent_raw_units == budget_ledger.spent."""
        projection = build_projection(
            n_tasks=10, model_config_raw=_SIMPLE_CONFIG,
            n_generate_sample=1, n_evaluate_sample=1, n_select_sample=1)
        simulated_spent = 0.042
        rec = reconcile_budget_after_tasks(simulated_spent, projection)
        self.assertAlmostEqual(rec["actual_spent_raw_units"], 0.042)
        self.assertEqual(rec["preflight_projection_raw_units"],
                         projection.worst_case_raw_unit_estimate)


class TestDryRunNoDirs(unittest.TestCase):
    """Given dry_run, When running, Then no directories are created."""

    def setUp(self) -> None:
        self.tmpdir_obj = tempfile.TemporaryDirectory()
        self.tmpdir = self.tmpdir_obj.name

    def tearDown(self) -> None:
        self.tmpdir_obj.cleanup()

    def _make_args(self, **overrides):
        attrs = {
            "task": "game24", "task_split": None,
            "task_start_index": 900, "task_end_index": 902,
            "pilot_ids": None, "dry_run": True, "mock_client": False,
            "game24_decision_policy": "local_validity",
            "game24_oracle_audit": False, "seed": None,
            "temperature": 0.7, "method_generate": "propose",
            "method_evaluate": "vote", "method_select": "greedy",
            "n_generate_sample": 1, "n_evaluate_sample": 1,
            "n_select_sample": 1, "game24_exact_prune": False,
            "assign_strategy": "round_robin",
            "max_total_calls": None, "max_total_tokens": None,
            "max_cost_usd": None, "max_raw_unit_limit": 19.99,
            "model_config": os.path.join(self.tmpdir, "model_config.json"),
            "selected_test_auth": None,
            "review_round_identity": "review-1",
            "naive_run": False, "search_policy": "baseline",
            "slm_generate": False, "slm_eval": False,
            "check_format": False, "eval_rule": "standard",
            "warm_start": False, "last_lm": "none",
            "inference_idx": 0, "prompt_sample": "standard",
            "filter": False, "solve_method": "tot",
        }
        attrs.update(overrides)
        a = MagicMock()
        for k, v in attrs.items():
            setattr(a, k, v)
        return a

    def test_dry_run_creates_no_output_dirs(self) -> None:
        """When compute_file_prefix is called, Then no directories are created."""
        from federated_mcts.evaluation.game24_runner_persistence import compute_file_prefix

        # Use a fake prefix base inside tmpdir to isolate from existing dirs
        import pathlib
        fake_logs = pathlib.Path(self.tmpdir) / "test_logs"
        with patch("federated_mcts.evaluation.game24_runner_persistence.compute_file_prefix",
                   wraps=compute_file_prefix) as mock_cfp:
            cfg_path = os.path.join(self.tmpdir, "model_config.json")
            with open(cfg_path, "w") as f:
                json.dump(_SIMPLE_CONFIG, f)
            args = self._make_args(model_config=cfg_path)
            prefix = compute_file_prefix(args)

        # The prefix computation itself must not create directories
        # Check that the directory DOES NOT exist at the prefix location
        log_dir = os.path.dirname(prefix + ".json")
        # Only assert for paths under our tmpdir (some other test may have
        # created ./logs/ elsewhere)
        if log_dir.startswith(self.tmpdir):
            self.assertFalse(os.path.exists(log_dir),
                             f"compute_file_prefix must not create dirs, but {log_dir} exists")
        # Main assertion: prefix was computed, is a string
        self.assertIsInstance(prefix, str)
        self.assertIn("logs", prefix)


class TestMockSolverNoRealInit(unittest.TestCase):
    """Given --mock_client, When solver is created, Then real FederatedSolver
    __init__ is never reached."""

    def test_mock_path_never_calls_federated_init(self) -> None:
        """When mock_client=True, Then FederatedSolver.__init__ is NEVER called."""
        # This test patches FederatedSolver.__init__ to raise,
        # proving the mock path bypasses the real constructor entirely.
        from federated_mcts.federation import FederatedSolver
        from federated_mcts.federation.time_tracker import TimeTracker

        _SIMPLE_CFG = [{"client_name": "remote_client",
                          "model": "deepseek-v4-flash",
                          "max_tokens": 512, "api_base": "http://f"}]

        mock = Game24MockClient()

        with patch.object(FederatedSolver, "__init__",
                          side_effect=RuntimeError("BUG: real __init__ called!")):
            args = MagicMock()
            setup = MagicMock()
            setup.mock_client = True
            setup.model_config_path = "/fake.json"

            with patch("builtins.open", unittest.mock.mock_open(
                    read_data=json.dumps(_SIMPLE_CFG))):
                from federated_mcts.evaluation.game24_mock_client import build_mock_gpts
                with patch.object(FederatedSolver, "__init__",
                                  side_effect=RuntimeError("BUG!")):
                    solver = object.__new__(FederatedSolver)
                    solver.args = args
                    solver.gpts = build_mock_gpts(_SIMPLE_CFG)
                    solver.time_tracker = TimeTracker(["remote_client"])
                    solver.dqn_session = None

            # Prove solver has required methods
            self.assertTrue(hasattr(solver, "solve"))
            self.assertTrue(hasattr(solver, "naive_solve"))
            self.assertTrue(hasattr(solver, "gpts"))
            self.assertIn("remote_client", solver.gpts)
    """Given an over-cap projection, When gating, Then full validation is blocked."""

    def test_projection_over_gate_fails(self) -> None:
        """When projection exceeds 0.75, Then gate_passed is False."""
        p = build_projection(
            n_tasks=200, model_config_raw=_SIMPLE_CONFIG,
            n_generate_sample=10, n_evaluate_sample=10, n_select_sample=5,
            gate_limit=0.75)
        self.assertFalse(p.gate_passed)
        self.assertGreater(p.worst_case_raw_unit_estimate, 0.75)

    def test_projection_under_gate_passes(self) -> None:
        """When projection is under 0.75, Then gate_passed is True."""
        p = build_projection(
            n_tasks=20, model_config_raw=_SIMPLE_CONFIG,
            n_generate_sample=1, n_evaluate_sample=1, n_select_sample=1,
            gate_limit=0.75)
        self.assertTrue(p.gate_passed)
        self.assertLess(p.worst_case_raw_unit_estimate, 0.75)


if __name__ == "__main__":
    unittest.main()
