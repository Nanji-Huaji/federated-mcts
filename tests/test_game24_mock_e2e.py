"""Offline mocked E2E safety: mock clients, dry run, budget accounting.

The mock client path is exercised through the normal solver constructor: the
runner builds real ``FederatedSolver`` instances with ``Game24MockClient``
adapters.  Nothing here bypasses ``FederatedSolver.__init__``.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from decimal import Decimal
from unittest.mock import MagicMock, patch

sys.path.insert(0, "src")

from federated_mcts.evaluation.game24_mock_client import (
    Game24MockClient,
    build_mock_gpts,
)
from federated_mcts.evaluation.game24_preflight import (
    PreflightProjection,
    reconcile_budget_after_tasks,
)
from federated_mcts.evaluation.game24_runner import (
    build_manifest,
    compute_task_raw_unit_cost,
)
from federated_mcts.evaluation.game24_runner_orch import setup_mock_solver
from federated_mcts.evaluation.game24_runner_persistence import (
    compute_file_prefix,
)
from federated_mcts.evaluation.game24_runner_setup import build_setup_from_args
from federated_mcts.federation import FederatedSolver


_SIMPLE_CONFIG = [
    {
        "client_name": "remote_client",
        "model": "deepseek-v4-flash",
        "max_tokens": 512,
        "api_base": "http://127.0.0.1:9/v1",
    },
    {
        "client_name": "local_client",
        "model": "Qwen/Qwen2.5-32B-Instruct",
        "max_tokens": 256,
        "api_base": "http://127.0.0.1:9/v1",
    },
]


def _args(**overrides):
    attrs = {
        "task": "game24",
        "task_split": None,
        "task_start_index": 900,
        "task_end_index": 903,
        "pilot_ids": None,
        "dry_run": True,
        "mock_client": True,
        "game24_decision_policy": "local_validity",
        "game24_oracle_audit": False,
        "seed": None,
        "temperature": 0.7,
        "method_generate": "propose",
        "method_evaluate": "vote",
        "method_select": "greedy",
        "n_generate_sample": 1,
        "n_evaluate_sample": 1,
        "n_select_sample": 1,
        "game24_exact_prune": False,
        "assign_strategy": "round_robin",
        "max_total_calls": None,
        "max_total_tokens": None,
        "max_cost_usd": None,
        "max_raw_unit_limit": 19.99,
        "model_config": "configs/model_config.json",
        "selected_test_auth": None,
        "review_round_identity": "review-1",
        "naive_run": False,
        "search_policy": "baseline",
        "slm_generate": False,
        "slm_eval": False,
        "check_format": False,
        "eval_rule": "standard",
        "warm_start": False,
        "last_lm": "none",
        "inference_idx": 0,
        "prompt_sample": "standard",
        "filter": False,
        "solve_method": "tot",
        "localbackend": "Qwen/Qwen2.5-32B-Instruct",
        "remotebackend": "deepseek-v4-flash",
    }
    attrs.update(overrides)
    holder = MagicMock()
    for key, value in attrs.items():
        setattr(holder, key, value)
    return holder


def _projection(**overrides) -> PreflightProjection:
    values = {
        "schema_version": 1,
        "n_tasks": 10,
        "model_names": ["deepseek-v4-flash"],
        "model_rates_raw": {"deepseek-v4-flash": {"prompt": 0.14, "completion": 0.28}},
        "solver_config": {"n_generate_sample": 1},
        "worst_case_calls": 100,
        "worst_case_prompt_tokens": 20_000,
        "worst_case_completion_tokens": 10_000,
        "worst_case_raw_unit_estimate": 3.25,
        "gate_limit": 20.0,
        "gate_passed": True,
    }
    values.update(overrides)
    return PreflightProjection(**values)


class TestMockClient(unittest.TestCase):
    """Given a mock GenerationClient, When called, Then responses are deterministic."""

    def setUp(self) -> None:
        self.mock = Game24MockClient()

    def test_deterministic_proposals(self) -> None:
        args = MagicMock()
        proposals = self.mock(args, "propose prompt", n=2)
        self.assertEqual(len(proposals), 2)
        self.assertEqual(self.mock.call_count, 1)
        for proposal in proposals:
            self.assertIn("(left:", proposal)

    def test_deterministic_votes(self) -> None:
        args = MagicMock()
        votes = self.mock(args, "vote prompt", n=1)
        self.assertEqual(len(votes), 1)
        self.assertIn("best choice", votes[0].lower())

    def test_deterministic_reproducibility(self) -> None:
        args = MagicMock()
        self.mock.reset()
        first = self.mock(args, "prompt", n=2)
        self.mock.reset()
        second = self.mock(args, "prompt", n=2)
        self.assertEqual(first, second)

    def test_no_network_calls(self) -> None:
        args = MagicMock()
        with patch.object(
            __import__("socket"), "socket", side_effect=RuntimeError("network!")
        ):
            result = self.mock(args, "test prompt", n=1)
        self.assertEqual(len(result), 1)


class TestBuildMockGpts(unittest.TestCase):
    def test_build_mock_gpts(self) -> None:
        gpts = build_mock_gpts(_SIMPLE_CONFIG)
        self.assertEqual(
            set(gpts), {"remote_client", "local_client"}
        )
        self.assertIsInstance(gpts["remote_client"], Game24MockClient)


class TestMockSolverConstruction(unittest.TestCase):
    """Given --mock_client, Then the normal solver constructor receives adapters."""

    def test_setup_mock_solver_constructs_real_solver(self) -> None:
        args = _args(mock_client=True, dry_run=False)
        setup = build_setup_from_args(
            args, "", "", model_configs=_SIMPLE_CONFIG
        )

        solver = setup_mock_solver(args, setup, solver_cls=FederatedSolver)

        self.assertIsInstance(solver, FederatedSolver)
        self.assertEqual(set(solver.gpts), {"remote_client", "local_client"})
        for client in solver.gpts.values():
            self.assertIsInstance(client, Game24MockClient)
        self.assertTrue(hasattr(solver, "solve"))
        self.assertTrue(hasattr(solver, "naive_solve"))


class TestDryRunResolution(unittest.TestCase):
    """Given --dry_run, Then setup/manifest resolution performs no I/O."""

    def test_dry_run_builds_setup_without_side_effects(self) -> None:
        args = _args()
        setup = build_setup_from_args(
            args, "data/splits/game24.json", "", model_configs=_SIMPLE_CONFIG
        )
        self.assertTrue(setup.dry_run)
        self.assertEqual(setup.task_ids, [900, 901, 902])

    def test_dry_run_manifest_keeps_game24_provenance(self) -> None:
        args = _args()
        setup = build_setup_from_args(
            args, "data/splits/game24.json", "", model_configs=_SIMPLE_CONFIG
        )
        manifest = build_manifest(setup)
        self.assertEqual(manifest.task_ids, [900, 901, 902])
        self.assertEqual(
            manifest.split_file_path, "data/splits/game24.json"
        )

    def test_compute_file_prefix_does_not_create_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            previous_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                args = _args(
                    task_start_index=0,
                    task_end_index=2,
                    localbackend="local-model",
                    remotebackend="remote-model",
                )
                prefix = compute_file_prefix(args)
                self.assertIsInstance(prefix, str)
                self.assertIn("logs", prefix)
                self.assertFalse(
                    os.path.exists(os.path.dirname(prefix + ".json"))
                )
                self.assertFalse(os.path.exists(prefix + ".json"))
            finally:
                os.chdir(previous_cwd)


class TestBudgetReconciliation(unittest.TestCase):
    """Given a projection, When reconciling, Then fields match the contract."""

    def test_spend_tracking_in_projection(self) -> None:
        projection = _projection()
        record = reconcile_budget_after_tasks(0.03, projection)
        self.assertAlmostEqual(record["actual_spent_raw_units"], 0.03)
        self.assertEqual(
            record["preflight_projection_raw_units"],
            projection.worst_case_raw_unit_estimate,
        )
        self.assertEqual(record["projection_gate_passed"], True)
        self.assertEqual(record["projection_gate_limit"], 20.0)
        self.assertEqual(record["preflight_schema_version"], 1)

    def test_zero_spend_reconciliation(self) -> None:
        record = reconcile_budget_after_tasks(0.0, _projection(n_tasks=5))
        self.assertEqual(record["actual_spent_raw_units"], 0.0)


class TestPerTaskSpendTracking(unittest.TestCase):
    """Given consecutive usage snapshots, Then exact Decimal costs accumulate."""

    def test_two_tasks_accumulate_spend(self) -> None:
        previous = {
            "deepseek-v4-flash": {
                "prompt_tokens": 1000,
                "completion_tokens": 500,
                "total_calls": 5,
            }
        }
        after_task1 = {
            "deepseek-v4-flash": {
                "prompt_tokens": 2000,
                "completion_tokens": 1000,
                "total_calls": 10,
            }
        }
        cost1 = compute_task_raw_unit_cost(previous, after_task1)
        self.assertIsInstance(cost1, Decimal)
        self.assertEqual(cost1, Decimal("0.00028"))

        after_task2 = {
            "deepseek-v4-flash": {
                "prompt_tokens": 3500,
                "completion_tokens": 2000,
                "total_calls": 18,
            }
        }
        cost2 = compute_task_raw_unit_cost(after_task1, after_task2)
        self.assertEqual(cost2, Decimal("0.00049"))

        total = compute_task_raw_unit_cost(previous, after_task2)
        self.assertEqual(total, cost1 + cost2)

    def test_zero_usage_no_cost(self) -> None:
        snapshot = {
            "deepseek-v4-flash": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_calls": 1,
            }
        }
        self.assertEqual(
            compute_task_raw_unit_cost(snapshot, snapshot), Decimal("0")
        )


if __name__ == "__main__":
    unittest.main()
