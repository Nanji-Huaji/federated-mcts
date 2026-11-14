"""Tests for B0 gate preflight deferral.

Covers: skip_full_val_gate=True defers static gate, default no-skip
exits on gate fail, dry_run + skip flag returns False, skip flag
no-op on pilot runs, and passing projection unaffected by skip flag.
"""

from __future__ import annotations

import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, "src")

from federated_mcts.evaluation.game24_runner_orch import (
    run_preflight_and_dry_run,
)
from federated_mcts.evaluation.game24_manifest import Game24Manifest
from federated_mcts.evaluation.game24_preflight import PreflightProjection


class TestPreflightGateDeferral(unittest.TestCase):
    """Given run_preflight_and_dry_run, When skip_full_val_gate=True and
    B0 evidence active, Then static full-validation gate defers to B0."""

    def setUp(self):
        from federated_mcts.evaluation.game24_runner_setup import ExperimentSetup
        self._base_setup_kw: dict = {
            "task": "game24",
            "task_ids": list(range(200)),  # full 200-ID validation
            "task_split_name": "val",
            "decision_policy": "local_validity",
            "oracle_audit": True,
            "seed": 17,
            "temperature": 0.9,
            "method_generate": "propose",
            "method_evaluate": "vote",
            "method_select": "greedy",
            "n_generate_sample": 2,
            "n_evaluate_sample": 3,
            "n_select_sample": 2,
            "game24_exact_prune": False,
            "assign_strategy": "round_robin",
            "max_call_limit": None,
            "max_token_limit": None,
            "max_cost_usd": None,
            "max_raw_unit_limit": 19.99,
            "model_config_path": "configs/model_config.json",
            "split_file_path": "",
            "is_selected_test": False,
            "review_round_identity": "review-1",
            "auth_registry_path": None,
            "file_name": "/tmp/b0_test",
            "pilot_ids": None,
            "dry_run": False,
            "mock_client": False,
        }
        self.SetupCls = ExperimentSetup

    def _setup(self, **kw) -> "ExperimentSetup":  # noqa: F821
        merged = dict(self._base_setup_kw)
        merged.update(kw)
        return self.SetupCls(**merged)

    def _manifest(self) -> MagicMock:
        return MagicMock(spec=Game24Manifest)

    def _failing_projection(self) -> "PreflightProjection":  # noqa: F821
        return PreflightProjection(
            schema_version=1,
            n_tasks=200,
            model_names=["deepseek-v4-pro"],
            model_rates_raw={"deepseek-v4-pro": {"prompt": 0.435, "completion": 0.87}},
            solver_config={"n_generate_sample": 2, "n_evaluate_sample": 3,
                           "n_select_sample": 2, "task_steps": 3},
            worst_case_calls=9600,
            worst_case_prompt_tokens=1920000,
            worst_case_completion_tokens=4915200,
            worst_case_raw_unit_estimate=1.645056,
            gate_limit=0.75,
            gate_passed=False,
        )

    def test_skip_full_val_gate_defers_static_block(self):
        setup = self._setup()
        manifest = self._manifest()
        projection = self._failing_projection()
        result = run_preflight_and_dry_run(
            setup, manifest, projection, skip_full_val_gate=True,
        )
        self.assertTrue(result,
                        "Should proceed when static gate is deferred to B0")

    def test_no_skip_full_val_gate_exits_on_block(self):
        setup = self._setup()
        manifest = self._manifest()
        projection = self._failing_projection()
        with self.assertRaises(SystemExit):
            run_preflight_and_dry_run(
                setup, manifest, projection, skip_full_val_gate=False,
            )

    def test_dry_run_with_skip_flag_returns_false(self):
        setup = self._setup(dry_run=True)
        manifest = self._manifest()
        projection = self._failing_projection()
        result = run_preflight_and_dry_run(
            setup, manifest, projection, skip_full_val_gate=True,
        )
        self.assertFalse(result, "dry_run should return False regardless of gate")

    def test_skip_flag_no_effect_on_pilot_small_ids(self):
        setup = self._setup(task_ids=list(range(20)))
        manifest = self._manifest()
        projection = self._failing_projection()
        result = run_preflight_and_dry_run(
            setup, manifest, projection, skip_full_val_gate=False,
        )
        self.assertTrue(result,
                        "Pilot runs (<=20 tasks) pass through static gate")

    def test_skip_flag_no_effect_on_passing_projection(self):
        setup = self._setup()
        manifest = self._manifest()
        passing = PreflightProjection(
            schema_version=1,
            n_tasks=200,
            model_names=["deepseek-v4-flash"],
            model_rates_raw={"deepseek-v4-flash": {"prompt": 0.14, "completion": 0.28}},
            solver_config={"n_generate_sample": 1, "n_evaluate_sample": 1,
                           "n_select_sample": 1, "task_steps": 3},
            worst_case_calls=2400,
            worst_case_prompt_tokens=480000,
            worst_case_completion_tokens=1228800,
            worst_case_raw_unit_estimate=0.42,
            gate_limit=0.75,
            gate_passed=True,
        )
        result = run_preflight_and_dry_run(
            setup, manifest, passing, skip_full_val_gate=False,
        )
        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()
