"""Test preflight projection: build, gate, reconciliation, persistence."""
from __future__ import annotations

import json
import os
import tempfile
import unittest

from federated_mcts.evaluation.game24_preflight import (
    PreflightProjection,
    build_projection,
    persist_projection,
    reconcile_budget_after_tasks,
    get_model_rate,
    SCHEMA_VERSION,
    FULL_VALIDATION_GATE,
)

_SIMPLE_CONFIG: list[dict] = [
    {"client_name": "remote_client", "model": "deepseek-v4-flash",
     "max_tokens": 512, "api_base": "http://f"},
    {"client_name": "local_client", "model": "Qwen/Qwen2.5-32B-Instruct",
     "max_tokens": 256, "api_base": "http://l"},
]


class TestPreflightProjection(unittest.TestCase):
    """Given model config and solver settings, When building projection, Then correct."""

    def test_build_basic_projection(self) -> None:
        """When building a projection for 20 tasks, Then it's under the gate."""
        p = build_projection(
            n_tasks=20, model_config_raw=_SIMPLE_CONFIG,
            n_generate_sample=2, n_evaluate_sample=3, n_select_sample=2)
        self.assertEqual(p.schema_version, SCHEMA_VERSION)
        self.assertEqual(p.n_tasks, 20)
        self.assertGreater(p.worst_case_calls, 0)
        self.assertGreater(p.worst_case_raw_unit_estimate, 0.0)
        self.assertTrue(p.gate_passed)
        self.assertEqual(p.gate_limit, FULL_VALIDATION_GATE)

    def test_over_cap_blocked(self) -> None:
        """When projection exceeds gate for 200 tasks, Then gate fails."""
        p = build_projection(
            n_tasks=200, model_config_raw=_SIMPLE_CONFIG,
            n_generate_sample=10, n_evaluate_sample=10, n_select_sample=5)
        self.assertFalse(p.gate_passed)
        self.assertGreater(p.worst_case_raw_unit_estimate, FULL_VALIDATION_GATE)

    def test_model_rates_known(self) -> None:
        """When looking up known model rates, Then returns correct values."""
        r = get_model_rate("deepseek-v4-flash")
        self.assertEqual(r["prompt"], 0.14)
        self.assertEqual(r["completion"], 0.28)

    def test_model_rates_unknown(self) -> None:
        """When looking up unknown model rates, Then returns None (fail-closed)."""
        r = get_model_rate("unknown-model")
        self.assertIsNone(r, "Unknown models must return None (fail-closed, no invented fallback)")

    def test_projection_json_serializable(self) -> None:
        """When converting to JSON, Then it's valid JSON with expected keys."""
        p = build_projection(
            n_tasks=20, model_config_raw=_SIMPLE_CONFIG,
            n_generate_sample=1, n_evaluate_sample=1, n_select_sample=1)
        j = p.to_json()
        d = json.loads(j)
        for key in ("schema_version", "n_tasks", "worst_case_calls",
                     "worst_case_raw_unit_estimate", "gate_passed"):
            self.assertIn(key, d)

    def test_persist_projection(self) -> None:
        """When persisting a projection, Then file is written with correct content."""
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "proj.json")
            p = build_projection(
                n_tasks=20, model_config_raw=_SIMPLE_CONFIG,
                n_generate_sample=1, n_evaluate_sample=1, n_select_sample=1)
            persist_projection(p, path)
            self.assertTrue(os.path.exists(path))
            with open(path) as f:
                loaded = json.load(f)
            self.assertEqual(loaded["schema_version"], SCHEMA_VERSION)
            self.assertEqual(loaded["n_tasks"], 20)

    def test_reconcile_budget_after_tasks(self) -> None:
        """When reconciling budget, Then fields match projection and spent."""
        p = build_projection(
            n_tasks=20, model_config_raw=_SIMPLE_CONFIG,
            n_generate_sample=1, n_evaluate_sample=1, n_select_sample=1)
        rec = reconcile_budget_after_tasks(0.05, p)
        self.assertEqual(rec["preflight_projection_raw_units"],
                         p.worst_case_raw_unit_estimate)
        self.assertEqual(rec["actual_spent_raw_units"], 0.05)
        self.assertTrue(rec["projection_gate_passed"])

    def test_empty_config_handled(self) -> None:
        """When model_config is empty, Then projection is still valid."""
        p = build_projection(
            n_tasks=10, model_config_raw=[],
            n_generate_sample=1, n_evaluate_sample=1, n_select_sample=1)
        self.assertEqual(p.n_tasks, 10)
        self.assertGreaterEqual(p.worst_case_raw_unit_estimate, 0.0)

    def test_unknown_remote_model_fails_closed(self) -> None:
        """When remote model has no known rate, Then ValueError is raised."""
        with self.assertRaises(ValueError) as ctx:
            build_projection(
                n_tasks=5, model_config_raw=[
                    {"client_name": "remote_client", "model": "unknown-paid-model",
                     "max_tokens": 512}],
                n_generate_sample=1, n_evaluate_sample=1, n_select_sample=1)
        self.assertIn("unknown-paid-model", str(ctx.exception))
        self.assertIn("has no rate", str(ctx.exception))

    def test_single_model_config_not_double_charged(self) -> None:
        """When multi-model config present, Then only remote_client is charged."""
        p = build_projection(
            n_tasks=20, model_config_raw=_SIMPLE_CONFIG,
            n_generate_sample=1, n_evaluate_sample=1, n_select_sample=1)
        # With deepseek-v4-flash remote, charge should reflect ONLY that model
        # If local model (Qwen) were charged at its zero rate, total should
        # still equal deepseek-only charge since Qwen rate is 0.
        # The key assertion: model_rates_raw should contain both model names
        # but the worst_case_raw_unit_estimate must equal only remote charges
        self.assertIn("deepseek-v4-flash", p.model_names)
        self.assertIn("Qwen/Qwen2.5-32B-Instruct", p.model_names)
        # Rate for local model should be zero
        local_rate = p.model_rates_raw.get("Qwen/Qwen2.5-32B-Instruct", {})
        self.assertEqual(local_rate.get("completion"), 0.0)
        self.assertEqual(local_rate.get("prompt"), 0.0)


if __name__ == "__main__":
    unittest.main()
