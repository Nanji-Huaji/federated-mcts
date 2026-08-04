"""Tests for B0 projection computation and zero-overhead proofs.

Covers: projection formula, scaled usage, over-gate failure, JSON
serialization, baseline proof pass, warm-start/DQN/exact-prune proof
failures.  All fixtures synthetic.
"""

from __future__ import annotations

import json
import sys
import unittest
from decimal import Decimal

sys.path.insert(0, "src")

from federated_mcts.evaluation.game24_b0_evidence import (
    B0EvidenceRecord,
    B0_EXPECTED_SPENT,
    CANONICAL_PILOT_IDS,
    SCALE_FACTOR,
    SCHEMA_VERSION,
    compute_b0_projection,
    prove_zero_startup_overhead,
    prove_zero_retry_allowance,
)
from federated_mcts.evaluation.game24_b0_manifest_loader import (
    B0SourceDecisionIdentity,
)


class TestB0Projection(unittest.TestCase):
    """Given evidence, When computing projection, Then correct terms."""

    def _ev(self, spent=B0_EXPECTED_SPENT) -> B0EvidenceRecord:
        return B0EvidenceRecord(
            schema_version=SCHEMA_VERSION, pilot_ids=list(CANONICAL_PILOT_IDS),
            pilot_id_count=20, decision_policy="local_validity",
            oracle_audit=True, seed=17, method_generate="propose",
            method_evaluate="vote", method_select="greedy",
            n_generate_sample=2, n_evaluate_sample=3, n_select_sample=2,
            calls=160, prompt_tokens=64960, completion_tokens=81919,
            total_tokens=146879, raw_units_spent=spent, tasks_completed=20,
            manifest_sha256="a", performance_sha256="b",
            journal_sha256="c", budget_sha256="d",
            decision_policy_fingerprint="dp", run_fingerprint="run",
            source_decision=B0SourceDecisionIdentity(
                model_config_sha256="a" * 64,
                model_names=["deepseek-v4-flash"],
                decision_policy="local_validity",
                seed=17,
                temperature=0.9,
                method_generate="propose",
                method_evaluate="vote",
                method_select="greedy",
                n_generate_sample=2,
                n_evaluate_sample=3,
                n_select_sample=2,
                game24_exact_prune=False,
                assign_strategy="round_robin",
                max_call_limit=None,
                max_token_limit=None,
                max_cost_usd=None,
                max_raw_unit_limit=19.99,
                runner_path="scripts/merged_run.py",
            ),
        )

    def test_exact_formula(self):
        p, sp, rp = compute_b0_projection(self._ev())
        self.assertEqual(p.pilot_usage, B0_EXPECTED_SPENT)
        self.assertEqual(p.scale_factor, SCALE_FACTOR)
        self.assertEqual(p.startup_overhead, Decimal("0"))
        self.assertEqual(p.retry_allowance, Decimal("0"))
        self.assertTrue(p.gate_passed)

    def test_scaled_is_10x(self):
        p, _, _ = compute_b0_projection(self._ev())
        exp = (B0_EXPECTED_SPENT * SCALE_FACTOR).quantize(Decimal("0.00000001"))
        self.assertEqual(p.scaled_usage, exp)

    def test_over_gate_fails(self):
        p, _, _ = compute_b0_projection(self._ev(spent=Decimal("0.1")))
        self.assertFalse(p.gate_passed)

    def test_json_serializable(self):
        p, _, _ = compute_b0_projection(self._ev())
        d = json.loads(p.to_json())
        self.assertIn("projected_raw_units", d)


class TestB0Proofs(unittest.TestCase):
    """Given B0 settings, When proving overhead, Then correct."""

    def test_baseline_passes(self):
        ok, v, _ = prove_zero_startup_overhead(method_generate="propose")
        self.assertTrue(ok)
        ok2, v2, _ = prove_zero_retry_allowance(2, 3)
        self.assertTrue(ok2)

    def test_warm_start_fails(self):
        self.assertFalse(prove_zero_startup_overhead(warm_start=True)[0])

    def test_dqn_fails(self):
        self.assertFalse(prove_zero_startup_overhead(search_policy="dqn")[0])

    def test_exact_prune_fails(self):
        self.assertFalse(prove_zero_startup_overhead(game24_exact_prune=True)[0])


if __name__ == "__main__":
    unittest.main()
