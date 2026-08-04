"""TDD tests for the shared raw-unit aggregate budget ledger.

Coverage targets for Todo 3:
- All planned experiments share ONE raw-unit budget, default 19.99
- No currency conversion (raw units only, parallelism with cost caps)
- fsync + lock + hash-chained append-only ledger (same pattern as auth registry)
- Runs cannot each spend 19.99 independently
- Calls/tokens/raw-cost caps: check before starting new task
- Incomplete stop states when any cap is reached
- Cap enforcement at task granularity (fail before next task)
"""

from __future__ import annotations

import threading
import os
import sys
import tempfile
import unittest

sys.path.insert(0, "src")


class TestBudgetLedgerUnit(unittest.TestCase):
    """Basic ledger operations: init, spend, cap enforcement."""

    def setUp(self) -> None:
        self.tmpdir_obj = tempfile.TemporaryDirectory()
        self.tmpdir = self.tmpdir_obj.name
        self.ledger_path = os.path.join(self.tmpdir, "budget_ledger.jsonl")

    def tearDown(self) -> None:
        self.tmpdir_obj.cleanup()

    def test_ledger_defaults_to_19_99(self):
        """A new budget tracker defaults to 19.99 raw units."""
        from federated_mcts.evaluation.game24_budget import (
            BudgetLedger,
        )
        ledger = BudgetLedger(self.ledger_path)
        self.assertAlmostEqual(ledger.remaining, 19.99)

    def test_ledger_respects_custom_limit(self):
        """Custom limit is honored."""
        from federated_mcts.evaluation.game24_budget import (
            BudgetLedger,
        )
        ledger = BudgetLedger(self.ledger_path, total_limit=5.0)
        self.assertAlmostEqual(ledger.remaining, 5.0)

    def test_spend_reduces_remaining(self):
        """Spending raw units reduces remaining."""
        from federated_mcts.evaluation.game24_budget import (
            BudgetLedger,
        )
        ledger = BudgetLedger(self.ledger_path, total_limit=10.0)
        ok = ledger.spend(3.0)
        self.assertTrue(ok)
        self.assertAlmostEqual(ledger.remaining, 7.0)

    def test_cannot_spend_below_zero(self):
        """Spending more than remaining fails and does not deduct."""
        from federated_mcts.evaluation.game24_budget import (
            BudgetLedger,
        )
        ledger = BudgetLedger(self.ledger_path, total_limit=3.0)
        ok = ledger.spend(5.0)
        self.assertFalse(ok)
        self.assertAlmostEqual(ledger.remaining, 3.0)

    def test_spend_exact_limit_works(self):
        """Spending exactly the remaining amount works."""
        from federated_mcts.evaluation.game24_budget import (
            BudgetLedger,
        )
        ledger = BudgetLedger(self.ledger_path, total_limit=5.0)
        ok = ledger.spend(5.0)
        self.assertTrue(ok)
        self.assertAlmostEqual(ledger.remaining, 0.0)

    def test_spend_is_idempotent_fail(self):
        """After spending exhausts budget, further spends fail."""
        from federated_mcts.evaluation.game24_budget import (
            BudgetLedger,
        )
        ledger = BudgetLedger(self.ledger_path, total_limit=2.0)
        self.assertTrue(ledger.spend(2.0))
        self.assertFalse(ledger.spend(0.01))

    def test_ledger_is_persistent(self):
        """Budget persists across instances reading the same file."""
        from federated_mcts.evaluation.game24_budget import (
            BudgetLedger,
        )
        l1 = BudgetLedger(self.ledger_path, total_limit=10.0)
        l1.spend(4.0)

        # New instance reads the same file
        l2 = BudgetLedger(self.ledger_path)
        self.assertAlmostEqual(l2.remaining, 6.0)

    def test_negative_spend_rejected(self):
        """Negative spend amounts raise ValueError."""
        from federated_mcts.evaluation.game24_budget import (
            BudgetLedger,
        )
        ledger = BudgetLedger(self.ledger_path)
        with self.assertRaises(ValueError):
            ledger.spend(-1.0)

    def test_zero_spend_is_allowed(self):
        """Spending 0 is a no-op."""
        from federated_mcts.evaluation.game24_budget import (
            BudgetLedger,
        )
        ledger = BudgetLedger(self.ledger_path, total_limit=1.0)
        ok = ledger.spend(0.0)
        self.assertTrue(ok)
        self.assertAlmostEqual(ledger.remaining, 1.0)

    def test_hash_chain_verifies(self):
        """Ledger entries form a verifiable hash chain."""
        from federated_mcts.evaluation.game24_budget import (
            BudgetLedger,
        )
        ledger = BudgetLedger(self.ledger_path, total_limit=10.0)
        ledger.spend(2.0)
        ledger.spend(3.0)

        ok, err = ledger.verify_chain()
        self.assertTrue(ok, f"Hash chain broken: {err}")


class TestBudgetCaps(unittest.TestCase):
    """Calls/tokens/cost caps with incomplete stop states."""

    def setUp(self) -> None:
        self.tmpdir_obj = tempfile.TemporaryDirectory()
        self.tmpdir = self.tmpdir_obj.name
        self.ledger_path = os.path.join(self.tmpdir, "budget_caps.jsonl")

    def tearDown(self) -> None:
        self.tmpdir_obj.cleanup()

    def test_call_cap_exceeded(self):
        """When call limit is reached, the run is marked incomplete."""
        from federated_mcts.evaluation.game24_budget import (
            BudgetLedger,
        )
        ledger = BudgetLedger(
            self.ledger_path,
            total_limit=19.99,
            max_calls=3,
        )
        # Simulate 3 tasks, each consuming 1 call
        for _i in range(3):
            ok, stop_reason = ledger.check_and_mark_calls(used_calls=_i + 1)
            self.assertIsNone(stop_reason)
            self.assertTrue(ok)

        # 4th task: cap exceeded
        ok, stop_reason = ledger.check_and_mark_calls(used_calls=4)
        self.assertFalse(ok)
        self.assertEqual(stop_reason, "max_total_calls")

    def test_token_cap_exceeded(self):
        """When token limit is reached, the run is marked incomplete."""
        from federated_mcts.evaluation.game24_budget import (
            BudgetLedger,
        )
        ledger = BudgetLedger(
            self.ledger_path,
            total_limit=19.99,
            max_tokens=1000,
        )
        ok, stop_reason = ledger.check_and_mark_tokens(used_tokens=1200)
        self.assertFalse(ok)
        self.assertEqual(stop_reason, "max_total_tokens")

    def test_token_cap_not_exceeded(self):
        """Token usage under cap passes."""
        from federated_mcts.evaluation.game24_budget import (
            BudgetLedger,
        )
        ledger = BudgetLedger(
            self.ledger_path,
            total_limit=19.99,
            max_tokens=1000,
        )
        ok, stop_reason = ledger.check_and_mark_tokens(used_tokens=500)
        self.assertTrue(ok)
        self.assertIsNone(stop_reason)

    def test_cap_none_means_unlimited(self):
        """None cap means no enforcement."""
        from federated_mcts.evaluation.game24_budget import (
            BudgetLedger,
        )
        ledger = BudgetLedger(self.ledger_path, total_limit=19.99)
        ok, stop_reason = ledger.check_and_mark_calls(used_calls=999999)
        self.assertTrue(ok)
        self.assertIsNone(stop_reason)


class TestBudgetConcurrency(unittest.TestCase):
    """Concurrent spends are safe."""

    def setUp(self) -> None:
        self.tmpdir_obj = tempfile.TemporaryDirectory()
        self.tmpdir = self.tmpdir_obj.name
        self.ledger_path = os.path.join(self.tmpdir, "budget_concurrent.jsonl")

    def tearDown(self) -> None:
        self.tmpdir_obj.cleanup()

    def test_concurrent_spends_dont_overdraw(self):
        """Multiple threads spending a shared budget cannot overdraw."""
        from federated_mcts.evaluation.game24_budget import (
            BudgetLedger,
        )

        # Shared ledger with small budget
        ledger = BudgetLedger(self.ledger_path, total_limit=3.0)

        results: list[bool] = []

        def _spend_one() -> None:
            results.append(ledger.spend(1.0))

        threads = [threading.Thread(target=_spend_one) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # At most 3 successes (budget is 3.0)
        successes = sum(results)
        self.assertLessEqual(
            successes, 3, f"Budget overdrawn: {successes} successes out of 5"
        )
        self.assertAlmostEqual(ledger.remaining, 3.0 - successes)


class TestBudgetIntegrationWithAuth(unittest.TestCase):
    """Budget ledger and auth registry work together."""

    def setUp(self) -> None:
        self.tmpdir_obj = tempfile.TemporaryDirectory()
        self.tmpdir = self.tmpdir_obj.name
        self.ledger_path = os.path.join(self.tmpdir, "budget.jsonl")
        self.auth_path = os.path.join(self.tmpdir, "auth.jsonl")

    def tearDown(self) -> None:
        self.tmpdir_obj.cleanup()

    def test_budget_survives_alongside_auth(self):
        """Both ledgers coexist and maintain independence."""
        from federated_mcts.evaluation.game24_budget import (
            BudgetLedger,
        )
        from federated_mcts.evaluation.game24_authorization import (
            AuthRegistry,
            make_auth_key,
        )
        import hashlib

        ledger = BudgetLedger(self.ledger_path, total_limit=5.0)
        ledger.spend(2.0)

        registry = AuthRegistry(self.auth_path)
        key = make_auth_key(
            ordered_test_ids_hash=hashlib.sha256(b"ids").hexdigest(),
            model_config_hash=hashlib.sha256(b"cfg").hexdigest(),
            decision_policy_fingerprint=hashlib.sha256(b"fp").hexdigest(),
            review_round_identity="r1",
        )
        registry.consume(key, hashlib.sha256(b"run").hexdigest(), is_selected_test=True)

        # Both files exist and are valid
        self.assertTrue(os.path.exists(self.ledger_path))
        self.assertTrue(os.path.exists(self.auth_path))


if __name__ == "__main__":
    unittest.main()
