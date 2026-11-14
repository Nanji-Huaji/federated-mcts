"""Validation tests for BudgetLedger — NaN, inf, bool, negative inputs rejected.

Extracted from test_game24_budget.py to keep ≤250 pure LOC per file.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

sys.path.insert(0, "src")


class TestBudgetValidation(unittest.TestCase):
    """NaN, inf, bool, negative inputs are rejected."""

    def setUp(self) -> None:
        self.tmpdir_obj = tempfile.TemporaryDirectory()
        self.tmpdir = self.tmpdir_obj.name

    def tearDown(self) -> None:
        self.tmpdir_obj.cleanup()

    def _ledger(self, **kw):
        from federated_mcts.evaluation.game24_budget import BudgetLedger
        return BudgetLedger(os.path.join(self.tmpdir, "b.jsonl"), **kw)

    def test_nan_total_limit_raises(self):
        with self.assertRaises(ValueError):
            self._ledger(total_limit=float("nan"))

    def test_inf_total_limit_raises(self):
        with self.assertRaises(ValueError):
            self._ledger(total_limit=float("inf"))

    def test_negative_total_limit_raises(self):
        with self.assertRaises(ValueError):
            self._ledger(total_limit=-1.0)

    def test_bool_total_limit_raises(self):
        with self.assertRaises((ValueError, TypeError)):
            self._ledger(total_limit=True)

    def test_nan_spend_raises(self):
        ledger = self._ledger(total_limit=10.0)
        with self.assertRaises(ValueError):
            ledger.spend(float("nan"))

    def test_inf_spend_raises(self):
        ledger = self._ledger(total_limit=10.0)
        with self.assertRaises(ValueError):
            ledger.spend(float("inf"))

    def test_bool_spend_raises(self):
        ledger = self._ledger(total_limit=10.0)
        with self.assertRaises((ValueError, TypeError)):
            ledger.spend(True)


if __name__ == "__main__":
    unittest.main()
