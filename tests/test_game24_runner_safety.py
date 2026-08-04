"""Regression tests for raw-unit ledger safety defects in the Game24 runner.

Covers:
- First-task charging (prev_snapshot starts as empty dict)
- Two-task cumulative spend
- Resume: completed tasks are not recharged
- Unknown billed model raises ValueError (fail-closed)
- Known zero-cost local models are allowed
"""

from __future__ import annotations

import sys
import unittest
from unittest.mock import MagicMock, patch

# ── Mock problematic module-level imports before loading game24_runner ──
sys.path.insert(0, "src")

# federated_mcts.models imports openai — mock it out before the module loads
_fake_models = MagicMock()
_fake_cost_limit = MagicMock()
_fake_cost_limit.cost_exceeded = MagicMock(return_value=False)
_fake_cost_limit.total_cost_usd = MagicMock(return_value=0.0)

with patch.dict("sys.modules", {
    "federated_mcts.models": _fake_models,
    "federated_mcts.utils.cost_limit": _fake_cost_limit,
}):
    from federated_mcts.evaluation.game24_runner import (
        compute_task_raw_unit_cost,
    )


class TestFirstTaskCharging(unittest.TestCase):
    """Given an empty prev_snapshot {}, When first task uses tokens,
    Then the full usage delta is charged exactly once."""

    def test_empty_prev_snapshot_charges_full_delta(self) -> None:
        """When prev_snapshot is {}, Then first task usage is fully charged."""
        prev: dict[str, dict[str, float]] = {}
        current = {
            "deepseek-v4-flash": {
                "prompt_tokens": 1000,
                "completion_tokens": 500,
                "total_tokens": 1500,
                "total_calls": 5,
            },
        }
        cost = compute_task_raw_unit_cost(prev, current)
        # deepseek-v4-flash: prompt=0.14, completion=0.28 per 1M tokens
        expected = (1000 * 0.14 + 500 * 0.28) / 1_000_000
        self.assertAlmostEqual(cost, expected, places=8)
        self.assertGreater(cost, 0.0)

    def test_empty_prev_no_usage_is_zero(self) -> None:
        """When prev_snapshot is {} and no tokens consumed, Then cost is zero."""
        prev: dict[str, dict[str, float]] = {}
        current: dict[str, dict[str, float]] = {}
        cost = compute_task_raw_unit_cost(prev, current)
        self.assertEqual(cost, 0.0)


class TestTwoTaskCumulativeSpend(unittest.TestCase):
    """Given consecutive task snapshots, When computing per-task costs,
    Then the sum of individual costs equals the total delta cost."""

    def test_two_tasks_sum_equals_total_delta(self) -> None:
        """When two tasks produce usage, Then cost1 + cost2 == total delta."""
        prev: dict[str, dict[str, float]] = {}
        after_task1 = {
            "deepseek-v4-flash": {
                "prompt_tokens": 1000,
                "completion_tokens": 500,
                "total_tokens": 1500,
                "total_calls": 5,
            },
        }
        after_task2 = {
            "deepseek-v4-flash": {
                "prompt_tokens": 2200,
                "completion_tokens": 1200,
                "total_tokens": 3400,
                "total_calls": 11,
            },
        }

        cost1 = compute_task_raw_unit_cost(prev, after_task1)
        cost2 = compute_task_raw_unit_cost(after_task1, after_task2)
        cost_total = compute_task_raw_unit_cost(prev, after_task2)

        self.assertAlmostEqual(cost1 + cost2, cost_total, places=8)
        self.assertGreater(cost1, 0.0)
        self.assertGreater(cost2, 0.0)

    def test_zero_usage_task_charges_zero(self) -> None:
        """When a task produces no token delta, Then cost is zero."""
        snap = {
            "gpt-4o": {
                "prompt_tokens": 5000,
                "completion_tokens": 2000,
                "total_tokens": 7000,
                "total_calls": 3,
            },
        }
        cost = compute_task_raw_unit_cost(snap, snap)  # no change
        self.assertEqual(cost, 0.0)


class TestResumeNoDoubleCharge(unittest.TestCase):
    """Given a resume scenario with restored completed tasks,
    When the first new task runs, Then only the new task's delta is charged
    and completed tasks contribute zero additional cost."""

    def test_resume_snapshot_captures_pre_loop_state(self) -> None:
        """When resume restores 2 completed tasks, then skips them,
        Then the first non-skipped task is charged only its own usage."""
        # Simulate: resume restores completed tasks, but model usage
        # tracker is empty (no tokens consumed in this invocation yet).
        # So prev_snapshot = {} before the loop.
        # Skipped tasks don't enter the delta block.
        # First new task (i=2) produces usage.
        prev_snapshot: dict[str, dict[str, float]] = {}
        after_first_new_task = {
            "gpt-4o": {
                "prompt_tokens": 3000,
                "completion_tokens": 1000,
                "total_tokens": 4000,
                "total_calls": 2,
            },
        }
        cost = compute_task_raw_unit_cost(prev_snapshot, after_first_new_task)
        # gpt-4o: prompt=2.5, completion=10.0 per 1M tokens
        expected = (3000 * 2.5 + 1000 * 10.0) / 1_000_000
        self.assertAlmostEqual(cost, expected, places=8)
        self.assertGreater(cost, 0.0)

    def test_resume_second_new_task_only_charges_increment(self) -> None:
        """When a second non-skipped task runs after resume,
        Then only the incremental usage from that task is charged."""
        after_task2 = {
            "gpt-4o": {
                "prompt_tokens": 3000,
                "completion_tokens": 1000,
                "total_tokens": 4000,
                "total_calls": 2,
            },
        }
        after_task3 = {
            "gpt-4o": {
                "prompt_tokens": 4500,
                "completion_tokens": 1600,
                "total_tokens": 6100,
                "total_calls": 4,
            },
        }
        cost2 = compute_task_raw_unit_cost(after_task2, after_task3)
        expected = ((4500 - 3000) * 2.5 + (1600 - 1000) * 10.0) / 1_000_000
        self.assertAlmostEqual(cost2, expected, places=8)
        self.assertGreater(cost2, 0.0)


class TestUnknownBilledModelReject(unittest.TestCase):
    """Given a model with positive token delta but no price entry,
    When computing raw-unit cost, Then ValueError is raised (fail-closed)."""

    def test_unknown_model_positive_delta_raises(self) -> None:
        """When an unknown model has positive prompt delta, Then ValueError."""
        prev: dict[str, dict[str, float]] = {}
        current = {
            "some-new-model-v2": {
                "prompt_tokens": 500,
                "completion_tokens": 0,
                "total_tokens": 500,
                "total_calls": 1,
            },
        }
        with self.assertRaises(ValueError) as ctx:
            compute_task_raw_unit_cost(prev, current)
        self.assertIn("Unknown billed model", str(ctx.exception))
        self.assertIn("some-new-model-v2", str(ctx.exception))

    def test_unknown_model_positive_completion_delta_raises(self) -> None:
        """When an unknown model has positive completion delta, Then ValueError."""
        prev: dict[str, dict[str, float]] = {}
        current = {
            "future-gpt-7": {
                "prompt_tokens": 0,
                "completion_tokens": 300,
                "total_tokens": 300,
                "total_calls": 1,
            },
        }
        with self.assertRaises(ValueError) as ctx:
            compute_task_raw_unit_cost(prev, current)
        self.assertIn("future-gpt-7", str(ctx.exception))

    def test_unknown_model_zero_delta_no_error(self) -> None:
        """When an unknown model has zero delta, Then no error (safe perimeter)."""
        snap = {
            "unlisted-model": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "total_calls": 1,
            },
        }
        # Same snap means zero delta — should not raise
        cost = compute_task_raw_unit_cost(snap, snap)
        self.assertEqual(cost, 0.0)

    def test_known_zero_cost_models_allowed(self) -> None:
        """When a known local model with zero price is used, Then cost is zero
        and no error is raised."""
        prev: dict[str, dict[str, float]] = {}
        current = {
            "phi-3-medium-4k-instruct": {
                "prompt_tokens": 5000,
                "completion_tokens": 200,
                "total_tokens": 5200,
                "total_calls": 2,
            },
            "meta-llama-3.1-8b-instruct@q4_k_m": {
                "prompt_tokens": 300,
                "completion_tokens": 100,
                "total_tokens": 400,
                "total_calls": 1,
            },
        }
        cost = compute_task_raw_unit_cost(prev, current)
        self.assertEqual(cost, 0.0)

    def test_mixed_known_and_unknown_fails_closed(self) -> None:
        """When a mix of known and unknown models with positive deltas exists,
        Then ValueError is raised naming the unknown model."""
        prev: dict[str, dict[str, float]] = {}
        current = {
            "deepseek-v4-flash": {
                "prompt_tokens": 800,
                "completion_tokens": 400,
                "total_tokens": 1200,
                "total_calls": 3,
            },
            "unknown-model-x": {
                "prompt_tokens": 100,
                "completion_tokens": 0,
                "total_tokens": 100,
                "total_calls": 1,
            },
        }
        with self.assertRaises(ValueError) as ctx:
            compute_task_raw_unit_cost(prev, current)
        self.assertIn("unknown-model-x", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
