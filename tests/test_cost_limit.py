import sys
import unittest

sys.path.insert(0, "src")

from federated_mcts.utils.cost_limit import cost_exceeded, total_cost_usd
from federated_mcts.models.usage import UsageTracker


class TestCostLimit(unittest.TestCase):
    def test_cost_limit_is_inclusive_and_handles_missing_costs(self):
        summary = {"a": {"cost": 1.5}, "b": {}}
        self.assertEqual(total_cost_usd(summary), 1.5)
        self.assertFalse(cost_exceeded(summary, None))
        self.assertFalse(cost_exceeded(summary, 2.0))
        self.assertTrue(cost_exceeded(summary, 1.5))

    def test_v4_pro_uses_conservative_cache_miss_rates(self):
        tracker = UsageTracker()
        self.assertAlmostEqual(
            tracker._calculate_cost("deepseek-v4-pro", 1_000_000, 1_000_000),
            1.305,
        )


if __name__ == "__main__":
    unittest.main()
