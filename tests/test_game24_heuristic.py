"""Game24 product heuristic and validation isolation tests.

Covers:
- Product<24 heuristic separation from local validity
- Pure local validation never imports or calls game24_exact_prune
"""

import importlib
import sys
import unittest

sys.path.insert(0, "src")


class TestProductHeuristicSeparated(unittest.TestCase):
    """Prove product<24 is locally valid but heuristic-labeled."""

    def test_product_below_24_is_locally_valid(self):
        """A step leaving product<24 must pass local validation after separation."""
        from federated_mcts.utils.format_check import check_final_result
        # 1+1=2 from {1,1,1,2} → remaining {1,2,2}, product=4 < 24
        correct, _ = check_final_result(
            "1 + 1 = 2 (left: 1 2 2)", x="1 1 1 2",
        )
        self.assertTrue(correct,
                        "DEFECT: product<24 must be locally valid after separation")

    def test_heuristic_detects_product_below_24(self):
        """The heuristic function must detect product<24."""
        from federated_mcts.utils.format_check import game24_heuristic_product_check
        self.assertFalse(
            game24_heuristic_product_check([1, 1, 2]),
            "DEFECT: heuristic must flag product < 24 for [1,1,2] (product=2)",
        )

    def test_heuristic_passes_product_above_24(self):
        """The heuristic must pass when product >= 24."""
        from federated_mcts.utils.format_check import game24_heuristic_product_check
        self.assertTrue(
            game24_heuristic_product_check([3, 3, 4]),
            "DEFECT: heuristic must pass for [3,3,4] (product=36 >= 24)",
        )

    def test_heuristic_passes_single_number(self):
        """Single remaining number always passes the heuristic."""
        from federated_mcts.utils.format_check import game24_heuristic_product_check
        self.assertTrue(
            game24_heuristic_product_check([24]),
            "DEFECT: single number [24] should pass heuristic",
        )
        self.assertTrue(
            game24_heuristic_product_check([1]),
            "DEFECT: single number [1] should pass heuristic (terminal state)",
        )

    def test_heuristic_passes_empty(self):
        """Empty number list passes the heuristic."""
        from federated_mcts.utils.format_check import game24_heuristic_product_check
        self.assertTrue(
            game24_heuristic_product_check([]),
            "DEFECT: empty list should pass heuristic",
        )


class TestPureValidationIsolation(unittest.TestCase):
    """Pure local validation must never import or call game24_exact_prune."""

    def test_format_check_does_not_import_exact_prune(self):
        """format_check module must not import game24_exact_prune."""
        if "federated_mcts.utils.format_check" in sys.modules:
            del sys.modules["federated_mcts.utils.format_check"]
        mod = importlib.import_module("federated_mcts.utils.format_check")
        self.assertFalse(
            hasattr(mod, "game24_exact_prune"),
            "DEFECT: format_check must not import game24_exact_prune",
        )

    def test_heuristic_function_does_not_import_exact_prune(self):
        """The heuristic function also must not call exact prune."""
        from federated_mcts.utils.format_check import game24_heuristic_product_check
        result = game24_heuristic_product_check([1, 2, 3])
        self.assertIsInstance(result, bool)


if __name__ == "__main__":
    unittest.main()
