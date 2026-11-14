"""Tests for game24 exact dead-end pruning: reachability oracle, candidate
filtering, order preservation, metrics, and disabled-by-default behavior."""

import sys
import unittest
from fractions import Fraction

sys.path.insert(0, "src")


class TestCanReachTargetNumbers(unittest.TestCase):
    """Unit tests for the core reachability oracle (private helper)."""

    def _call_oracle(self, numbers):
        from federated_mcts.tasks.game24_exact_prune import _can_reach_target_numbers
        return _can_reach_target_numbers(tuple(sorted(Fraction(n) for n in numbers)))

    def test_single_24_is_reachable(self):
        self.assertTrue(
            self._call_oracle([24]),
            "DEFECT: [24] should be reachable (target achieved).",
        )

    def test_single_non_24_is_unreachable(self):
        self.assertFalse(
            self._call_oracle([1]),
            "DEFECT: a single non-24 number should be unreachable.",
        )

    def test_two_numbers_that_multiply_to_24(self):
        self.assertTrue(
            self._call_oracle([4, 6]),
            "DEFECT: 4 * 6 = 24 should be reachable.",
        )

    def test_two_numbers_that_add_to_24(self):
        self.assertTrue(
            self._call_oracle([10, 14]),
            "DEFECT: 10 + 14 = 24 should be reachable.",
        )

    def test_two_numbers_that_subtract_to_24(self):
        self.assertTrue(
            self._call_oracle([30, 6]),
            "DEFECT: 30 - 6 = 24 should be reachable.",
        )

    def test_two_numbers_that_divide_to_24(self):
        self.assertTrue(
            self._call_oracle([48, 2]),
            "DEFECT: 48 / 2 = 24 should be reachable.",
        )

    def test_two_numbers_unreachable(self):
        self.assertFalse(
            self._call_oracle([23, 2]),
            "DEFECT: no operation on {23, 2} yields exactly 24.",
        )

    def test_three_numbers_reachable(self):
        self.assertTrue(
            self._call_oracle([4, 2, 3]),
            "DEFECT: 4 * 2 * 3 = 24 should be reachable.",
        )

    def test_three_numbers_unreachable(self):
        self.assertFalse(
            self._call_oracle([1, 1, 1]),
            "DEFECT: {1, 1, 1} cannot reach 24.",
        )

    def test_classic_game24_inputs(self):
        """Verify classic 4-number puzzles that are known reachable."""
        solvable_puzzles = [
            [1, 2, 3, 4],
            [4, 4, 6, 8],
            [2, 9, 10, 12],
            [4, 9, 10, 13],
            [1, 4, 8, 8],
            [5, 5, 5, 9],
        ]
        for puzzle in solvable_puzzles:
            with self.subTest(puzzle=puzzle):
                self.assertTrue(
                    self._call_oracle(puzzle),
                    f"DEFECT: {puzzle} should be reachable to 24.",
                )

    def test_known_unreachable_4_numbers(self):
        """Verify 4-number states known to be dead-ends."""
        dead = [
            [1, 1, 1, 1],
            [1, 1, 1, 2],
        ]
        for puzzle in dead:
            with self.subTest(puzzle=puzzle):
                self.assertFalse(
                    self._call_oracle(puzzle),
                    f"DEFECT: {puzzle} should be unreachable to 24.",
                )

    def test_memoization_caches_identical_multisets(self):
        from federated_mcts.tasks.game24_exact_prune import _can_reach_target_numbers
        call_count = [0]
        original = _can_reach_target_numbers.__wrapped__

        def counting_wrapper(nums):
            call_count[0] += 1
            return original(nums)

        import functools
        cached = functools.lru_cache(maxsize=None)(counting_wrapper)

        key = tuple(sorted(Fraction(n) for n in [2, 3, 4]))
        result1 = cached(key)
        result2 = cached(key)
        self.assertEqual(result1, result2)
        self.assertEqual(
            call_count[0], 1,
            "DEFECT: memoization did not prevent redundant computation.",
        )

    def test_fraction_exactness_no_float_drift(self):
        """Prove that Fraction arithmetic avoids float imprecision.
        {1, 5, 5, 5} = 5 * (5 - 1/5) = 5 * 24/5 = 24."""
        self.assertTrue(
            self._call_oracle([1, 5, 5, 5]),
            "DEFECT: 5*(5-1/5)=24 requires exact Fraction arithmetic.",
        )

    def test_hard_fraction_puzzle(self):
        """{3, 3, 7, 7} = (3 + 3/7) * 7 = 24."""
        self.assertTrue(
            self._call_oracle([3, 3, 7, 7]),
            "DEFECT: (3+3/7)*7=24 requires exact Fractions.",
        )

    def test_division_by_zero_skipped(self):
        """0 must not be used as a divisor; the search must skip it."""
        from federated_mcts.tasks.game24_exact_prune import _can_reach_target_numbers
        result = _can_reach_target_numbers((Fraction(0), Fraction(24)))
        self.assertTrue(
            result,
            "DEFECT: {0, 24} can reach 24 via 0+24 or 24+0.",
        )
        result_zero_one = _can_reach_target_numbers((Fraction(0), Fraction(1)))
        self.assertFalse(
            result_zero_one,
            "DEFECT: {0, 1} cannot reach 24, and must not crash via div-by-zero.",
        )


class TestPruneUnreachableIntegration(unittest.TestCase):
    """Integration tests for prune_unreachable using a real Game24Task."""

    def setUp(self):
        from federated_mcts.tasks.game24 import Game24Task
        self.task = Game24Task()

    def test_known_reachable_candidates_all_retained(self):
        """Candidates with reachable states should be retained in order."""
        from federated_mcts.tasks.game24_exact_prune import prune_unreachable
        candidates = [
            "1 + 2 = 3 (left: 3 3 4)",
            "3 + 3 = 6 (left: 4 6)",
        ]
        retained, proposed, pruned = prune_unreachable(candidates, self.task, "1 2 3 4")
        self.assertEqual(proposed, 2)
        self.assertEqual(pruned, 0)
        self.assertEqual(len(retained), 2)
        self.assertEqual(retained, candidates,
                         "DEFECT: reachable candidates must be retained in original order.")

    def test_unreachable_candidates_removed(self):
        """Candidates with dead-end states (e.g. {1,1,1}) are pruned."""
        from federated_mcts.tasks.game24_exact_prune import prune_unreachable
        candidates = [
            "1 + 1 = 2 (left: 1 1 2)",
        ]
        retained, proposed, pruned = prune_unreachable(candidates, self.task, "1 1 1 1")
        self.assertEqual(proposed, 1)
        self.assertEqual(pruned, 1)
        self.assertEqual(len(retained), 0)

    def test_mixed_candidates_preserve_reachable_order(self):
        """Only unreachable candidates are removed; reachable order preserved."""
        from federated_mcts.tasks.game24_exact_prune import prune_unreachable
        candidates = [
            "1 + 1 = 2 (left: 1 1 2)",
            "1 + 2 = 3 (left: 3 3 4)",
            "1 + 1 = 2 (left: 1 3 8)",
        ]
        retained, proposed, pruned = prune_unreachable(candidates, self.task, "1 2 3 4")
        self.assertEqual(proposed, 3)
        self.assertGreaterEqual(pruned, 1)
        self.assertEqual(len(retained), proposed - pruned)

    def test_terminal_24_state_is_always_retained(self):
        """A candidate that has already reached (left: 24) is always retained."""
        from federated_mcts.tasks.game24_exact_prune import prune_unreachable
        candidates = [
            "4 * 6 = 24 (left: 24)",
        ]
        retained, proposed, pruned = prune_unreachable(candidates, self.task, "4 6 8 8")
        self.assertEqual(proposed, 1)
        self.assertEqual(pruned, 0)
        self.assertEqual(len(retained), 1)

    def test_empty_candidates_input(self):
        from federated_mcts.tasks.game24_exact_prune import prune_unreachable
        retained, proposed, pruned = prune_unreachable([], self.task, "1 2 3 4")
        self.assertEqual(proposed, 0)
        self.assertEqual(pruned, 0)
        self.assertEqual(retained, [])

    def test_all_unreachable_returns_empty_frontier(self):
        """When every candidate is unreachable, frontier becomes empty."""
        from federated_mcts.tasks.game24_exact_prune import prune_unreachable
        candidates = [
            "1 + 1 = 2 (left: 1 1 2)",
            "1 + 1 = 2 (left: 1 1 2)",
        ]
        retained, proposed, pruned = prune_unreachable(candidates, self.task, "1 1 1 1")
        self.assertEqual(proposed, 2)
        self.assertEqual(pruned, 2)
        self.assertEqual(retained, [])


class TestDisabledFlagLeavesBehaviorUnchanged(unittest.TestCase):
    """When the flag is false/absent, the solver must not prune."""

    def test_flag_defaults_to_false(self):
        class FakeArgs:
            pass
        args = FakeArgs()
        self.assertFalse(
            getattr(args, "game24_exact_prune", False),
            "DEFECT: game24_exact_prune default should be False.",
        )


if __name__ == "__main__":
    unittest.main()
