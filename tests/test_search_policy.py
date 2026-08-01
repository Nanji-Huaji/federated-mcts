from fractions import Fraction
import sys
import unittest

sys.path.insert(0, "src")

from federated_mcts.core.search_policy import (
    adaptive_beam_width,
    deduplicate_candidates,
    diverse_select,
)
from federated_mcts.tasks.game24 import Game24Task


class _StateTask:
    @staticmethod
    def canonical_state_key(x, y):
        return tuple(sorted(y.split()))


class TestCanonicalStateDeduplication(unittest.TestCase):
    def test_game24_key_normalizes_order_and_rationals(self):
        task = Game24Task.__new__(Game24Task)

        first = task.canonical_state_key("1 2 3 4", "1 / 2 = 0.5 (left: 3 4 1/2)\n")
        second = task.canonical_state_key("1 2 3 4", "2 / 4 = 0.5 (left: 0.5 4 3)\n")

        self.assertEqual(first, second)
        self.assertEqual(first, (Fraction(1, 2), Fraction(3), Fraction(4)))

    def test_dedup_keeps_first_representative_per_state(self):
        candidates = ["1 2", "2 1", "1 3"]

        unique, states = deduplicate_candidates(_StateTask(), "x", candidates)

        self.assertEqual(unique, ["1 2", "1 3"])
        self.assertEqual(states, [("1", "2"), ("1", "3")])

    def test_game24_success_requires_a_valid_trajectory(self):
        task = Game24Task.__new__(Game24Task)
        invalid = "1 + 1 = 24 (left: 24)\n"
        valid = (
            "1 + 2 = 3 (left: 3 3 4)\n"
            "3 + 3 = 6 (left: 4 6)\n"
            "6 * 4 = 24 (left: 24)\n"
        )

        self.assertFalse(task.is_success_state("1 2 3 4", invalid))
        self.assertTrue(task.is_success_state("1 2 3 4", valid))


class TestAdaptiveDiverseSelection(unittest.TestCase):
    def test_tied_cutoff_expands_beam(self):
        width = adaptive_beam_width(
            [1.0, 1.0, 1.0, 0.5], base_width=2, max_expansion=2,
            uncertainty_margin=0.1, confidence_margin=0.8,
        )

        self.assertEqual(width, 4)

    def test_high_confidence_contracts_beam(self):
        width = adaptive_beam_width(
            [2.0, 0.5, 0.4], base_width=3, max_expansion=2,
            uncertainty_margin=0.1, confidence_margin=1.0,
        )

        self.assertEqual(width, 2)

    def test_mmr_prefers_novel_state_over_duplicate(self):
        candidates = ["best", "duplicate", "novel"]
        states = [("1", "2"), ("1", "2"), ("8", "9")]

        selected = diverse_select(
            candidates, [1.0, 0.95, 0.8], states, width=2, diversity_weight=0.5,
        )

        self.assertEqual(selected, [0, 2])


if __name__ == "__main__":
    unittest.main()
