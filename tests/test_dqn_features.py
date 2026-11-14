"""Feature-extraction contract for the Budget-Aware DQN controller.

The pre-decision state vector must be computable from candidate/dedup
structure, previous-step value stats and consumed budgets ONLY.  The action
is chosen from this state, so the current step's ranking values must never
leak into it.  The vector is finite, normalized into [0, 1] and has 12 dims
(13 when the previous joint-rank flag is included).
"""

import sys
import unittest

sys.path.insert(0, "src")

import numpy as np

from federated_mcts.core.dqn.features import (
    MAX_CANDIDATE_COUNT,
    extract_state_features,
)

# indices of the 16-dim base vector (index 11 is the previous-step range)
_STRUCTURAL = (0, 1, 2)
_VALUE_STATS = (3, 4, 5, 6, 7, 11)


def _extract(**kwargs):
    base = dict(
        candidates=["a", "b"],
        states=[("a",), ("b",)],
        previous_values=[0.8, 0.6],
        step=1,
        total_steps=3,
        tokens_consumed=100,
        token_budget=1000,
        latency_consumed=1.0,
        latency_budget=10.0,
    )
    base.update(kwargs)
    return extract_state_features(**base)


class TestStateFeatureExtraction(unittest.TestCase):
    def test_default_state_is_16_dim_finite_and_normalized(self):
        """Given a representative pre-decision structure, when the state is
        extracted, then it has 12 dims, all finite and inside [0, 1]."""
        state = _extract()

        self.assertEqual(state.shape, (16,))
        self.assertTrue(np.all(np.isfinite(state)))
        self.assertTrue(np.all((state >= 0.0) & (state <= 1.0)))

    def test_optional_joint_rank_flag_extends_to_17_dim(self):
        """Given include of the previous joint-rank flag, when the state is
        extracted, then the vector has 17 dims and the flag feature is 1.0."""
        state = _extract(previous_joint_rank=True)

        self.assertEqual(state.shape, (17,))
        self.assertEqual(state[16], 1.0)

    def test_structural_features_do_not_depend_on_value_stats(self):
        """Given identical candidate/dedup structure, when only the previous
        value stats differ, then the structural features are unchanged while
        the value-stat features move."""
        low = _extract(previous_values=[0.1, 0.1])
        high = _extract(previous_values=[0.9, 0.9])

        np.testing.assert_array_equal(low[_STRUCTURAL], high[_STRUCTURAL])
        self.assertTrue(np.any(low[_VALUE_STATS] != high[_VALUE_STATS]))

    def test_candidate_counts_are_normalized_against_max(self):
        """Given 1 candidate versus MAX_CANDIDATE_COUNT candidates, when the
        state is extracted, then feature 0 is 1/MAX and 1 respectively."""
        sparse = _extract(candidates=["a"], states=[("a",)])
        dense_candidates = [f"c{i}" for i in range(MAX_CANDIDATE_COUNT)]
        dense = _extract(
            candidates=dense_candidates,
            states=[(name,) for name in dense_candidates],
        )

        self.assertEqual(sparse[0], 1.0 / MAX_CANDIDATE_COUNT)
        self.assertEqual(dense[0], 1.0)

    def test_duplicate_ratio_is_zero_for_unique_candidates(self):
        """Given only unique states, when the state is extracted, then the
        duplicate-ratio feature is exactly zero."""
        state = _extract()

        self.assertEqual(state[2], 0.0)

    def test_empty_candidates_and_zero_budgets_stay_finite(self):
        """Given empty candidates, absent values and zero budgets, when the
        state is extracted, then no NaN is produced and the affected features
        are exactly zero."""
        state = _extract(
            candidates=[], states=[], previous_values=None,
            tokens_consumed=0, token_budget=0,
            latency_consumed=0.0, latency_budget=0.0,
        )

        self.assertTrue(np.all(np.isfinite(state)))
        self.assertEqual(state[0], 0.0)
        self.assertEqual(state[1], 0.0)
        self.assertEqual(state[2], 0.0)
        self.assertEqual(state[9], 0.0)
        self.assertEqual(state[10], 0.0)

    def test_previous_value_stats_default_to_zero_when_absent(self):
        """Given no previous-step values, when the state is extracted, then
        every previous-value statistic is exactly zero."""
        state = _extract(previous_values=None)

        for index in _VALUE_STATS:
            self.assertEqual(state[index], 0.0)


if __name__ == "__main__":
    unittest.main()
