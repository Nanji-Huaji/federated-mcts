"""Reward-shaping contract for the Budget-Aware DQN controller.

The terminal correctness reward (exact Game24 success) is assigned ONLY to
the final transition of an episode; every prior transition keeps a zero
correctness base.  Token and latency penalties are monotonic non-decreasing
in consumption and bounded to [0, 1].
"""

import sys
import unittest

sys.path.insert(0, "src")

from federated_mcts.core.dqn.rewards import (
    correctness_reward,
    latency_penalty,
    rewards_for_episode,
    token_penalty,
)


class TestTerminalCorrectnessReward(unittest.TestCase):
    def test_correctness_reward_is_exact_one_or_zero(self):
        """Given a success verdict, when the correctness reward is computed,
        then it is exactly 1.0 for success and 0.0 for failure."""
        self.assertEqual(correctness_reward(True), 1.0)
        self.assertEqual(correctness_reward(False), 0.0)

    def test_terminal_reward_only_on_final_transition(self):
        """Given a 3-transition successful episode, when rewards are shaped,
        then only the final transition carries the correctness reward and the
        prior transitions keep a zero base minus penalties."""
        transitions = [
            {"tokens": 10, "latency": 0.1},
            {"tokens": 30, "latency": 0.3},
            {"tokens": 60, "latency": 0.6},
        ]

        rewards = rewards_for_episode(transitions, terminal_success=True)

        self.assertEqual(len(rewards), 3)
        self.assertAlmostEqual(rewards[0], 0.0 - token_penalty(10) - latency_penalty(0.1))
        self.assertAlmostEqual(rewards[1], 0.0 - token_penalty(30) - latency_penalty(0.3))
        self.assertAlmostEqual(rewards[2], 1.0 - token_penalty(60) - latency_penalty(0.6))

    def test_failed_episode_final_transition_gets_zero_correctness(self):
        """Given a failed single-transition episode, when rewards are shaped,
        then the final transition receives a zero correctness base."""
        rewards = rewards_for_episode(
            [{"tokens": 5, "latency": 0.1}],
            terminal_success=False,
        )

        self.assertAlmostEqual(rewards[0], 0.0 - token_penalty(5) - latency_penalty(0.1))


class TestMonotonicPenalties(unittest.TestCase):
    def test_token_penalty_is_monotonic_non_decreasing(self):
        """Given increasing token counts, when the token penalty is computed,
        then it never decreases and stays bounded in [0, 1]."""
        samples = list(range(0, 101, 10))
        penalties = [token_penalty(tokens) for tokens in samples]

        self.assertEqual(penalties, sorted(penalties))
        self.assertEqual(token_penalty(0), 0.0)
        self.assertLessEqual(penalties[-1], 1.0)

    def test_latency_penalty_is_monotonic_non_decreasing(self):
        """Given increasing latencies, when the latency penalty is computed,
        then it never decreases and stays bounded in [0, 1]."""
        samples = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
        penalties = [latency_penalty(seconds) for seconds in samples]

        self.assertEqual(penalties, sorted(penalties))
        self.assertEqual(latency_penalty(0.0), 0.0)
        self.assertLessEqual(penalties[-1], 1.0)

    def test_penalties_are_non_negative_and_bounded(self):
        """Given arbitrary consumption values, when penalties are computed,
        then they never fall outside [0, 1]."""
        for tokens in (0, 7, 50, 500):
            self.assertTrue(0.0 <= token_penalty(tokens) <= 1.0)
        for seconds in (0.0, 0.3, 5.0):
            self.assertTrue(0.0 <= latency_penalty(seconds) <= 1.0)


if __name__ == "__main__":
    unittest.main()
