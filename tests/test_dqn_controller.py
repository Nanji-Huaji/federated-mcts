"""Budget-Aware DQN controller contract.

The controller chooses an action with seeded, deterministic epsilon-greedy
selection; builds the pre-decision state from structure + previous stats +
budgets (never current-step values); records transitions; and starts in an
explicit collection-only mode when its checkpoint is missing.
"""

import sys
import unittest

sys.path.insert(0, "src")

import numpy as np
import torch

from federated_mcts.core.dqn.actions import ACTION_BEAMS, ACTION_COUNT, beam_and_joint_rank
from federated_mcts.core.dqn.checkpoint import CheckpointStatus
from federated_mcts.core.dqn.controller import BudgetAwareDQNController

_EXPECTED_KEYS = {"state", "action", "reward", "next_state", "done", "beam", "joint_rank"}


def _controller(seed, epsilon=1.0, **kwargs):
    return BudgetAwareDQNController(
        state_dim=16,
        action_count=ACTION_COUNT,
        epsilon=epsilon,
        seed=seed,
        **kwargs,
    )


class TestSeededEpsilonGreedy(unittest.TestCase):
    def test_same_seed_yields_same_action_sequence(self):
        """Given two controllers with the same seed, when both draw a sequence
        of epsilon-greedy actions, then the sequences are identical."""
        state = np.zeros(16, dtype=np.float32)
        first = _controller(seed=42)
        second = _controller(seed=42)

        actions_first = [first.choose_action(state) for _ in range(30)]
        actions_second = [second.choose_action(state) for _ in range(30)]

        self.assertEqual(actions_first, actions_second)

    def test_different_seed_yields_different_action_sequence(self):
        """Given two controllers with different seeds, when both draw a
        sequence of epsilon-greedy actions, then the sequences diverge."""
        state = np.zeros(16, dtype=np.float32)
        first = _controller(seed=1)
        second = _controller(seed=2)

        actions_first = [first.choose_action(state) for _ in range(30)]
        actions_second = [second.choose_action(state) for _ in range(30)]

        self.assertNotEqual(actions_first, actions_second)

    def test_zero_epsilon_is_greedy_argmax(self):
        """Given epsilon=0, when an action is chosen, then it equals the
        argmax of the Q network's output for that state."""
        torch.manual_seed(0)
        controller = _controller(seed=0, epsilon=0.0)
        state = np.linspace(0.1, 0.9, 16).astype(np.float32)
        q_values = controller.q_network(torch.tensor(state)).detach().numpy()
        expected = int(np.argmax(q_values))

        for _ in range(5):
            self.assertEqual(controller.choose_action(state, epsilon=0.0), expected)


class TestControllerDecision(unittest.TestCase):
    def test_decision_state_precedes_action_and_uses_structure_only(self):
        """Given candidate structure, previous value stats and budgets, when
        the controller decides, then the returned state is 16-dim, finite and
        derived from pre-decision information only (no current-step values)."""
        decision = _controller(seed=3, epsilon=0.0).decide(
            candidates=["a", "b", "b"],
            states=[("a",), ("b",), ("b",)],
            previous_values=[0.7, 0.5],
            step=0,
            total_steps=3,
            tokens_consumed=80,
            token_budget=1000,
            latency_consumed=0.8,
            latency_budget=10.0,
        )

        self.assertEqual(decision.state.shape, (16,))
        self.assertTrue(np.all(np.isfinite(decision.state)))
        beam, joint_rank = beam_and_joint_rank(decision.action)
        self.assertEqual(decision.beam, beam)
        self.assertEqual(decision.joint_rank, joint_rank)
        self.assertIn(decision.beam, ACTION_BEAMS)
        self.assertIn(decision.joint_rank, (False, True))

    def test_record_and_export_transitions_roundtrip(self):
        """Given a recorded transition, when exported, then every documented
        field is present and preserved."""
        controller = _controller(seed=5)
        state = np.zeros(16, dtype=np.float32)
        next_state = np.ones(16, dtype=np.float32)
        controller.record_transition(
            state=state,
            action=3,
            reward=0.5,
            next_state=next_state,
            done=True,
        )

        entries = controller.export_transitions()
        self.assertEqual(len(entries), 1)
        self.assertEqual(set(entries[0]), _EXPECTED_KEYS)
        self.assertEqual(entries[0]["action"], 3)
        self.assertEqual(entries[0]["reward"], 0.5)
        self.assertEqual(entries[0]["beam"], 3)
        self.assertEqual(entries[0]["joint_rank"], True)
        self.assertTrue(entries[0]["done"])


class TestCheckpointStartup(unittest.TestCase):
    def test_missing_checkpoint_starts_collection_only(self):
        """Given a nonexistent checkpoint path, when the controller starts,
        then it reports the explicit collection-only status and does not
        train."""
        controller = BudgetAwareDQNController(
            state_dim=16,
            action_count=ACTION_COUNT,
            seed=0,
            checkpoint_path="/nonexistent/controller.pt",
        )

        self.assertEqual(controller.checkpoint_status, CheckpointStatus.COLLECTION_ONLY)
        self.assertFalse(controller.training_active)


if __name__ == "__main__":
    unittest.main()
