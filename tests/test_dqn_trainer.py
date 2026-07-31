"""Double-DQN trainer contract.

train_step() must produce a finite, positive loss and drive Q toward the
TD target when gamma is 0.  The target network must be a separate network
that lags the online network until sync_target().  build_trainer() and the
planned train_main() CLI entry must be seed-deterministic.
"""

import math
import sys
import unittest

sys.path.insert(0, "src")

import torch

from federated_mcts.core.dqn.network import DQNetwork
from federated_mcts.core.dqn.replay_buffer import ReplayBuffer
from federated_mcts.core.dqn.trainer import (
    DoubleDQNTrainer,
    build_trainer,
    train_main,
)

_STATE_DIM = 12
_ACTION_COUNT = 8


def _fill(buffer, count=64):
    for index in range(count):
        buffer.push(
            state=torch.full((_STATE_DIM,), float(index % 10) / 10.0),
            action=index % _ACTION_COUNT,
            reward=1.0 if index % 7 == 0 else 0.0,
            next_state=torch.full((_STATE_DIM,), float((index + 1) % 10) / 10.0),
            done=index % 9 == 0,
        )


def _trainer(buffer, gamma=0.99, batch_size=32):
    torch.manual_seed(0)
    return DoubleDQNTrainer(
        q_network=DQNetwork(_STATE_DIM, _ACTION_COUNT, seed=0),
        target_network=DQNetwork(_STATE_DIM, _ACTION_COUNT, seed=0),
        learning_rate=1e-3,
        gamma=gamma,
        tau=0.005,
        replay_buffer=buffer,
        batch_size=batch_size,
    )


class TestDoubleDQNTrainer(unittest.TestCase):
    def test_train_step_produces_finite_positive_loss(self):
        """Given a filled replay buffer, when train_step() runs, then the
        returned Double-DQN loss is finite and positive."""
        buffer = ReplayBuffer(256)
        _fill(buffer)
        trainer = _trainer(buffer)

        loss = trainer.train_step()

        self.assertTrue(math.isfinite(loss))
        self.assertGreater(loss, 0.0)

    def test_gamma_zero_repeated_steps_decrease_loss(self):
        """Given gamma=0 (fixed target = reward) and the same batch, when
        train_step() runs twice, then the second loss is strictly lower."""
        buffer = ReplayBuffer(64)
        _fill(buffer, count=32)
        trainer = _trainer(buffer, gamma=0.0, batch_size=32)

        loss_first = trainer.train_step()
        loss_second = trainer.train_step()

        self.assertLess(loss_second, loss_first)

    def test_target_network_lags_until_sync(self):
        """Given an online and a target network seeded differently, when
        compared before any update, then their outputs differ; after
        sync_target() they agree."""
        torch.manual_seed(0)
        trainer = DoubleDQNTrainer(
            q_network=DQNetwork(_STATE_DIM, _ACTION_COUNT, seed=1),
            target_network=DQNetwork(_STATE_DIM, _ACTION_COUNT, seed=2),
            learning_rate=1e-3,
            gamma=0.99,
            tau=1.0,
            replay_buffer=ReplayBuffer(8),
            batch_size=4,
        )
        states = torch.randn(3, _STATE_DIM)

        self.assertFalse(torch.allclose(trainer.q_network(states), trainer.target_network(states)))

        trainer.sync_target()

        self.assertTrue(torch.allclose(trainer.q_network(states), trainer.target_network(states)))


class TestTrainerEntryPoints(unittest.TestCase):
    def test_build_trainer_is_seed_deterministic(self):
        """Given the same seed twice and a different seed once, when
        build_trainer() constructs networks, then equal seeds yield identical
        Q outputs and a different seed diverges."""
        states = torch.randn(2, _STATE_DIM)
        first = build_trainer(state_dim=_STATE_DIM, action_count=_ACTION_COUNT, seed=11)
        second = build_trainer(state_dim=_STATE_DIM, action_count=_ACTION_COUNT, seed=11)
        other = build_trainer(state_dim=_STATE_DIM, action_count=_ACTION_COUNT, seed=12)

        self.assertTrue(torch.allclose(first.q_network(states), second.q_network(states)))
        self.assertFalse(torch.allclose(first.q_network(states), other.q_network(states)))

    def test_train_main_parses_cli_args_and_builds_trainer(self):
        """Given a seeded train_main invocation, when the trainer is built,
        then it owns a Q network of the requested action count."""
        trainer = train_main(["--seed", "42", "--state_dim", "12", "--action_count", "8"])

        self.assertEqual(trainer.q_network(torch.zeros(1, _STATE_DIM)).shape, (1, _ACTION_COUNT))


if __name__ == "__main__":
    unittest.main()
