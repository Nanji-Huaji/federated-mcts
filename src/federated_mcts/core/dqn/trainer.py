"""Double-DQN trainer: finite-loss updates with a lagging target network."""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from federated_mcts.core.dqn.network import DQNetwork
from federated_mcts.core.dqn.replay_buffer import ReplayBuffer


class DoubleDQNTrainer:
    def __init__(
        self,
        q_network,
        target_network,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,
        replay_buffer: ReplayBuffer | None = None,
        batch_size: int = 32,
    ):
        self.q_network = q_network
        self.target_network = target_network
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(q_network.parameters(), lr=learning_rate)

    def train_step(self) -> float:
        if self.replay_buffer is None:
            raise RuntimeError("replay buffer is required for training")
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        current = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(dim=1)
            targets = (
                rewards
                + self.gamma
                * self.target_network(next_states)
                .gather(1, next_actions.unsqueeze(1))
                .squeeze(1)
                * (1.0 - dones)
            )
        loss = F.mse_loss(current, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self._soft_update()
        return float(loss.item())

    def _soft_update(self) -> None:
        with torch.no_grad():
            for online, target in zip(self.q_network.parameters(), self.target_network.parameters()):
                target.mul_(1.0 - self.tau)
                target.add_(online, alpha=self.tau)

    def sync_target(self) -> None:
        self.target_network.load_state_dict(self.q_network.state_dict())


def build_trainer(
    *,
    state_dim: int = 12,
    action_count: int = 8,
    seed: int = 0,
    hidden_sizes=(64, 64),
    learning_rate: float = 1e-3,
    gamma: float = 0.99,
    tau: float = 0.005,
    capacity: int = 10000,
    batch_size: int = 32,
) -> DoubleDQNTrainer:
    q_network = DQNetwork(state_dim, action_count, hidden_sizes=hidden_sizes, seed=seed)
    target_network = DQNetwork(state_dim, action_count, hidden_sizes=hidden_sizes, seed=seed)
    return DoubleDQNTrainer(
        q_network=q_network,
        target_network=target_network,
        learning_rate=learning_rate,
        gamma=gamma,
        tau=tau,
        replay_buffer=ReplayBuffer(capacity),
        batch_size=batch_size,
    )


def train_main(argv=None) -> DoubleDQNTrainer:
    """Planned trainer CLI entry: parse arguments and build the trainer
    without touching a network or the API (offline construction only)."""
    parser = argparse.ArgumentParser(description="Build a Double-DQN trainer (offline).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--state_dim", type=int, default=12)
    parser.add_argument("--action_count", type=int, default=8)
    parser.add_argument("--hidden", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--capacity", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    args = parser.parse_args(argv)
    return build_trainer(
        state_dim=args.state_dim,
        action_count=args.action_count,
        seed=args.seed,
        hidden_sizes=tuple(args.hidden),
        capacity=args.capacity,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        tau=args.tau,
    )
