"""Budget-Aware DQN controller: seeded epsilon-greedy policy over the 8-action
beam/joint-rank space with checkpoint-driven collection-only startup.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from federated_mcts.core.dqn.actions import ACTION_COUNT, beam_and_joint_rank
from federated_mcts.core.dqn.checkpoint import (
    CheckpointConfigurationError,
    CheckpointStatus,
    load_checkpoint,
)
from federated_mcts.core.dqn.features import extract_state_features
from federated_mcts.core.dqn.network import DQNetwork
from federated_mcts.core.dqn.replay_buffer import ReplayBuffer


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return value


@dataclass(frozen=True)
class ActionDecision:
    action: int
    beam: int
    joint_rank: bool
    state: np.ndarray


class BudgetAwareDQNController:
    def __init__(
        self,
        *,
        state_dim: int = 12,
        action_count: int = ACTION_COUNT,
        epsilon: float = 1.0,
        seed: int = 0,
        checkpoint_path: str | None = None,
        collect_without_checkpoint: bool = True,
        hidden_sizes=(64, 64),
        capacity: int = 10000,
    ):
        self.state_dim = state_dim
        self.action_count = action_count
        self.epsilon = float(epsilon)
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self.q_network = DQNetwork(state_dim, action_count, hidden_sizes=hidden_sizes, seed=seed)
        self.replay_buffer = ReplayBuffer(capacity)
        self._transitions: list[dict] = []
        self.checkpoint_status = CheckpointStatus.COLLECTION_ONLY
        self.training_active = False
        if checkpoint_path is not None:
            result = load_checkpoint(
                checkpoint_path,
                state_dim=state_dim,
                action_count=action_count,
                hidden_sizes=hidden_sizes,
            )
            if result.status is CheckpointStatus.RESTORED:
                if result.state_dict is None:
                    raise CheckpointConfigurationError("restored checkpoint has no model state")
                self.q_network.load_state_dict(result.state_dict["model"])
                self.checkpoint_status = CheckpointStatus.RESTORED
                self.training_active = True
            elif not collect_without_checkpoint:
                raise CheckpointConfigurationError(
                    f"checkpoint {checkpoint_path!r} is missing and collection "
                    "without a checkpoint is not permitted"
                )

    @property
    def collection_only(self) -> bool:
        return self.checkpoint_status is CheckpointStatus.COLLECTION_ONLY

    def _effective_epsilon(self, epsilon: float | None) -> float:
        eps = self.epsilon if epsilon is None else float(epsilon)
        # While collecting data (no restored policy) keep exploring at >= 0.5
        # unless the caller explicitly overrides epsilon for this draw.
        if epsilon is None and self.collection_only:
            eps = max(eps, 0.5)
        return eps

    def choose_action(self, state, epsilon: float | None = None) -> int:
        eps = self._effective_epsilon(epsilon)
        if self._rng.random() < eps:
            return int(self._rng.integers(0, self.action_count))
        with torch.no_grad():
            q_values = self.q_network(torch.as_tensor(state, dtype=torch.float32))
        return int(q_values.argmax(dim=-1).item())

    def decide(
        self,
        *,
        candidates,
        states,
        previous_values,
        step: int,
        total_steps: int,
        tokens_consumed: float,
        token_budget: float,
        latency_consumed: float,
        latency_budget: float,
        previous_joint_rank: bool | None = None,
        task_optimal_length: int | None = None,
        current_remaining_distance: int | None = None,
        previous_beam_width: int | None = None,
    ) -> ActionDecision:
        state = extract_state_features(
            candidates=candidates,
            states=states,
            previous_values=previous_values,
            step=step,
            total_steps=total_steps,
            tokens_consumed=tokens_consumed,
            token_budget=token_budget,
            latency_consumed=latency_consumed,
            latency_budget=latency_budget,
            previous_joint_rank=previous_joint_rank,
            task_optimal_length=task_optimal_length,
            current_remaining_distance=current_remaining_distance,
            previous_beam_width=previous_beam_width,
        )
        if state.shape[0] != self.state_dim:
            padded = np.zeros(self.state_dim, dtype=np.float32)
            size = min(state.shape[0], self.state_dim)
            padded[:size] = state[:size]
            state = padded
        action = self.choose_action(state)
        beam, joint_rank = beam_and_joint_rank(action)
        return ActionDecision(action=action, beam=beam, joint_rank=joint_rank, state=state)

    def record_transition(self, *, state, action, reward, next_state, done) -> None:
        beam, joint_rank = beam_and_joint_rank(action)
        entry = {
            "state": _plain(state),
            "action": int(action),
            "reward": float(reward),
            "next_state": None if next_state is None else _plain(next_state),
            "done": bool(done),
            "beam": int(beam),
            "joint_rank": bool(joint_rank),
        }
        self._transitions.append(entry)
        self.replay_buffer.push(
            state=torch.as_tensor(state, dtype=torch.float32),
            action=int(action),
            reward=float(reward),
            next_state=None if next_state is None else torch.as_tensor(next_state, dtype=torch.float32),
            done=bool(done),
        )

    def export_transitions(self) -> list[dict]:
        return list(self._transitions)
