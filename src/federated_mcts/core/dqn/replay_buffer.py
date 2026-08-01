"""Fixed-capacity replay buffer for Double-DQN training."""

from __future__ import annotations

import random

import torch


class ReplayBuffer:
    def __init__(self, capacity: int):
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = capacity
        self._states: list[torch.Tensor] = []
        self._actions: list[int] = []
        self._rewards: list[float] = []
        self._next_states: list[torch.Tensor | None] = []
        self._dones: list[bool] = []

    def push(self, *, state, action, reward, next_state, done) -> None:
        self._states.append(torch.as_tensor(state, dtype=torch.float32))
        self._actions.append(int(action))
        self._rewards.append(float(reward))
        self._next_states.append(
            None if next_state is None else torch.as_tensor(next_state, dtype=torch.float32)
        )
        self._dones.append(bool(done))
        if len(self._states) > self.capacity:
            for storage in (self._states, self._actions, self._rewards, self._next_states, self._dones):
                storage.pop(0)

    def __len__(self) -> int:
        return len(self._states)

    def sample(self, batch_size: int):
        count = len(self._states)
        if batch_size > count:
            raise ValueError(f"cannot sample {batch_size} from {count} buffered transitions")
        indices = random.sample(range(count), batch_size)
        state_dim = self._states[0].shape[0]
        states = torch.stack([self._states[i] for i in indices])
        actions = torch.tensor([self._actions[i] for i in indices], dtype=torch.long)
        rewards = torch.tensor([self._rewards[i] for i in indices], dtype=torch.float32)
        next_states = torch.stack([
            self._next_states[i] if self._next_states[i] is not None else torch.zeros(state_dim)
            for i in indices
        ])
        dones = torch.tensor([1.0 if self._dones[i] else 0.0 for i in indices], dtype=torch.float32)
        return states, actions, rewards, next_states, dones
