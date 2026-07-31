"""Small MLP Q-network for the Budget-Aware DQN controller."""

from __future__ import annotations

import math

import torch
from torch import nn


class DQNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_count: int,
        hidden_sizes=(64, 64),
        seed: int | None = None,
    ):
        super().__init__()
        sizes = (state_dim,) + tuple(hidden_sizes) + (action_count,)
        layers: list[nn.Module] = []
        for in_features, out_features in zip(sizes[:-1], sizes[1:]):
            layers.append(nn.Linear(in_features, out_features))
            if out_features != action_count:
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)
        if seed is not None:
            self._seed_weights(seed)

    def _seed_weights(self, seed: int) -> None:
        generator = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for layer in self.layers:
                if isinstance(layer, nn.Linear):
                    bound = 1.0 / math.sqrt(layer.weight.shape[1])
                    layer.weight.uniform_(-bound, bound, generator=generator)
                    layer.bias.uniform_(-bound, bound, generator=generator)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        single = states.dim() == 1
        x = states.float()
        if single:
            x = x.unsqueeze(0)
        x = self.layers(x)
        return x.squeeze(0) if single else x
