"""Session construction from solver arguments for the DQN search session."""

from __future__ import annotations

from federated_mcts.core.dqn.actions import ACTION_COUNT
from federated_mcts.core.dqn.controller import BudgetAwareDQNController
from federated_mcts.core.dqn.session import DqnSearchSession


def build_dqn_session(args, *, jsonl_path: str | None = None) -> DqnSearchSession:
    controller = BudgetAwareDQNController(
        state_dim=getattr(args, "dqn_state_dim", 12),
        action_count=ACTION_COUNT,
        epsilon=getattr(args, "dqn_epsilon", 1.0),
        seed=getattr(args, "dqn_seed", getattr(args, "seed", 0)),
        checkpoint_path=getattr(args, "dqn_checkpoint", None),
        capacity=getattr(args, "dqn_capacity", 10000),
    )
    return DqnSearchSession(
        controller,
        token_budget=getattr(args, "dqn_token_budget", 5000.0),
        latency_budget=getattr(args, "dqn_latency_budget", 60.0),
        jsonl_path=jsonl_path if jsonl_path is not None else getattr(args, "dqn_jsonl", None),
    )
