"""Budget-Aware DQN controller for the federated MCTS solver."""

from federated_mcts.core.dqn.actions import (
    ACTION_BEAMS,
    ACTION_COUNT,
    ACTION_JOINT_RANKS,
    action_for,
    beam_and_joint_rank,
)
from federated_mcts.core.dqn.checkpoint import (
    CheckpointConfigurationError,
    CheckpointMetadataError,
    CheckpointStatus,
    load_checkpoint,
    save_checkpoint,
)
from federated_mcts.core.dqn.controller import BudgetAwareDQNController
from federated_mcts.core.dqn.factory import build_dqn_session
from federated_mcts.core.dqn.features import extract_state_features
from federated_mcts.core.dqn.network import DQNetwork
from federated_mcts.core.dqn.recorder import JSONLTransitionRecorder
from federated_mcts.core.dqn.replay_buffer import ReplayBuffer
from federated_mcts.core.dqn.rewards import (
    correctness_reward,
    latency_penalty,
    rewards_for_episode,
    token_penalty,
)
from federated_mcts.core.dqn.session import DqnSearchSession
from federated_mcts.core.dqn.step import DqnStepOutcome
from federated_mcts.core.dqn.trainer import DoubleDQNTrainer, build_trainer, train_main

__all__ = [
    "ACTION_BEAMS",
    "ACTION_COUNT",
    "ACTION_JOINT_RANKS",
    "action_for",
    "beam_and_joint_rank",
    "BudgetAwareDQNController",
    "CheckpointConfigurationError",
    "CheckpointMetadataError",
    "CheckpointStatus",
    "correctness_reward",
    "DoubleDQNTrainer",
    "DQNetwork",
    "DqnSearchSession",
    "build_dqn_session",
    "build_trainer",
    "extract_state_features",
    "JSONLTransitionRecorder",
    "latency_penalty",
    "load_checkpoint",
    "ReplayBuffer",
    "rewards_for_episode",
    "save_checkpoint",
    "token_penalty",
    "train_main",
]
