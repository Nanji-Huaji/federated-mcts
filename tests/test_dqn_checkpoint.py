"""Checkpoint contract for the Budget-Aware DQN controller.

A missing checkpoint file loads into an explicit collection-only status (no
exception).  A checkpoint whose recorded metadata is incompatible with the
requested network shape raises the typed CheckpointMetadataError.  A
roundtrip through save/load preserves Q-network outputs exactly.
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, "src")

import torch

from federated_mcts.core.dqn.checkpoint import (
    CheckpointMetadataError,
    CheckpointStatus,
    load_checkpoint,
    save_checkpoint,
)
from federated_mcts.core.dqn.network import DQNetwork

_STATE_DIM = 12
_ACTION_COUNT = 8
_HIDDEN = (64, 64)
_METADATA = {
    "state_dim": _STATE_DIM,
    "action_count": _ACTION_COUNT,
    "hidden_sizes": [64, 64],
}


def _save(tmp, metadata=None, net=None):
    path = os.path.join(tmp, "controller.pt")
    save_checkpoint(
        path,
        model_state=(net or DQNetwork(_STATE_DIM, _ACTION_COUNT, seed=0)).state_dict(),
        metadata=metadata if metadata is not None else _METADATA,
    )
    return path


class TestCheckpointLoading(unittest.TestCase):
    def test_missing_checkpoint_returns_collection_only_state(self):
        """Given a nonexistent checkpoint path, when loaded, then the result
        is the explicit collection-only status with no network state."""
        result = load_checkpoint(
            "/nonexistent/controller.pt",
            state_dim=_STATE_DIM,
            action_count=_ACTION_COUNT,
            hidden_sizes=_HIDDEN,
        )

        self.assertEqual(result.status, CheckpointStatus.COLLECTION_ONLY)
        self.assertIsNone(result.state_dict)
        self.assertIsNone(result.metadata)

    def test_incompatible_state_dim_raises_typed_error(self):
        """Given a checkpoint trained with state_dim 12, when loaded with
        state_dim 13, then CheckpointMetadataError is raised."""
        with tempfile.TemporaryDirectory() as tmp:
            path = _save(tmp)
            with self.assertRaises(CheckpointMetadataError):
                load_checkpoint(
                    path,
                    state_dim=13,
                    action_count=_ACTION_COUNT,
                    hidden_sizes=_HIDDEN,
                )

    def test_incompatible_action_count_raises_typed_error(self):
        """Given a checkpoint trained with 8 actions, when loaded with 4
        actions, then CheckpointMetadataError is raised."""
        with tempfile.TemporaryDirectory() as tmp:
            path = _save(tmp)
            with self.assertRaises(CheckpointMetadataError):
                load_checkpoint(
                    path,
                    state_dim=_STATE_DIM,
                    action_count=4,
                    hidden_sizes=_HIDDEN,
                )

    def test_incompatible_hidden_sizes_raise_typed_error(self):
        """Given a checkpoint with hidden sizes [64, 64], when loaded with a
        different architecture, then CheckpointMetadataError is raised."""
        with tempfile.TemporaryDirectory() as tmp:
            path = _save(tmp)
            with self.assertRaises(CheckpointMetadataError):
                load_checkpoint(
                    path,
                    state_dim=_STATE_DIM,
                    action_count=_ACTION_COUNT,
                    hidden_sizes=(128, 64),
                )

    def test_checkpoint_roundtrip_preserves_q_outputs(self):
        """Given a trained Q network, when its state is saved and loaded into
        a freshly-seeded network, then the Q outputs are identical."""
        torch.manual_seed(0)
        states = torch.tensor(
            [
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.25, 0.75],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.5],
            ],
            dtype=torch.float32,
        )
        source = DQNetwork(_STATE_DIM, _ACTION_COUNT, seed=7)
        q_before = source(states)

        with tempfile.TemporaryDirectory() as tmp:
            path = _save(tmp, net=source)
            result = load_checkpoint(
                path,
                state_dim=_STATE_DIM,
                action_count=_ACTION_COUNT,
                hidden_sizes=_HIDDEN,
            )
            self.assertEqual(result.status, CheckpointStatus.RESTORED)
            restored = DQNetwork(_STATE_DIM, _ACTION_COUNT, seed=99)
            restored.load_state_dict(result.state_dict["model"])

        q_after = restored(states)
        self.assertTrue(torch.allclose(q_before, q_after, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
