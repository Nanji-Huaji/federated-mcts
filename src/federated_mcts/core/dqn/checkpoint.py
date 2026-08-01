"""Checkpoint persistence for the Budget-Aware DQN controller.

A missing checkpoint loads into an explicit collection-only status.  A
checkpoint whose metadata does not match the requested network shape raises
the typed CheckpointMetadataError.  Loading uses safe Torch patterns: a CPU
map_location and weights-only deserialization.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum

import torch


class CheckpointMetadataError(Exception):
    """Raised when checkpoint metadata is incompatible with the request."""


class CheckpointConfigurationError(Exception):
    """Raised when a checkpoint is required but missing and collection
    without a checkpoint is not permitted."""


class CheckpointStatus(Enum):
    COLLECTION_ONLY = "collection_only"
    RESTORED = "restored"


@dataclass(frozen=True)
class CheckpointLoadResult:
    status: CheckpointStatus
    state_dict: dict | None
    metadata: dict | None


def save_checkpoint(
    path: str,
    *,
    model_state: dict,
    metadata: dict,
    target_state: dict | None = None,
    optimizer_state: dict | None = None,
) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    payload = {"metadata": metadata, "model": model_state}
    if target_state is not None:
        payload["target_model"] = target_state
    if optimizer_state is not None:
        payload["optimizer"] = optimizer_state
    torch.save(payload, path)


def _metadata_mismatch(metadata: dict, *, state_dim: int, action_count: int, hidden_sizes) -> bool:
    return (
        metadata.get("state_dim") != state_dim
        or metadata.get("action_count") != action_count
        or list(metadata.get("hidden_sizes") or []) != list(hidden_sizes)
    )


def load_checkpoint(
    path: str,
    *,
    state_dim: int,
    action_count: int,
    hidden_sizes,
) -> CheckpointLoadResult:
    if not os.path.exists(path):
        return CheckpointLoadResult(CheckpointStatus.COLLECTION_ONLY, None, None)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    metadata = payload.get("metadata") or {}
    if _metadata_mismatch(metadata, state_dim=state_dim, action_count=action_count, hidden_sizes=hidden_sizes):
        raise CheckpointMetadataError(
            f"checkpoint {path!r} metadata {metadata} is incompatible with "
            f"state_dim={state_dim}, action_count={action_count}, hidden_sizes={list(hidden_sizes)}"
        )
    state_dict = {"model": payload["model"]}
    if "target_model" in payload:
        state_dict["target_model"] = payload["target_model"]
    if "optimizer" in payload:
        state_dict["optimizer"] = payload["optimizer"]
    return CheckpointLoadResult(CheckpointStatus.RESTORED, state_dict, metadata)
