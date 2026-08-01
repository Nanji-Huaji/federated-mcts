#!/usr/bin/env python3
"""Offline Double-DQN trainer for the Budget-Aware DQN controller.

Consumes a JSONL transition file (as produced by DQN collection mode under
search_policy=dqn --dqn_jsonl), trains the Q network offline, and saves a
validated checkpoint that the controller loads via --dqn_checkpoint.

No network, GPU, or API access is used.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import torch

from federated_mcts.core.dqn.checkpoint import (
    CheckpointStatus,
    load_checkpoint,
    save_checkpoint,
)
from federated_mcts.core.dqn.recorder import JSONLTransitionRecorder
from federated_mcts.core.dqn.trainer import build_trainer


def _fill_buffer(transitions, buffer) -> None:
    for transition in transitions:
        buffer.push(
            state=torch.tensor(transition["state"], dtype=torch.float32),
            action=int(transition["action"]),
            reward=float(transition["reward"]),
            next_state=(
                None
                if transition.get("next_state") is None
                else torch.tensor(transition["next_state"], dtype=torch.float32)
            ),
            done=bool(transition["done"]),
        )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Offline Double-DQN trainer.")
    parser.add_argument("--jsonl", required=True, help="JSONL file of collected transitions")
    parser.add_argument("--checkpoint", required=True, help="output checkpoint path")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--state_dim", type=int, default=12)
    parser.add_argument("--action_count", type=int, default=8)
    parser.add_argument("--hidden", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--capacity", type=int, default=10000)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    args = parser.parse_args(argv)

    transitions = JSONLTransitionRecorder(args.jsonl).read_all()
    if not transitions:
        parser.error(f"no transitions found in {args.jsonl!r}")

    trainer = build_trainer(
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
    _fill_buffer(transitions, trainer.replay_buffer)

    steps = max(1, args.epochs) * max(1, len(transitions) // max(1, args.batch_size))
    total_loss = 0.0
    steps_run = 0
    for _ in range(steps):
        if len(trainer.replay_buffer) < args.batch_size:
            break
        total_loss += trainer.train_step()
        steps_run += 1
    trainer.sync_target()
    print(
        f"Trained {steps_run} steps over {len(transitions)} transitions; "
        f"mean loss {total_loss / max(1, steps_run):.6f}"
    )

    metadata = {
        "state_dim": args.state_dim,
        "action_count": args.action_count,
        "hidden_sizes": list(args.hidden),
        "format_version": 1,
    }
    save_checkpoint(
        args.checkpoint,
        model_state=trainer.q_network.state_dict(),
        target_state=trainer.target_network.state_dict(),
        metadata=metadata,
    )

    result = load_checkpoint(
        args.checkpoint,
        state_dim=args.state_dim,
        action_count=args.action_count,
        hidden_sizes=tuple(args.hidden),
    )
    if result.status is not CheckpointStatus.RESTORED:
        parser.error(f"checkpoint validation failed for {args.checkpoint!r}")
    print(f"Saved validated checkpoint to {args.checkpoint}")


if __name__ == "__main__":
    main()
