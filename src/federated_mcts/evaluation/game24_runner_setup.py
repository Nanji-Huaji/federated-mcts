"""Experiment setup builder and CLI argument registration.

Extracted from ``game24_runner.py`` to keep each module ≤250 pure LOC.
Provides ``ExperimentSetup``, ``build_setup_from_args``, and
``add_runner_cli_args``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from federated_mcts.config import DecisionSpec

from federated_mcts.evaluation.game24_manifest import resolve_task_ids
from federated_mcts.evaluation.game24_safety import (
    validate_task_resolution,
    validate_pilot_split_identity,
)

# Re-export for backward compatibility
__all__ = [
    "ExperimentSetup",
    "build_setup_from_args",
    "add_runner_cli_args",
]


@dataclass(frozen=True, slots=True)
class ExperimentSetup:
    """Immutable experiment config extracted from CLI args."""

    task: str
    task_ids: list[int]
    task_split_name: str | None
    decision_policy: str
    oracle_audit: bool
    seed: int | None
    temperature: float
    method_generate: str
    method_evaluate: str
    method_select: str
    n_generate_sample: int
    n_evaluate_sample: int
    n_select_sample: int
    game24_exact_prune: bool
    assign_strategy: str
    max_call_limit: int | None
    max_token_limit: int | None
    max_cost_usd: float | None
    max_raw_unit_limit: float
    model_config_path: str
    split_file_path: str
    is_selected_test: bool
    review_round_identity: str
    auth_registry_path: str | None
    file_name: str
    pilot_ids: list[int] | None
    dry_run: bool
    mock_client: bool
    decision: DecisionSpec | None = None
    model_config_json: str | None = None

    @property
    def model_configs(self) -> list[dict]:
        """Fresh views of the resolved model config; callers cannot mutate setup."""
        if self.model_config_json is not None:
            return json.loads(self.model_config_json)
        # Legacy programmatic construction; the canonical runner always resolves once.
        if not os.path.exists(self.model_config_path):
            return []
        with open(self.model_config_path, encoding="utf-8") as stream:
            return json.load(stream)


def build_setup_from_args(args, split_file_path: str, file_name: str, *,
                          decision: DecisionSpec | None = None,
                          model_configs: list[dict] | None = None) -> ExperimentSetup:
    """Extract experiment config from parsed CLI args with safety gates."""
    import sys

    options = decision if decision is not None else args
    seed = getattr(options, "seed", None)
    task_split = getattr(args, "task_split", None)
    pilot_list_raw = getattr(args, "pilot_ids", None)
    pilot_list: list[int] | None = (
        pilot_list_raw if isinstance(pilot_list_raw, list) else None
    )
    dry_run = bool(getattr(args, "dry_run", False))
    mock = bool(getattr(args, "mock_client", False))
    dp = getattr(args, "game24_decision_policy", None)
    if dp is None:
        dp = "local_validity"

    # ── Safety gate: validate task resolution ────────────────────────
    ok, err = validate_task_resolution(
        pilot_ids=pilot_list,
        task_split=task_split,
        task_start_index=getattr(args, "task_start_index", 900),
        task_end_index=getattr(args, "task_end_index", 1000),
        selected_test_auth=getattr(args, "selected_test_auth", None),
        decision_policy=dp,
    )
    if not ok:
        print(f"SAFETY REJECTED: {err}", file=sys.stderr)
        sys.exit(1)

    # ── Pilot ID resolution ──────────────────────────────────────────
    if pilot_list is not None:
        task_indices = pilot_list
        if split_file_path and os.path.exists(split_file_path):
            ok2, err2 = validate_pilot_split_identity(
                split_file_path, pilot_list, args.task)
            if not ok2:
                print(f"SAFETY REJECTED: {err2}", file=sys.stderr)
                sys.exit(1)
    else:
        task_indices = resolve_task_ids(
            task=args.task,
            split_file_path=split_file_path,
            task_split=task_split,
            task_start_index=args.task_start_index,
            task_end_index=args.task_end_index,
        )

    is_sel = bool(getattr(args, "selected_test_auth", None))

    return ExperimentSetup(
        task=args.task,
        task_ids=task_indices,
        task_split_name=task_split,
        decision_policy=dp,
        oracle_audit=bool(getattr(args, "game24_oracle_audit", False)),
        seed=seed,
        temperature=getattr(options, "temperature", 0.9),
        method_generate=getattr(options, "method_generate", "propose"),
        method_evaluate=getattr(options, "method_evaluate", "vote"),
        method_select=getattr(options, "method_select", "greedy"),
        n_generate_sample=getattr(options, "n_generate_sample", 1),
        n_evaluate_sample=getattr(options, "n_evaluate_sample", 1),
        n_select_sample=getattr(options, "n_select_sample", 1),
        game24_exact_prune=getattr(options, "game24_exact_prune", False),
        assign_strategy=getattr(options, "assign_strategy", "round_robin"),
        max_call_limit=getattr(args, "max_total_calls", None),
        max_token_limit=getattr(args, "max_total_tokens", None),
        max_cost_usd=getattr(args, "max_cost_usd", None),
        max_raw_unit_limit=getattr(args, "max_raw_unit_limit", 19.99),
        model_config_path=getattr(args, "model_config", "configs/model_config.json"),
        split_file_path=split_file_path,
        is_selected_test=is_sel,
        review_round_identity=getattr(args, "review_round_identity", "review-1"),
        auth_registry_path=getattr(args, "selected_test_auth", None),
        file_name=file_name,
        pilot_ids=pilot_list,
        dry_run=dry_run,
        mock_client=mock,
        decision=decision,
        model_config_json=json.dumps(model_configs, sort_keys=True) if model_configs is not None else None,
    )


def add_runner_cli_args(parser) -> None:
    """Register Todo 3 + Wave 3 CLI arguments on an argparse parser."""
    parser.add_argument(
        "--seed", type=int, default=None,
        help="seed for random/numpy (client-side determinism)",
    )
    parser.add_argument(
        "--game24_decision_policy", type=str, default=None,
        choices=["local_validity", "heuristic_rules", "oracle_exact_pruned"],
        help="decision policy identity (selected-test if --selected_test_auth set)",
    )
    parser.add_argument(
        "--game24_oracle_audit", action="store_true", default=False,
        help="enable post-decision oracle audit (orthogonal to decision_policy)",
    )
    parser.add_argument(
        "--max_total_calls", type=int, default=None,
        help="stop before next task when total API calls exceed this",
    )
    parser.add_argument(
        "--max_total_tokens", type=int, default=None,
        help="stop before next task when total tokens exceed this",
    )
    parser.add_argument(
        "--max_raw_unit_limit", type=float, default=19.99,
        help="shared raw-unit budget for all planned experiments",
    )
    parser.add_argument(
        "--selected_test_auth", type=str, default=None,
        help="path to auth registry JSONL; enable selected-test protection",
    )
    parser.add_argument(
        "--review_round_identity", type=str, default="review-1",
        help="review round label for auth key (default: review-1)",
    )
    parser.add_argument(
        "--pilot_ids", type=int, nargs="*", default=None,
        help="explicit validation pilot task IDs; cannot combine with --task_split",
    )
    parser.add_argument(
        "--dry_run", action="store_true", default=False,
        help="resolve flags/IDs/config/manifest/projection but do not create "
             "output dirs/files, journals/ledgers/auth registries, or construct "
             "solver/client",
    )
    parser.add_argument(
        "--mock_client", action="store_true", default=False,
        help="use deterministic mock client (no HTTP/API calls); "
             "for offline E2E testing only",
    )
