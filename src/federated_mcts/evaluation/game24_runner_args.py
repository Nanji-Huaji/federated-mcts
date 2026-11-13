"""Merged experiment runner CLI argument parsing.

Extracted from ``scripts/merged_run.py`` to keep each module ≤250 pure LOC.
Provides the ``parse_args`` function with all runner + B0-gate CLI flags.
"""

from __future__ import annotations

import argparse

from federated_mcts.evaluation.game24_runner_setup import add_runner_cli_args
from federated_mcts.evaluation.game24_b0_gate import add_b0_cli_args

__all__ = ["parse_args"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="fmcts-run", description="Run federated search experiments")
    parser.add_argument("--localbackend", type=str, default="bartowski/Phi-3-medium-128k-instruct-GGUF")
    parser.add_argument("--remotebackend", type=str, default="qwen2.5-32b-instruct")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--task", type=str, required=True, choices=["game24", "text", "crosswords", "gsm8k", "wikilogic", "hyblogic", "blocksworld"])
    parser.add_argument("--task_split", type=str, default=None, choices=["train", "val", "test"],
                        help="use predefined split from data/splits/{task}.json (overrides --task_start/--task_end)")
    parser.add_argument("--task_start_index", type=int, default=900)
    parser.add_argument("--task_end_index", type=int, default=1000)
    parser.add_argument("--naive_run", action="store_true")
    parser.add_argument("--prompt_sample", type=str, choices=["standard", "cot"])
    parser.add_argument("--method_generate", type=str, choices=["sample", "propose"])
    parser.add_argument("--method_evaluate", type=str, choices=["value", "vote"])
    parser.add_argument("--method_select", type=str, choices=["sample", "greedy"], default="greedy")
    parser.add_argument("--n_generate_sample", type=int, default=1)
    parser.add_argument("--n_evaluate_sample", type=int, default=1)
    parser.add_argument("--n_select_sample", type=int, default=1)
    parser.add_argument("--search_policy", choices=["baseline", "diverse", "dqn"], default="baseline")
    parser.add_argument("--diverse_joint_rank", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--diversity_weight", type=float, default=0.35)
    parser.add_argument("--beam_expand", type=int, default=2)
    parser.add_argument("--beam_uncertainty_margin", type=float, default=0.1)
    parser.add_argument("--beam_confidence_margin", type=float, default=0.8)
    parser.add_argument("--dqn_seed", type=int, default=0)
    parser.add_argument("--dqn_epsilon", type=float, default=1.0)
    parser.add_argument("--dqn_state_dim", type=int, default=16)
    parser.add_argument("--dqn_capacity", type=int, default=10000)
    parser.add_argument("--dqn_token_budget", type=float, default=5000.0)
    parser.add_argument("--dqn_latency_budget", type=float, default=60.0)
    parser.add_argument("--dqn_oracle_distance_reward", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dqn_oracle_distance_scale", type=float, default=0.25)
    parser.add_argument("--dqn_checkpoint", type=str, default=None)
    parser.add_argument("--dqn_jsonl", type=str, default=None)
    parser.add_argument("--solve_method", type=str, choices=["naive", "tot", "speculative_solve", "federated_solve"], default="tot")
    parser.add_argument("--slm_generate", action="store_true")
    parser.add_argument("--slm_eval", action="store_true")
    parser.add_argument("--check_format", action="store_true")
    parser.add_argument("--eval_rule", action="store_true")
    parser.add_argument("--warm_start", action="store_true")
    parser.add_argument("--model_config", type=str, default="configs/model_config.json")
    parser.add_argument("--inference_idx", type=int, default=0)
    parser.add_argument("--last_lm", action="store_true")
    parser.add_argument("--assign_strategy", type=str, default="round_robin",
                        choices=["round_robin", "difficulty", "bandit"])
    parser.add_argument("--filter", action="store_true")
    parser.add_argument("--game24_exact_prune", action="store_true", default=False)
    parser.add_argument("--max_cost_usd", type=float, default=None)
    # ── Todo 3: runner args from focused module ─────────────────────
    add_runner_cli_args(parser)
    # ── B0 measured-evidence gate args ──────────────────────────────
    add_b0_cli_args(parser)
    parser.add_argument(
        "--resume", action="store_true", default=False,
        help="resume from an existing journal if manifest fingerprint matches",
    )
    return parser.parse_args(argv)
