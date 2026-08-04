"""Shared synthetic fixture helpers for Game24 analysis tests.

Produces minimal yet valid manifest, performance, and per-task JSON strings
that match the output format of ``scripts/merged_run.py``.
"""

from __future__ import annotations

import json


def make_manifest_json(**overrides) -> str:
    """Return a minimal Game24 manifest JSON string."""
    defaults: dict = {
        "task": "game24",
        "task_ids": [3, 9, 16, 19],
        "task_split_name": "val",
        "split_file_sha256": "a" * 64,
        "ordered_task_ids_sha256": "b" * 64,
        "model_config_sha256": "c" * 64,
        "model_names": ["TestModel"],
        "decision_policy": "local_validity",
        "oracle_audit": False,
        "seed": 17,
        "temperature": 0.9,
        "method_generate": "propose",
        "method_evaluate": "vote",
        "method_select": "greedy",
        "n_generate_sample": 2,
        "n_evaluate_sample": 3,
        "n_select_sample": 2,
        "game24_exact_prune": False,
        "assign_strategy": "round_robin",
        "runner_path": "scripts/merged_run.py",
        "effective_cli": "",
        "git_commit": "abc1234",
        "worktree_dirty": False,
        "remote_nondeterminism_notice": True,
        "schema_version": 1,
        "max_call_limit": None,
        "max_token_limit": None,
        "max_cost_usd": None,
        "max_raw_unit_limit": 19.99,
        "decision_policy_fingerprint": "d" * 64,
        "run_fingerprint": "e" * 64,
    }
    defaults.update(overrides)
    return json.dumps(defaults, sort_keys=True, separators=(",", ":"))


def make_performance_json(
    *,
    acc: float = 0.5,
    solved_count: int = 2,
    total_tokens: int = 5000,
    cost: float = 0.05,
    complete: bool = True,
    stop_reason: str | None = None,
    decision_fp: str = "d" * 64,
    run_fp: str = "e" * 64,
    budget_remaining: float = 19.0,
    budget_spent: float = 0.99,
    started_tasks: int = 4,
    **extra,
) -> str:
    """Return a minimal Game24 performance JSON string."""
    data: dict = {
        "avg_sum": acc,
        "acc": acc,
        "sm": "local-model",
        "llm": "remote-model",
        "completion_tokens": total_tokens // 2,
        "prompt_tokens": total_tokens // 2,
        "total_tokens": total_tokens,
        "estimated_cost_usd": cost,
        "stop_reason": stop_reason,
        "complete": complete,
        "decision_policy_fingerprint": decision_fp,
        "run_fingerprint": run_fp,
        "budget_remaining": budget_remaining,
        "budget_spent": budget_spent,
        "started_tasks": started_tasks,
    }
    data.update(extra)
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def make_tasks_json(
    *,
    task_ids: list[int] | None = None,
    solved_ids: set[int] | None = None,
    parse_failure_ids: set[int] | None = None,
) -> str:
    """Return a synthetic per-task JSON array matching merged_run.py output.

    Solved means ``info["r"] == 1``.  Parse failure means empty ``ys``
    and is treated as unsolved.
    """
    if task_ids is None:
        task_ids = [3, 9, 16, 19]
    if solved_ids is None:
        solved_ids = set()
    if parse_failure_ids is None:
        parse_failure_ids = set()

    records: list[dict] = []
    for tid in task_ids:
        is_solved = tid in solved_ids
        is_parse_fail = tid in parse_failure_ids

        if is_parse_fail:
            ys_val: list[str] = [""]
            infos_val: list[dict] = [{"r": 0}]
        elif is_solved:
            ys_val = ["8 * (4 - 1) = 24"]
            infos_val = [{"r": 1}]
        else:
            ys_val = ["1 + 2 + 3 = 6 (left: 6)"]
            infos_val = [{"r": 0}]

        record: dict = {
            "idx": tid,
            "ys": ys_val,
            "infos": infos_val,
            "usage_so_far": {
                "remote_client": {
                    "completion_tokens": 100,
                    "prompt_tokens": 200,
                    "total_tokens": 300,
                    "total_calls": 5,
                },
            },
            "time_consumption": {"generate": 1.5, "evaluate": 0.5, "total": 2.0},
        }
        records.append(record)

    return json.dumps(records, sort_keys=True, separators=(",", ":"))
