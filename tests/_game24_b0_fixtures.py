"""Shared B0 gate test fixtures — test-local typed helpers.

Provides synthetic manifest, performance, journal, budget, args builders.
No side effects; all paths are caller-managed via tmpdir.
"""

from __future__ import annotations

import json
import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, "src")

from federated_mcts.evaluation.game24_b0_evidence import (
    CANONICAL_PILOT_IDS,
)
from federated_mcts.evaluation.game24_b0_manifest_loader import (
    derive_fingerprints_from_manifest_dict,
)


def _mf(**kw) -> dict:
    """Build B0 manifest dict — no derived fingerprint fields."""
    d: dict = {
        "task": "game24",
        "task_ids": list(CANONICAL_PILOT_IDS),
        "task_split_name": "val",
        "decision_policy": "local_validity",
        "oracle_audit": True,
        "seed": 17,
        "temperature": 0.9,
        "method_generate": "propose",
        "method_evaluate": "vote",
        "method_select": "greedy",
        "n_generate_sample": 2,
        "n_evaluate_sample": 3,
        "n_select_sample": 2,
        "game24_exact_prune": False,
        # ── Budget limits ─────────────────────────────────
        "max_call_limit": None,
        "max_token_limit": None,
        "max_cost_usd": None,
        "max_raw_unit_limit": 19.99,
        # ── Solver extras ─────────────────────────────────
        "assign_strategy": "round_robin",
        "runner_path": "scripts/merged_run.py",
        "effective_cli": "",
        # ── File / model / git hashes (placeholder) ────────
        "split_file_sha256": "a" * 64,
        "ordered_task_ids_sha256": "b" * 64,
        "model_config_sha256": "c" * 64,
        "model_names": ["qwen3.7-plus"],
        # ── Git metadata ──────────────────────────────────
        "git_commit": "0000000",
        "worktree_dirty": False,
    }
    d.update(kw)
    return d


def _derive_fps(mf_dict: dict) -> tuple[str, str]:
    return derive_fingerprints_from_manifest_dict(mf_dict)


def _pf(run_fp: str, dp_fp: str, **kw) -> dict:
    d: dict = {
        "acc": 0.0, "avg_sum": 0.0,
        "llm": "qwen3.7-plus",
        "qwen3.7-plus": {
            "total_calls": 165, "prompt_tokens": 85992,
            "completion_tokens": 84054, "total_tokens": 170046,
            "cost": 0.1688832,
        },
        "estimated_cost_usd": 0.1688832,
        "budget_spent": 0.1688832,
        "budget_remaining": 19.8235328,
        "complete": True, "run_fingerprint": run_fp,
        "decision_policy_fingerprint": dp_fp,
    }
    d.update(kw)
    return d


def _jf(run_fp: str, task_ids=None) -> dict:
    ids = task_ids if task_ids is not None else list(CANONICAL_PILOT_IDS)
    j: dict = {"_meta": {"run_fingerprint": run_fp}}
    for tid in ids:
        j[str(tid)] = {"task_id": tid, "ys": [f"r{tid}"], "_completed": True}
    return j


def _bf(spent=0.1688832, limit=19.99) -> list[dict]:
    import hashlib

    def sh(obj):
        d = json.dumps(obj, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(d).hexdigest()

    init = {"kind": "init", "timestamp_utc": "2025-01-01T00:00:00Z",
            "prev_hash": "", "total_limit": limit}
    init["entry_hash"] = sh(init)
    se = {"kind": "spend", "timestamp_utc": "2025-01-01T00:01:00Z",
          "prev_hash": init["entry_hash"], "amount": spent,
          "remaining": limit - spent}
    se["entry_hash"] = sh(se)
    return [init, se]


def _make_all(tmp, mf_dict=None, pf_dict=None, jf_dict=None, bf_list=None) -> dict[str, str]:
    ps: dict[str, str] = {}
    if mf_dict is not None:
        p = os.path.join(tmp, "manifest.json")
        with open(p, "w") as f:
            json.dump(mf_dict, f)
        ps["manifest"] = p
    if pf_dict is not None:
        p = os.path.join(tmp, "performance.json")
        with open(p, "w") as f:
            json.dump(pf_dict, f)
        ps["performance"] = p
    if jf_dict is not None:
        p = os.path.join(tmp, "journal.json")
        with open(p, "w") as f:
            json.dump(jf_dict, f)
        ps["journal"] = p
    if bf_list is not None:
        p = os.path.join(tmp, "budget.jsonl")
        with open(p, "w") as f:
            for e in bf_list:
                f.write(json.dumps(e, sort_keys=True,
                                   separators=(",", ":")) + "\n")
        ps["budget"] = p
    return ps


def _args(**kw) -> MagicMock:
    args = MagicMock()
    defaults = {
        "task_split": None, "pilot_ids": None,
        "b0_manifest": None, "b0_performance": None,
        "b0_journal": None, "b0_budget": None,
        "b0_dry_run": False,
    }
    defaults.update(kw)
    for k, v in defaults.items():
        setattr(args, k, v)
    return args
