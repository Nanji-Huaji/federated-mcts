"""Thin runner helpers for journal, resume, task persistence, and file naming.

Extracted from ``scripts/merged_run.py`` so the runner stays ≤250 pure LOC.
"""
from __future__ import annotations

import json
import os

from federated_mcts.evaluation.game24_manifest import Game24Manifest
from federated_mcts.evaluation.game24_persistence import (
    AtomicTaskJournal,
    ResumeValidator,
)


def _read_model_config(config_path: str) -> list[dict]:
    with open(config_path) as f:
        return json.load(f)


def _resolve_backends_from_config(config_path: str, args) -> tuple[str, str]:
    """Return (localbackend, remotebackend) from config, mutating args."""
    cfg = _read_model_config(config_path)
    lb = next(d["model"] for d in cfg if d.get("client_name") == "local_client")
    rb = next(d["model"] for d in cfg if d.get("client_name") == "remote_client")
    args.localbackend = lb
    args.remotebackend = rb
    return lb, rb


def compute_file_prefix(args) -> str:
    """Return the output filename prefix WITHOUT creating directories.

    Pure computation — no os.makedirs, no I/O side effects.
    """
    if hasattr(args, "model_config"):
        _resolve_backends_from_config(args.model_config, args)

    if getattr(args, "naive_run", False):
        prefix = (
            f"./logs/{args.task}/{args.localbackend}/{args.remotebackend}/"
            f"{args.temperature}_naive_{args.prompt_sample}_sample_"
            f"{args.n_generate_sample}_start{args.task_start_index}"
            f"_end{args.task_end_index}_usingLLM"
        )
    else:
        policy_suffix = {
            "diverse": "_policy_diverse",
            "dqn": "_policy_dqn",
        }.get(getattr(args, "search_policy", "baseline"), "")
        prefix = (
            f"./logs/{args.task}/{args.solve_method}/{args.remotebackend}/"
            f"{args.temperature}_{args.method_generate}_n_generate_sample_"
            f"{args.n_generate_sample}_{args.method_evaluate}_n_evaluate_sample_"
            f"{args.n_evaluate_sample}_method_select_{args.method_select}"
            f"_n_select_sample_{args.n_select_sample}{policy_suffix}"
            f"_start{args.task_start_index}_end{args.task_end_index}"
            f"_smg_{args.slm_generate}_sme_{args.slm_eval}"
            f"_check_{args.check_format}_rule_{args.eval_rule}"
            f"_warm_{args.warm_start}_last_{args.last_lm}"
            f"_idx_{args.inference_idx}"
        )
    if getattr(args, "game24_exact_prune", False):
        prefix += "_exactprune_True"
    return prefix


def file_name_generater(args) -> str:
    """Generate the output file name prefix AND create parent directories.

    Maintained for backward compatibility — new callers should prefer
    ``compute_file_prefix`` + explicit directory creation.
    """
    prefix = compute_file_prefix(args)
    os.makedirs(os.path.dirname(prefix + ".json"), exist_ok=True)
    return prefix


class InjectedInterruption(BaseException):
    """Controlled test interruption that propagates past ``except Exception``."""


def setup_journal_and_resume(
    *,
    file_name: str,
    manifest: Game24Manifest,
    resume: bool,
    task_ids: list[int],
) -> tuple[AtomicTaskJournal, set[int]]:
    """Create journal, validate resume, return skip-set.

    Exits with ``sys.exit(1)`` on fatal rejections.
    """
    import sys

    journal = AtomicTaskJournal(file_name + "_journal.json")
    expected_fp = manifest.run_fingerprint if resume else ""
    rv = ResumeValidator(journal, expected_fp)
    can_cont, skip_str_ids, reason = rv.validate_resume(task_ids)
    if not can_cont:
        print(f"FATAL: Resume rejected — {reason}", file=sys.stderr)
        sys.exit(1)

    journal.write_meta(run_fingerprint=manifest.run_fingerprint)
    skip_ids = {int(s) for s in skip_str_ids}
    if skip_ids:
        print(f"Resume: skipping {len(skip_ids)} completed tasks: {sorted(skip_ids)}")
    return journal, skip_ids


def persist_task(
    journal: AtomicTaskJournal,
    task_id: int,
    ys: list[str],
    task_info: dict,
    task_metrics_json: str,
    elapsed_ms: float,
) -> None:
    """Atomically write a completed task record with full runner state."""
    journal.write_task(str(task_id), {
        "task_id": task_id,
        "ys": ys[-1] if ys else "",
        "task_info": task_info,
        "task_metrics_json": task_metrics_json,
        "elapsed_ms": elapsed_ms,
        "_completed": True,
    })


def restore_completed_tasks(
    journal: AtomicTaskJournal,
    completed_ids: set[int],
    *,
    logs: list[dict],
    per_task_metrics: list,
) -> tuple[int, int]:
    """Load completed task records from journal into runner state.

    Returns ``(cnt_avg_delta, cnt_any_delta)`` that the caller
    adds to its running counters.
    """
    records = journal.read_all()
    cnt_avg_delta = 0
    cnt_any_delta = 0
    for tid in sorted(completed_ids, key=int):
        rec = records.get(str(tid), {})
        if not rec:
            continue
        info = rec.get("task_info", {})
        if info:
            logs.append(info)
        accs: list[int] = []
        for r in info.get("infos", []):
            if isinstance(r, dict):
                accs.append(int(r.get("r", 0)))
        cnt_avg_delta += sum(accs)
        cnt_any_delta += int(any(accs))
        tm_json = rec.get("task_metrics_json", "")
        if tm_json:
            import json as _json
            try:
                per_task_metrics.append(_json.loads(tm_json))
            except Exception:
                pass
    return cnt_avg_delta, cnt_any_delta


def make_task_recorder(
    is_game24: bool, task_id: int, task_input: str,
    decision_policy: str, oracle_audit: bool,
) -> object | None:
    """Create a TaskSessionRecorder for Game24 tasks."""
    if not is_game24:
        return None
    from federated_mcts.evaluation.game24_task_recorder import (
        TaskSessionRecorder)
    return TaskSessionRecorder(
        task_id=str(task_id), task_input=task_input,
        decision_policy=decision_policy,
        enable_audit=oracle_audit)
