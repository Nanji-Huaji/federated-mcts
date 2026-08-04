"""Strict paired Game24 experiment analysis and bootstrap.

Consumes versioned manifests and performance/results from the
game24-model-only-improvements plan.  Produces frozen ``PairedAnalysis``
with exact denominators and descriptive bootstrap CIs.
"""

from __future__ import annotations

import json
import math
import os
import random as _random
from dataclasses import dataclass
from typing import Any

# ═══════════════════════════════════════════════════════════════════
#  Frozen value objects
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class ExperimentResult:
    """Immutable container for one validated experiment result."""
    manifest: dict[str, Any]
    task_ids: list[int]
    solved_per_task: dict[int, bool]
    solved_count: int
    total_tasks: int
    accuracy: float
    total_calls: int
    total_tokens: int
    cost: float
    decision_policy: str
    decision_policy_fingerprint: str
    run_fingerprint: str
    complete: bool
    stop_reason: str | None
    task_records: list[dict]
    performance_raw: dict[str, Any]


@dataclass(frozen=True, slots=True)
class PairedAnalysis:
    """Frozen result of a paired comparison (descriptive only)."""
    baseline: ExperimentResult
    candidate: ExperimentResult
    paired_outcomes: list[tuple[int, int]]
    common_task_ids: list[int]
    baseline_accuracy: float
    candidate_accuracy: float
    delta: float
    ci_95_lower: float
    ci_95_upper: float
    bootstrap_seed: int
    validation_errors: list[str]


# ═══════════════════════════════════════════════════════════════════
#  Boundary parsers
# ═══════════════════════════════════════════════════════════════════

def load_result(
    manifest_path: str,
    performance_path: str,
    tasks_path: str,
) -> ExperimentResult:
    """Parse and validate one experiment result from on-disk files.

    Raises ``FileNotFoundError`` for missing files, ``ValueError`` for
    malformed/dangerous/incomplete records.
    """
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    if not os.path.exists(performance_path):
        raise FileNotFoundError(f"performance not found: {performance_path}")
    if not os.path.exists(tasks_path):
        raise FileNotFoundError(f"tasks not found: {tasks_path}")

    with open(manifest_path) as f:
        manifest: dict[str, Any] = json.load(f)
    with open(performance_path) as f:
        performance: dict[str, Any] = json.load(f)
    with open(tasks_path) as f:
        task_records: list[dict] = json.load(f)

    # ── Completeness ──────────────────────────────────────────────
    complete: bool = performance.get("complete", True)
    if type(complete) is not bool:
        raise ValueError(f"'complete' must be bool; got {type(complete).__name__}")
    if not complete:
        stop = performance.get("stop_reason", "unknown")
        raise ValueError(f"Result is incomplete (stop_reason={stop!r})")

    # ── Task-level solved/unsolved (compute first) ────────────────
    manifest_task_ids: list[int] = manifest.get("task_ids", [])
    solved_per_task: dict[int, bool] = {}
    for record in task_records:
        tid = record.get("idx", -1)
        infos = record.get("infos", [])
        solved = any(info.get("r", 0) == 1 for info in infos)
        solved_per_task[tid] = solved

    solved_count = sum(1 for v in solved_per_task.values() if v)
    total_tasks = len(task_records)
    accuracy = solved_count / total_tasks if total_tasks > 0 else 0.0

    # ── Numeric validation (from performance metadata) ────────────
    cost_raw = performance.get("estimated_cost_usd", 0.0)
    acc_raw = performance.get("acc", None)
    _reject_non_finite(cost_raw, "estimated_cost_usd")
    if acc_raw is not None:
        _reject_non_finite(acc_raw, "acc")
        if isinstance(acc_raw, bool):
            raise ValueError("'acc' must be a number, not bool")
        # Cross-check: performance acc must match computed accuracy
        if abs(float(acc_raw) - accuracy) > 0.001 and total_tasks > 0:
            raise ValueError(
                f"performance acc ({float(acc_raw):.4f}) != computed "
                f"accuracy ({accuracy:.4f}) from task data"
            )

    # ── Resource extraction ───────────────────────────────────────
    total_comp = performance.get("completion_tokens", 0)
    total_prompt = performance.get("prompt_tokens", 0)
    total_tokens_raw = _safe_int(total_comp) + _safe_int(total_prompt)

    total_calls_raw = 0
    for record in task_records:
        usage = record.get("usage_so_far", {})
        if isinstance(usage, dict):
            for _client, udata in usage.items():
                if isinstance(udata, dict):
                    total_calls_raw += _safe_int(udata.get("total_calls", 0))

    # ── Identity ──────────────────────────────────────────────────
    decision_policy = str(manifest.get("decision_policy", "local_validity"))
    decision_fp = str(performance.get("decision_policy_fingerprint", ""))
    run_fp = str(performance.get("run_fingerprint", ""))

    return ExperimentResult(
        manifest=manifest,
        task_ids=manifest_task_ids,
        solved_per_task=solved_per_task,
        solved_count=solved_count,
        total_tasks=total_tasks,
        accuracy=accuracy,
        total_calls=total_calls_raw,
        total_tokens=total_tokens_raw,
        cost=float(cost_raw),
        decision_policy=decision_policy,
        decision_policy_fingerprint=decision_fp,
        run_fingerprint=run_fp,
        complete=bool(complete),
        stop_reason=performance.get("stop_reason"),
        task_records=task_records,
        performance_raw=performance,
    )


# ═══════════════════════════════════════════════════════════════════
#  Paired analysis
# ═══════════════════════════════════════════════════════════════════

def run_paired_analysis(
    baseline: ExperimentResult,
    candidate: ExperimentResult,
    *,
    seed: int = 42,
) -> PairedAnalysis:
    """Strict paired comparison with deterministic bootstrap CI.

    Raises ``ValueError`` on task-ID mismatch or oracle mislabeling.
    """
    errors: list[str] = []
    base_ids = sorted(baseline.task_ids)
    cand_ids = sorted(candidate.task_ids)

    if base_ids != cand_ids:
        raise ValueError(
            f"Task ID mismatch: baseline {len(base_ids)} vs "
            f"candidate {len(cand_ids)} ids.  Paired comparison "
            f"requires identical ordered task sets."
        )
    if not base_ids:
        errors.append("Both experiments have zero tasks")

    # Oracle-assisted conditions cannot be paired as model-only
    for label, result in [("baseline", baseline), ("candidate", candidate)]:
        if result.decision_policy == "oracle_exact_pruned":
            raise ValueError(
                f"{label} decision_policy='oracle_exact_pruned' — "
                f"oracle-assisted results must not be presented as "
                f"model-only evidence."
            )

    # Build paired outcomes
    paired: list[tuple[int, int]] = []
    for tid in base_ids:
        b_s = 1 if baseline.solved_per_task.get(tid, False) else 0
        c_s = 1 if candidate.solved_per_task.get(tid, False) else 0
        paired.append((b_s, c_s))

    base_acc = baseline.solved_count / max(baseline.total_tasks, 1)
    cand_acc = candidate.solved_count / max(candidate.total_tasks, 1)
    delta = cand_acc - base_acc

    ci_lower, ci_upper = compute_paired_bootstrap_ci(paired, seed=seed)

    return PairedAnalysis(
        baseline=baseline,
        candidate=candidate,
        paired_outcomes=paired,
        common_task_ids=base_ids,
        baseline_accuracy=base_acc,
        candidate_accuracy=cand_acc,
        delta=delta,
        ci_95_lower=ci_lower,
        ci_95_upper=ci_upper,
        bootstrap_seed=seed,
        validation_errors=errors,
    )


def compute_paired_bootstrap_ci(
    paired: list[tuple[int, int]],
    *,
    seed: int = 42,
    n_resamples: int = 10_000,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Deterministic seeded paired bootstrap 95% CI on the delta.

    Descriptive interval — NOT confirmatory evidence of superiority.
    Returns ``(ci_lower, ci_upper)``.
    """
    if not paired:
        return (0.0, 0.0)

    n = len(paired)
    rng = _random.Random(seed)
    deltas: list[float] = []

    for _ in range(n_resamples):
        indices = [rng.randint(0, n - 1) for _ in range(n)]
        b_sum = sum(paired[i][0] for i in indices)
        c_sum = sum(paired[i][1] for i in indices)
        deltas.append((c_sum - b_sum) / n)

    deltas.sort()
    lo = int(alpha / 2 * n_resamples)
    hi = int((1 - alpha / 2) * n_resamples) - 1
    lo = max(0, min(lo, n_resamples - 1))
    hi = max(0, min(hi, n_resamples - 1))
    return (deltas[lo], deltas[hi])


# ═══════════════════════════════════════════════════════════════════
#  Internal helpers
# ═══════════════════════════════════════════════════════════════════

def _reject_non_finite(value: Any, field_name: str) -> None:
    """Raise ``ValueError`` if *value* is NaN, Inf, or -Inf."""
    if isinstance(value, (int, float)):
        fval = float(value)
        if math.isnan(fval) or math.isinf(fval):
            raise ValueError(f"{field_name} must be finite; got {value!r}")


def _safe_int(value: Any) -> int:
    """Convert *value* to int, returning 0 for unmappable inputs."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
