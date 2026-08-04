"""Versioned preflight projection artifact for Game24 experiments.

Computes a worst-case raw-unit cost estimate before any model calls
are made, using rates from the model configuration and the runner's
configured solver limits, and gates full 200-ID validation at 0.75
raw units.

Also integrates actual raw-unit spending reconciliation with the
``BudgetLedger`` after task usage deltas.

Persistence: only for non-dry real/mocked runs.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

SCHEMA_VERSION: int = 1
"""Preflight projection schema version."""

# ── Model rate constants (raw-units per 1M tokens) ──
# These are the canonical rates from model_config_vllm.json projects.
# Raw-unit is the sum of prompt and completion cost in the project's
# dollar-denominated pricing (no currency conversion).
_MODEL_RATES: dict[str, dict[str, float]] = {
    "deepseek-v4-flash": {"prompt": 0.14, "completion": 0.28},
    "deepseek-v4-pro": {"prompt": 0.435, "completion": 0.87},
    "Qwen/Qwen2.5-32B-Instruct": {"prompt": 0.0, "completion": 0.0},
    "gpt-4o": {"prompt": 2.5, "completion": 10.0},
    "gpt-4o-2024-08-06": {"prompt": 2.5, "completion": 10.0},
    "gpt-4": {"prompt": 15.0, "completion": 30.0},
    "gpt-3.5-turbo": {"prompt": 0.75, "completion": 1.5},
}

PILOT_PROJECTION_GATE: float = 0.75
"""Maximum projected raw units for the 20-ID validation pilot."""

FULL_VALIDATION_GATE: float = 0.75
"""Maximum projected raw units for full 200-ID validation."""


@dataclass(frozen=True, slots=True)
class PreflightProjection:
    """Immutable preflight projection artifact.

    Records inputs, worst-case estimates, and the gate result."""
    schema_version: int
    n_tasks: int
    model_names: list[str]
    model_rates_raw: dict[str, dict[str, float]]
    solver_config: dict[str, object]
    worst_case_calls: int
    worst_case_prompt_tokens: int
    worst_case_completion_tokens: int
    worst_case_raw_unit_estimate: float
    gate_limit: float
    gate_passed: bool

    def to_json(self) -> str:
        """Serialize as canonical JSON (sorted keys)."""
        d: dict[str, object] = {
            "schema_version": self.schema_version,
            "n_tasks": self.n_tasks,
            "model_names": self.model_names,
            "model_rates_raw": self.model_rates_raw,
            "solver_config": self.solver_config,
            "worst_case_calls": self.worst_case_calls,
            "worst_case_prompt_tokens": self.worst_case_prompt_tokens,
            "worst_case_completion_tokens": self.worst_case_completion_tokens,
            "worst_case_raw_unit_estimate": self.worst_case_raw_unit_estimate,
            "gate_limit": self.gate_limit,
            "gate_passed": self.gate_passed,
        }
        return json.dumps(d, sort_keys=True, indent=2, ensure_ascii=False)


def get_model_rate(model_name: str) -> dict[str, float] | None:
    """Return prompt/completion rates for *model_name* in raw units/1M tokens.

    Returns ``None`` for unknown models (fail-closed: no invented fallback).
    """
    return _MODEL_RATES.get(model_name)


def build_projection(
    *,
    n_tasks: int,
    model_config_raw: list[dict[str, object]],
    n_generate_sample: int,
    n_evaluate_sample: int,
    n_select_sample: int,
    task_steps: int = 3,
    gate_limit: float | None = None,
) -> PreflightProjection:
    """Build a worst-case preflight projection.

    The projection uses the configured solver maximums to compute
    the absolute upper bound on calls, tokens, and raw-unit cost.

    **Charging rule**: only the ``remote_client`` route is charged.
    The local client (vLLM) is zero-cost.  All token/call totals are
    attributed exactly once to the remote model.

    **Fail-closed**: if the remote model has no known rate and is
    a paid model (not zero-cost), a ``ValueError`` is raised rather
    than using invented fallback rates.

    Args:
        n_tasks: Number of tasks in the run (e.g. 20 for pilot, 200 for full).
        model_config_raw: The model configuration entries.
        n_generate_sample: Number of generation samples per step.
        n_evaluate_sample: Number of evaluation samples per step.
        n_select_sample: Number of selections per step.
        task_steps: Number of steps (tree depth) per task.
        gate_limit: Raw-unit gate threshold (defaults to FULL_VALIDATION_GATE).

    Raises:
        ValueError: If the remote model is unknown and not zero-cost.
    """
    if gate_limit is None:
        gate_limit = FULL_VALIDATION_GATE

    # ── Identify remote_client model and its max_tokens ─────────────
    remote_model: str | None = None
    local_model: str = ""
    max_tokens_remote: int = 512
    max_tokens_local: int = 256
    all_model_names: list[str] = []

    for entry in model_config_raw:
        name = str(entry.get("model", ""))
        if name:
            all_model_names.append(name)
        mt_val = entry.get("max_tokens", 512)
        if isinstance(mt_val, (int, float)):
            mt = int(mt_val)
        else:
            mt = 512
        client = str(entry.get("client_name", ""))
        if client == "remote_client":
            remote_model = name
            max_tokens_remote = mt
        elif client == "local_client":
            local_model = name
            max_tokens_local = mt

    # ── Resolve remote model rate (fail-closed) ─────────────────────
    if remote_model is None:
        # No remote_client configured — zero cost
        model_rates: dict[str, dict[str, float]] = {}
        rate_prompt = 0.0
        rate_completion = 0.0
    else:
        known_rate = _MODEL_RATES.get(remote_model)
        if known_rate is None:
            # Unknown remote model — fail closed unless zero-cost
            is_zero_cost = remote_model in {"Qwen/Qwen2.5-32B-Instruct"}
            if not is_zero_cost:
                raise ValueError(
                    f"Unknown remote model {remote_model!r} has no rate configured. "
                    f"Known rates: {sorted(_MODEL_RATES.keys())}. "
                    "Add rate or use a known model."
                )
            known_rate = {"prompt": 0.0, "completion": 0.0}
        rate_prompt = known_rate["prompt"]
        rate_completion = known_rate["completion"]
        model_rates = {remote_model: known_rate}
        if local_model:
            model_rates[local_model] = {"prompt": 0.0, "completion": 0.0}

    # ── Solver config for the artifact ─────────────────────────────
    solver_config: dict[str, object] = {
        "n_generate_sample": n_generate_sample,
        "n_evaluate_sample": n_evaluate_sample,
        "n_select_sample": n_select_sample,
        "task_steps": task_steps,
    }

    # ── Worst-case call/token counts ────────────────────────────────
    gen_calls_per_task = task_steps * n_select_sample * n_generate_sample
    eval_calls_per_task = task_steps * n_select_sample * n_generate_sample * n_evaluate_sample
    calls_per_task = gen_calls_per_task + eval_calls_per_task
    worst_case_calls = calls_per_task * n_tasks

    # Prompt tokens: per-call estimate ~200 tokens (prompt + state)
    prompt_tokens_per_call = 200
    worst_case_prompt_tokens = worst_case_calls * prompt_tokens_per_call

    # Completion tokens: generation uses remote max_tokens,
    # evaluation also uses remote max_tokens (baseline route)
    worst_case_completion_tokens = (
        gen_calls_per_task * n_tasks * max_tokens_remote
        + eval_calls_per_task * n_tasks * max_tokens_remote
    )

    # ── Raw-unit estimate: charge remote_client model exactly once ──
    total_raw = (
        worst_case_prompt_tokens * rate_prompt / 1_000_000
        + worst_case_completion_tokens * rate_completion / 1_000_000
    )

    return PreflightProjection(
        schema_version=SCHEMA_VERSION,
        n_tasks=n_tasks,
        model_names=all_model_names,
        model_rates_raw=model_rates,
        solver_config=solver_config,
        worst_case_calls=worst_case_calls,
        worst_case_prompt_tokens=worst_case_prompt_tokens,
        worst_case_completion_tokens=worst_case_completion_tokens,
        worst_case_raw_unit_estimate=round(total_raw, 6),
        gate_limit=gate_limit,
        gate_passed=total_raw <= gate_limit,
    )


def persist_projection(projection: PreflightProjection, path: str) -> None:
    """Write the projection artifact to *path* (only for non-dry runs)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(projection.to_json())


def reconcile_budget_after_tasks(
    budget_ledger_spent: float,
    projection: PreflightProjection,
) -> dict[str, object]:
    """Produce a reconciliation record comparing actual vs. projected spend.

    Args:
        budget_ledger_spent: Actual raw units spent (from BudgetLedger).
        projection: The preflight projection.

    Returns:
        A dictionary with reconciliation fields ready for result enrichment.
    """
    return {
        "preflight_projection_raw_units": projection.worst_case_raw_unit_estimate,
        "actual_spent_raw_units": budget_ledger_spent,
        "projection_gate_passed": projection.gate_passed,
        "projection_gate_limit": projection.gate_limit,
        "preflight_schema_version": projection.schema_version,
    }
