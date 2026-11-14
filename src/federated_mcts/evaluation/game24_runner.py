"""Game24 experiment runner integration module.

Provides a thin orchestration layer for ``scripts/merged_run.py``:
manifest construction, budget ledger setup, durable selected-test
authorization (enforced BEFORE any model client construction), and
result enrichment with preflight reconciliation.

Setup/CLI types live in ``game24_runner_setup.py``.
Preflight/dry-run/mock orchestration lives in ``game24_runner_orch.py``.

Follows Todo 3 V2 reviewer requirements + Wave 3 baseline safety:
- Auth consumed before FederatedSolver/client construction
- Diagnostics use distinct keys (``is_selected_test=False``)
- Second/deleted-output/resume/audit/schema attempts rejected
"""

from __future__ import annotations

import json
from decimal import Decimal
from federated_mcts.models.rates import DEFAULT_RATE_CARD
import os

from federated_mcts.evaluation.game24_manifest import (
    Game24Manifest,
    apply_seed,
)
from federated_mcts.evaluation.game24_runner_setup import (
    ExperimentSetup,
)
from federated_mcts.evaluation.game24_budget import BudgetLedger
from federated_mcts.evaluation.game24_authorization import (
    AuthRegistry,
    make_auth_key,
)
from federated_mcts.evaluation.game24_preflight import (
    FULL_VALIDATION_GATE,
    PreflightProjection,
    build_projection,
    reconcile_budget_after_tasks,
)
from federated_mcts.models import get_model_usage_summary
from federated_mcts.utils.cost_limit import cost_exceeded

# Re-export for callers
__all__ = [
    "ExperimentSetup",
    "build_manifest",
    "build_preflight",
    "setup_budget_ledger",
    "consume_authorization",
    "check_budget_caps",
    "enrich_result",
]


def build_manifest(setup: ExperimentSetup) -> Game24Manifest:
    """Create the immutable manifest from experiment setup."""
    model_config_raw = setup.model_configs

    seed_value = setup.seed if setup.seed is not None else 0
    return Game24Manifest(
        task=setup.task,
        task_ids=setup.task_ids,
        task_split_name=setup.task_split_name,
        decision_policy=setup.decision_policy,
        oracle_audit=setup.oracle_audit,
        seed=seed_value,
        temperature=setup.temperature,
        method_generate=setup.method_generate,
        method_evaluate=setup.method_evaluate,
        method_select=setup.method_select,
        n_generate_sample=setup.n_generate_sample,
        n_evaluate_sample=setup.n_evaluate_sample,
        n_select_sample=setup.n_select_sample,
        game24_exact_prune=setup.game24_exact_prune,
        assign_strategy=setup.assign_strategy,
        max_call_limit=setup.max_call_limit,
        max_token_limit=setup.max_token_limit,
        max_cost_usd=setup.max_cost_usd,
        max_raw_unit_limit=setup.max_raw_unit_limit,
        split_file_path=setup.split_file_path,
        model_config_raw=model_config_raw,
        schema_version=2 if setup.decision is not None else 1,
        decision_spec=setup.decision,
        runner_path="fmcts-run" if setup.decision is not None else "scripts/merged_run.py",
    )


def setup_budget_ledger(setup: ExperimentSetup) -> BudgetLedger:
    """Create the shared raw-unit budget ledger."""
    return BudgetLedger(
        ledger_path=setup.file_name + "_budget.jsonl",
        total_limit=setup.max_raw_unit_limit,
        max_calls=setup.max_call_limit,
        max_tokens=setup.max_token_limit,
    )


def build_preflight(setup: ExperimentSetup) -> PreflightProjection | None:
    """Build a preflight projection from the experiment setup.

    Returns ``None`` if the model config cannot be loaded.
    """
    model_config_raw = setup.model_configs
    if not model_config_raw:
        return None

    gate = FULL_VALIDATION_GATE
    return build_projection(
        n_tasks=200 if setup.task_split_name == "test" else len(setup.task_ids),
        model_config_raw=model_config_raw,
        n_generate_sample=setup.n_generate_sample,
        n_evaluate_sample=setup.n_evaluate_sample,
        n_select_sample=setup.n_select_sample,
        task_steps=3,
        gate_limit=gate,
    )


def consume_authorization(
    manifest: Game24Manifest,
    setup: ExperimentSetup,
) -> tuple[bool, str]:
    """Consume the selected-test authorization key BEFORE any client.

    MUST be called before ``FederatedSolver`` or any model client
    is constructed.  Returns ``(True, "")`` on success or
    ``(False, error_message)`` if the key was already consumed.

    Diagnostics (``is_selected_test=False``) skip consumption.
    """
    if setup.auth_registry_path is None:
        return True, ""  # No auth required
    if not setup.is_selected_test:
        return True, ""  # Diagnostic runs don't consume selected-test slot

    registry = AuthRegistry(setup.auth_registry_path)
    key = make_auth_key(
        ordered_test_ids_hash=manifest.ordered_task_ids_sha256,
        model_config_hash=manifest.model_config_sha256,
        decision_policy_fingerprint=manifest.decision_policy_fingerprint,
        review_round_identity=setup.review_round_identity,
    )
    ok_consume = registry.consume(
        key, manifest.run_fingerprint, is_selected_test=True,
    )
    if not ok_consume:
        return False, (
            "Authorization FAILED: key already consumed. "
            f"registry={setup.auth_registry_path} "
            f"review_round={setup.review_round_identity}"
        )
    return True, ""


def check_budget_caps(
    budget_ledger: BudgetLedger,
    max_cost_usd: float | None,
    *, usage: dict | None = None,
) -> tuple[bool, str | None]:
    """Check all budget caps before starting the next task.

    Returns ``(True, None)`` if all caps are satisfied, or
    ``(False, stop_reason)`` if a cap has been exceeded.
    """
    if usage is None:
        usage = get_model_usage_summary()
    total_calls = sum(s.get("total_calls", 0) for s in usage.values())
    total_tokens = sum(s.get("total_tokens", 0) for s in usage.values())

    ok_calls, reason = budget_ledger.check_and_mark_calls(used_calls=total_calls)
    if not ok_calls:
        return False, reason
    ok_tokens, reason = budget_ledger.check_and_mark_tokens(used_tokens=total_tokens)
    if not ok_tokens:
        return False, reason
    if cost_exceeded(usage, max_cost_usd):
        return False, "max_cost_usd"
    if budget_ledger.remaining <= 0:
        return False, "max_raw_unit_limit"
    return True, None


def enrich_result(
    res_json: dict,
    manifest: Game24Manifest,
    budget_ledger: BudgetLedger,
    stop_reason: str | None,
    *,
    projection: PreflightProjection | None = None,
) -> dict:
    """Add Todo 3 enrichment + preflight reconciliation to the performance result."""
    res_json["complete"] = (stop_reason is None)
    res_json["decision_policy_fingerprint"] = manifest.decision_policy_fingerprint
    res_json["run_fingerprint"] = manifest.run_fingerprint
    res_json["budget_remaining"] = budget_ledger.remaining
    res_json["budget_spent"] = budget_ledger.spent
    res_json["started_tasks"] = budget_ledger.started_tasks
    if projection is not None:
        reconcile = reconcile_budget_after_tasks(budget_ledger.spent, projection)
        res_json.update(reconcile)
    return res_json


def compute_task_raw_unit_cost(
    prev_summary: dict,
    current_summary: dict,
) -> Decimal:
    """Price incremental usage exactly with the shared versioned rate card."""
    total = Decimal("0")
    for model, cur in current_summary.items():
        prev = prev_summary.get(model, {})
        prompt = int(cur.get("prompt_tokens", 0)) - int(prev.get("prompt_tokens", 0))
        completion = int(cur.get("completion_tokens", 0)) - int(prev.get("completion_tokens", 0))
        if prompt < 0 or completion < 0:
            raise ValueError("Run usage counters decreased")
        if prompt or completion:
            total += DEFAULT_RATE_CARD.cost_raw_units(model, completion, prompt)
    return total
