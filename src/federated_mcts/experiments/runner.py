"""Plan and execute an experiment through one public, offline-testable seam.

Preparation reads inputs and validates gates. Execution alone owns authorization,
client construction, run-scoped accounting, durable commits and report export.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal
import json
import os
from types import SimpleNamespace

from federated_mcts.config import DecisionSpec, load_model_config
from federated_mcts.tasks import get_task
from federated_mcts.models import get_model_usage_summary
from federated_mcts.models.usage import usage_scope
from federated_mcts.federation import FederatedSolver
from federated_mcts.federation.outcome import finish_task
from federated_mcts.utils.cost_limit import cost_exceeded, total_cost_usd
from federated_mcts.evaluation.blocksworld_metrics import TaskOracleMetrics, merge_oracle_metrics, task_oracle_metrics
from federated_mcts.evaluation.game24_runner_setup import ExperimentSetup, build_setup_from_args
from federated_mcts.evaluation.game24_runner import (
    build_manifest, build_preflight, setup_budget_ledger, consume_authorization,
    check_budget_caps, enrich_result, compute_task_raw_unit_cost,
)
from federated_mcts.evaluation.game24_runner_orch import run_preflight_and_dry_run, setup_mock_solver
from federated_mcts.evaluation.game24_b0_gate import enforce_b0_gate
from federated_mcts.evaluation.game24_runner_persistence import (
    compute_file_prefix, setup_journal_and_resume, restore_completed_tasks, make_task_recorder,
)
from federated_mcts.evaluation.game24_persistence import TaskCommitter, BudgetExhaustedError
from federated_mcts.evaluation.game24_preflight import PreflightProjection, persist_projection
from federated_mcts.evaluation.game24_manifest import Game24Manifest, apply_seed


@dataclass(frozen=True, slots=True)
class RunPlan:
    setup: ExperimentSetup
    decision: DecisionSpec
    manifest: Game24Manifest
    projection: PreflightProjection | None
    resume: bool
    local_model: str
    remote_model: str


def prepare(args, solve_method: str | None = None) -> RunPlan | None:
    """Resolve inputs once and validate, without creating clients or artifacts."""
    options = SimpleNamespace(**vars(args))
    if solve_method is not None:
        options.solve_method = solve_method
    decision = DecisionSpec.from_args(options)
    model_configs = load_model_config(options.model_config, temperature=decision.temperature)
    from federated_mcts.models.rates import DEFAULT_RATE_CARD
    # Never discover an unknown billing rate only after making a paid call.
    for entry in model_configs:
        DEFAULT_RATE_CARD.rate_for(entry["model"])
    names = {entry["client_name"]: entry["model"] for entry in model_configs}
    split = getattr(options, "task_split", None)
    pilot = getattr(options, "pilot_ids", None)
    split_path = f"data/splits/{options.task}.json" if options.task == "game24" or split or pilot else ""
    setup = build_setup_from_args(options, split_path, "", decision=decision, model_configs=model_configs)

    # Naming consumes resolved values, not a path it would reopen and mutate.
    prefix_options = SimpleNamespace(**{key: value for key, value in vars(options).items()
                                       if key != "model_config"})
    for name, value in decision.identity().items():
        setattr(prefix_options, name, value)
    prefix_options.localbackend = names.get("local_client", "none")
    prefix_options.remotebackend = names.get("remote_client", next(iter(names.values())))
    setup = replace(setup, file_name=compute_file_prefix(prefix_options, task_count=len(setup.task_ids)))
    manifest = build_manifest(setup)
    telemetry_supported = decision.solve_method == "tot" and decision.search_policy == "baseline"
    if setup.task == "game24" and not telemetry_supported:
        if setup.oracle_audit or setup.is_selected_test or len(setup.task_ids) > 20:
            raise ValueError("Game24 audited/full-validation experiments require baseline tot; other modes do not provide complete decision telemetry")
    projection = build_preflight(setup) if setup.task == "game24" and telemetry_supported else None
    has_b0 = any(getattr(options, f"b0_{name}", None) for name in ("manifest", "performance", "journal", "budget"))
    # Validate B0 identity even for --dry_run: a dry-run must not bypass a gate.
    if not run_preflight_and_dry_run(setup, manifest, projection, skip_full_val_gate=has_b0):
        if has_b0:
            enforce_b0_gate(options, manifest=manifest)
        return None
    if not enforce_b0_gate(options, manifest=manifest):
        return None
    return RunPlan(setup, decision, manifest, projection, bool(getattr(options, "resume", False)),
                   prefix_options.localbackend, prefix_options.remotebackend)


def _usage_delta(previous: dict, current: dict) -> dict[str, int]:
    delta = {}
    for model, stats in current.items():
        for name in ("completion_tokens", "prompt_tokens", "total_tokens", "total_calls"):
            amount = int(stats.get(name, 0)) - int(previous.get(model, {}).get(name, 0))
            if amount < 0:
                raise ValueError("Run usage counters decreased")
            if amount:
                delta[f"{model}.{name}"] = amount
    return delta


def _restored_usage(journal, skip_ids: set[int]) -> dict:
    """Recover counters from durable per-task facts, not a global last snapshot."""
    from federated_mcts.models.rates import DEFAULT_RATE_CARD
    totals: dict = {}
    for tid, record in journal.read_all().items():
        if tid.startswith("_") or int(tid) not in skip_ids:
            continue
        for key, value in record.get("usage_delta", {}).items():
            model, _, field = key.rpartition(".")
            stats = totals.setdefault(model, {})
            stats[field] = stats.get(field, 0) + int(value)
    for model, stats in totals.items():
        stats["cost"] = float(DEFAULT_RATE_CARD.cost_raw_units(
            model, stats.get("completion_tokens", 0), stats.get("prompt_tokens", 0)))
    return totals


def _combined_usage(restored: dict, current: dict) -> dict:
    result = {model: dict(stats) for model, stats in restored.items()}
    for model, stats in current.items():
        merged = result.setdefault(model, {})
        for key, value in stats.items():
            merged[key] = merged.get(key, 0) + value
    return result


def execute(plan: RunPlan) -> dict:
    """Execute a validated plan; repeated calls never share usage counters."""
    with usage_scope():
        return _execute(plan)


def _execute(plan: RunPlan) -> dict:
    setup, decision, manifest = plan.setup, plan.decision, plan.manifest
    decision.validate_checkpoint()
    file_name = setup.file_name
    journal, _ = setup_journal_and_resume(file_name=file_name, manifest=manifest,
                                         resume=plan.resume, task_ids=setup.task_ids)
    ledger = setup_budget_ledger(setup)
    committer = TaskCommitter(journal, ledger, run_id=manifest.run_fingerprint)
    recovery = committer.recover()
    if not recovery.fully_reconciled:
        raise ValueError("Run has unmetered legacy records or unmatched charges; refusing an unsafe resume")
    skip_ids = {int(tid) for tid in recovery.settled_ids}
    if decision.seed is not None:
        apply_seed(decision.seed)
    ok, error = consume_authorization(manifest, setup)
    if not ok:
        raise ValueError(error)
    os.makedirs(os.path.dirname(file_name) or ".", exist_ok=True)
    with open(file_name + "_manifest.json", "w", encoding="utf-8") as stream:
        stream.write(manifest.to_json())
    if plan.projection is not None:
        persist_projection(plan.projection, file_name + "_preflight.json")
    solver = setup_mock_solver(decision, setup, solver_cls=FederatedSolver)

    logs: list[dict] = []
    task_metrics: list = []
    oracle_metrics: list[TaskOracleMetrics] = []
    cnt_avg = cnt_any = 0
    if skip_ids:
        cnt_avg, cnt_any = restore_completed_tasks(journal, skip_ids, logs=logs, per_task_metrics=task_metrics)
        oracle_metrics = [entry["blocksworld_oracle"] for entry in logs if "blocksworld_oracle" in entry]
    restored = _restored_usage(journal, skip_ids)
    previous = get_model_usage_summary()
    usage = _combined_usage(restored, previous)
    stop_reason = None
    telemetry_supported = setup.task == "game24" and decision.solve_method == "tot" and decision.search_policy == "baseline"
    pending_cost = Decimal("0")
    for index in setup.task_ids:
        if index in skip_ids:
            continue
        can_continue, stop_reason = check_budget_caps(ledger, setup.max_cost_usd, usage=usage)
        if not can_continue:
            break
        # A crash before a metered outcome exists cannot safely be replayed:
        # external API costs may already have occurred. Keep a durable marker
        # so recovery refuses to invent zero usage or silently pay twice.
        journal.put_task(str(index), {"task_id": index, "_completed": False,
                                      "execution_state": "running"})
        ledger.mark_task_started()
        task = get_task(setup.task)
        recorder = make_task_recorder(telemetry_supported, index, task.get_input(index),
                                     setup.decision_policy, setup.oracle_audit)
        solve_client = "local_client" if decision.search_policy == "dqn" and "local_client" in solver.gpts else "remote_client"
        if solve_client not in solver.gpts:
            solve_client = next(iter(solver.gpts))
        outcome = finish_task(solver, task, index, method=decision.solve_method,
                              assign_strategy=decision.assign_strategy, solve_client=solve_client,
                              task_recorder=recorder)
        ys, verdicts, info = list(outcome.candidates), list(outcome.verdicts), outcome.trace
        oracle = task_oracle_metrics(task, index, ys)
        if oracle is not None:
            info["blocksworld_oracle"] = oracle
            oracle_metrics.append(oracle)
        current = get_model_usage_summary()
        delta = _usage_delta(previous, current)
        cost = compute_task_raw_unit_cost(previous, current)
        previous = current
        usage = _combined_usage(restored, current)
        info.update(idx=index, ys=ys, infos=verdicts, usage_delta=delta,
                    usage_so_far=usage, time_consumption=solver.latency_dict)
        metrics = recorder.to_task_metrics() if recorder is not None else None
        # A failure here is fatal. Never warn-and-continue past an uncommitted task.
        try:
            committer.commit(index, ys=ys, task_info=info, usage=delta, cost=cost,
                             task_metrics_json=json.dumps(metrics) if metrics is not None else "",
                             elapsed_ms=recorder.total_elapsed_ms if recorder is not None else 0.0)
        except BudgetExhaustedError:
            stop_reason = "max_raw_unit_limit"
            pending_cost = Decimal(str(cost))
        if metrics is not None:
            task_metrics.append(metrics)
        logs.append(info)
        rewards = [verdict["r"] for verdict in verdicts]
        cnt_avg += sum(rewards)
        cnt_any += any(rewards)
        if stop_reason is not None:
            break
        if cost_exceeded(usage, setup.max_cost_usd):
            stop_reason = "max_cost_usd"
            break

    # Check the final task too, not only the beginning of the next iteration.
    # An exactly exhausted raw-unit budget is allowed when every task finished.
    if stop_reason is None:
        ok_calls, call_reason = ledger.check_and_mark_calls(
            sum(stats.get("total_calls", 0) for stats in usage.values()))
        ok_tokens, token_reason = ledger.check_and_mark_tokens(
            sum(stats.get("total_tokens", 0) for stats in usage.values()))
        if not ok_calls or not ok_tokens:
            stop_reason = call_reason or token_reason
    order = {index: position for position, index in enumerate(setup.task_ids)}
    logs.sort(key=lambda item: order[item["idx"]])
    # Export reports from committed/recoverable facts; their absence is recoverable.
    with open(file_name + ".json", "w", encoding="utf-8") as stream:
        json.dump(logs, stream, indent=4)
    result = {"avg_sum": cnt_avg / len(logs) if logs else 0.0,
              "acc": cnt_any / len(logs) if logs else 0.0,
              "sm": plan.local_model, "llm": plan.remote_model, **usage,
              "estimated_cost_usd": total_cost_usd(usage),
              "telemetry_complete": telemetry_supported,
              "pending_cost_raw": str(pending_cost)}
    if stop_reason is not None:
        result["stop_reason"] = stop_reason
    enrich_result(result, manifest, ledger, stop_reason, projection=plan.projection)
    merge_oracle_metrics(oracle_metrics, result)
    if task_metrics:
        from federated_mcts.evaluation.game24_metrics import summarize_experiment_metrics
        result["experiment_metrics"] = summarize_experiment_metrics(task_metrics)
    result["task_count"] = len(logs)
    with open(file_name + "_performance.json", "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=4)
    return result


def run(args, solve_method: str | None = None):
    plan = prepare(args, solve_method)
    return execute(plan) if plan is not None else None
