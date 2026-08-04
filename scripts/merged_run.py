import os
import json
import sys

from federated_mcts.env_loader import load_dotenv
load_dotenv()


from federated_mcts.tasks import get_task

from federated_mcts.models import get_model_usage_summary
from federated_mcts.utils.cost_limit import cost_exceeded, total_cost_usd

from federated_mcts.federation import FederatedSolver

from federated_mcts.evaluation.blocksworld_metrics import merge_oracle_metrics, task_oracle_metrics

# ── Todo 3: experiment runner integration (manifest, budget, auth) ─
from federated_mcts.evaluation.game24_runner_setup import (
    build_setup_from_args,
)
from federated_mcts.evaluation.game24_runner import (
    build_manifest,
    build_preflight,
    setup_budget_ledger,
    consume_authorization,
    check_budget_caps,
    enrich_result,
    compute_task_raw_unit_cost,
)
from federated_mcts.evaluation.game24_runner_orch import (
    run_preflight_and_dry_run,
    setup_mock_solver,
)
from federated_mcts.evaluation.game24_b0_gate import (
    enforce_b0_gate,
)
from federated_mcts.evaluation.game24_runner_persistence import (
    compute_file_prefix,
    setup_journal_and_resume,
    persist_task,
    restore_completed_tasks,
    make_task_recorder,
)
from federated_mcts.evaluation.game24_preflight import persist_projection
from federated_mcts.evaluation.game24_runner_args import parse_args


def file_name_generater(args):
    if hasattr(args, "model_config"):
        with open(args.model_config) as f:
            model_config = json.load(f)
        args.localbackend = next(
            entry["model"] for entry in model_config
            if entry.get("client_name") == "local_client"
        )
        args.remotebackend = next(
            entry["model"] for entry in model_config
            if entry.get("client_name") == "remote_client"
        )
    if getattr(args, "naive_run", False):
        file_name = (
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
        file_name = (
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
        file_name += "_exactprune_True"
    return file_name


def run(args, solve_method):
    # ── Compute file name (NO directory creation) ────────────────────
    file_name = compute_file_prefix(args)

    # ── Todo 3: setup manifest, budget, seed ───────────────────────
    task_split = getattr(args, "task_split", None)
    pilot_ids = getattr(args, "pilot_ids", None)
    split_path = f"data/splits/{args.task}.json" if (task_split or pilot_ids) else ""
    setup = build_setup_from_args(args, split_path, file_name)
    manifest = build_manifest(setup)

    # ── Preflight gate check + dry run ──────────────────────────────
    projection = build_preflight(setup)

    # When B0 measured evidence is requested, defer the static full-validation
    # gate to the B0 measured-projection gate below.  The static projection is
    # still computed and printed for diagnostics; only the full-200 gate is skipped.
    has_b0_evidence = bool(
        getattr(args, "b0_manifest", None)
        or getattr(args, "b0_performance", None)
        or getattr(args, "b0_journal", None)
        or getattr(args, "b0_budget", None)
    )
    if not run_preflight_and_dry_run(
        setup, manifest, projection, skip_full_val_gate=has_b0_evidence,
    ):
        return  # dry run — no side effects performed

    # ── B0 measured-evidence projection gate (before side effects) ───
    if not enforce_b0_gate(args, manifest=manifest):
        return  # b0_dry_run — no side effects performed

    # ── Todo 7: persistence and resume ───────────────────────────────
    resume = bool(getattr(args, "resume", False))
    journal, skip_ids = setup_journal_and_resume(
        file_name=file_name, manifest=manifest,
        resume=resume, task_ids=setup.task_ids,
    )
    is_game24 = setup.task == "game24"

    # ── Create output directories (only after resume validation) ──────
    os.makedirs(os.path.dirname(file_name + ".json"), exist_ok=True)

    manifest_path = file_name + "_manifest.json"
    with open(manifest_path, "w") as f:
        f.write(manifest.to_json())
    print(f"Manifest saved to {manifest_path}")
    print(f"  decision_policy_fingerprint: {manifest.decision_policy_fingerprint}")
    print(f"  run_fingerprint: {manifest.run_fingerprint}")

    # ── Persist preflight (non-dry real/mocked runs only) ────────────
    if projection is not None:
        persist_projection(projection, file_name + "_preflight.json")

    budget_ledger = setup_budget_ledger(setup)

    # ── Todo 3 V2: auth BEFORE solver ─────────────────────────────
    ok, err = consume_authorization(manifest, setup)
    if not ok:
        print(f"FATAL: {err}", file=sys.stderr)
        sys.exit(1)

    solver = setup_mock_solver(args, setup, solver_cls=FederatedSolver)
    fn_map = {
        "naive": solver.naive_solve,
        "tot": solver.solve,
        "speculative_solve": solver.speculative_solve,
        "federated_solve": solver.federated_solve,
    }
    solve_fn = fn_map[solve_method]

    logs, cnt_avg, cnt_any = [], 0, 0
    stop_reason: str | None = None
    token_consumption = {}
    oracle_task_metrics = []
    per_task_metrics: list = []
    prev_usage_snapshot: dict = {}

    # ── Resume: restore completed task records into logs/counters ─
    if resume and skip_ids:
        ca, cy = restore_completed_tasks(
            journal, skip_ids, logs=logs,
            per_task_metrics=per_task_metrics)
        cnt_avg += ca; cnt_any += cy

    for i in setup.task_ids:
        if i in skip_ids:
            print(f"Task {i}: skipping (already completed in journal)")
            continue

        # ── Todo 3: per-task cap checks ────────────────────────────
        budget_ledger.mark_task_started()
        can_cont, cap_reason = check_budget_caps(
            budget_ledger, getattr(args, "max_cost_usd", None),
        )
        if not can_cont:
            stop_reason = cap_reason
            print(f"Stop: {cap_reason}")
            break

        print(f"Task {i}")
        task = get_task(args.task)
        task_recorder = make_task_recorder(is_game24, i,
            task.get_input(i), setup.decision_policy,
            setup.oracle_audit)

        solve_client = "local_client" if getattr(args, "search_policy", "baseline") == "dqn" else "remote_client"
        ys, info = solve_fn(task, i, assign_strategy=args.assign_strategy,
                            solve_client=solve_client, task_recorder=task_recorder)
        infos, output_list = [], []
        for y in ys:
            r, new_output = task.test_output_modify(i, y)
            if new_output not in output_list:
                output_list.append(new_output)
            else:
                r = {"r": 0}
            infos.append(r)

        if getattr(args, "search_policy", "baseline") == "dqn":
            session = getattr(solver, "dqn_session", None)
            if session is not None:
                session.finalize(any(ri["r"] for ri in infos))

        oracle = task_oracle_metrics(task, i, ys)
        if oracle is not None:
            info["blocksworld_oracle"] = oracle
            oracle_task_metrics.append(oracle)

        token_consumption = get_model_usage_summary()
        info.update({
            "idx": i, "ys": ys, "infos": infos,
            "usage_so_far": token_consumption,
            "time_consumption": solver.latency_dict,
        })
        logs.append(info)

        # ── Todo 7: atomic persistence after each task ────────────
        if task_recorder is not None:
            try:
                task_metrics = task_recorder.to_task_metrics()
                per_task_metrics.append(task_metrics)
                persist_task(journal, i, ys, info,
                             json.dumps(task_metrics),
                             task_recorder.total_elapsed_ms)
            except Exception as e:
                print(f"Warning: persistence write failed for task {i}: {e}")

        # ── Todo 7: per-task usage delta ──────────────────────────
        current = get_model_usage_summary()
        delta: dict = {}
        for m, stats in current.items():
            p = prev_usage_snapshot.get(m, {})
            for k in ("completion_tokens", "prompt_tokens",
                      "total_tokens", "total_calls"):
                cv, pv = int(stats.get(k, 0)), int(p.get(k, 0))
                if cv != pv:
                    delta[f"{m}.{k}"] = cv - pv
        info["usage_delta"] = delta
        # ── Raw-unit spending from usage delta ────────────────
        task_cost_raw = compute_task_raw_unit_cost(prev_usage_snapshot, current)
        prev_usage_snapshot = current

        if task_cost_raw > 0.0:
            if not budget_ledger.spend(task_cost_raw):
                print(f"FATAL: budget exceeded on task {i}. "
                      f"Remaining: {budget_ledger.remaining:.6f}")
                stop_reason = "max_raw_unit_limit"
                break

        with open(file_name + ".json", "w") as f:
            json.dump(logs, f, indent=4)

        accs = [ri["r"] for ri in infos]
        cnt_avg += sum(accs)
        cnt_any += any(accs)
        print(i, "sum(accs)", sum(accs), "cnt_avg", cnt_avg, "cnt_any", cnt_any, "\n")
        if cost_exceeded(token_consumption, args.max_cost_usd):
            stop_reason = "max_cost_usd"
            break

    n = len(logs)
    avg_sum = cnt_avg / n if n else 0.0
    accuracy = cnt_any / n if n else 0.0
    print("The average sum is ", avg_sum, ". The accuracy is: ", accuracy)
    print("Token consumption: ", token_consumption)

    res_json: dict = {
        "avg_sum": avg_sum, "acc": accuracy,
        "sm": args.localbackend, "llm": args.remotebackend,
    }
    res_json.update(token_consumption)
    res_json["estimated_cost_usd"] = total_cost_usd(token_consumption)
    if stop_reason is not None:
        res_json["stop_reason"] = stop_reason
    enrich_result(res_json, manifest, budget_ledger, stop_reason, projection=projection)
    merge_oracle_metrics(oracle_task_metrics, res_json)

    # ── Todo 7: experiment aggregation ────────────────────────────
    if per_task_metrics:
        from federated_mcts.evaluation.game24_metrics import summarize_experiment_metrics
        res_json["experiment_metrics"] = summarize_experiment_metrics(per_task_metrics)
        res_json["task_count"] = len(per_task_metrics)

    with open(file_name + "_performance.json", "w") as f:
        json.dump(res_json, f, indent=4)
        print(f"Performance results saved to {file_name}_performance.json")


if __name__ == "__main__":
    args = parse_args()
    if args.filter:
        print("Filtering enabled. Only runs with specific criteria will be executed.")
    else:
        print("No filtering applied. All runs will be executed.")
    run(args, args.solve_method)
