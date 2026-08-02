"""Federated MCTS orchestrator — manages multi-model reasoning across clients."""

import json
import os
import time
from functools import partial
from math import ceil
from typing import Dict, List, Tuple, Callable, Optional
from federated_mcts.utils.exhaustive import assert_never

import numpy as np

from federated_mcts.models.api_client import gpt
from federated_mcts.models import DraftModel, get_model_usage_summary
from federated_mcts.core.diverse_search import DiverseSearch
from federated_mcts.core.dqn.factory import build_dqn_session
from federated_mcts.core.search_policy import state_key
from federated_mcts.federation.federated_execution import FederatedExecutionMixin
from federated_mcts.federation.standard_solvers import GenerationClient, StandardSolversMixin
from federated_mcts.federation.task_assign import (
    TaskAssignment,
)
from federated_mcts.federation.time_tracker import ClientTiming, TimeTracker


class NoClientsConfiguredError(RuntimeError):
    def __str__(self) -> str:
        return "No clients available for federated solving"


class FederatedSolver(StandardSolversMixin, FederatedExecutionMixin):
    """Orchestrates MCTS reasoning across multiple model clients.

    Replaces the old ToTMethods class. Supports:
    - Single-model solving (solve, naive_solve)
    - Speculative federated solving (speculative_solve) — local generation + remote evaluation
    - Full federated solving (federated_solve) — multi-model task assignment
    """

    def __init__(self, args):
        self.args = args
        model_config = args.model_config
        with open(model_config, "r") as f:
            self.model_config = json.load(f)

        for model in self.model_config:
            print(f"model: {model}")
            if "api_key_environment_variable" in model:
                env_var = model["api_key_environment_variable"]
                model["api_key"] = os.environ.get(env_var, "")
            elif "api_key" not in model:
                model["api_key"] = os.environ.get("OPENAI_API_KEY", "")
            else:
                model["api_key"] = ""
                print(f"Warning: No API key found for {model['client_name']}, using empty string.")
            if "api_base" not in model:
                model["api_base"] = "https://try-chatapi.com/v1"
                print(f"Warning: No API base found for {model['client_name']}, using default base.")

        self.gpts: dict[str, GenerationClient] = {}
        for model_cfg in self.model_config:
            client_name = model_cfg["client_name"]
            if model_cfg.get("type") == "vllm":
                from federated_mcts.models.vllm_client import VLLMClient
                client = VLLMClient.from_config(model_cfg)
                self.gpts[client_name] = client
            else:
                self.gpts[client_name] = partial(
                    gpt,
                    model=model_cfg["model"],
                    temperature=model_cfg.get("temperature", args.temperature),
                    max_tokens=model_cfg.get("max_tokens", 1000),
                    api_base=model_cfg["api_base"],
                    api_key=model_cfg.get("api_key", ""),
                )

        client_names = [m["client_name"] for m in self.model_config]
        self.time_tracker = TimeTracker(client_names)
        self._strategy_cache = {}

    @property
    def latency_dict(self) -> Dict[str, float]:
        return self.time_tracker.latency_dict

    def early_stop(self, task):
        pass

    def _get_strategy(self, name: str, task):
        if not hasattr(self, "_strategy_cache"):
            self._strategy_cache = {}
        key = (name, type(task).__name__, task.steps)
        if key not in self._strategy_cache:
            from federated_mcts.federation.task_assign import get_strategy
            self._strategy_cache[key] = get_strategy(name, total_steps=task.steps)
        return self._strategy_cache[key]

    def federated_solve(self, task, idx, to_print=True,
                        assign_strategy=None, **kwargs):
        x = task.get_input(idx)
        ys = [""]
        infos = []
        search_policy = getattr(self.args, "search_policy", "baseline")
        diverse_search = DiverseSearch(self.args) if search_policy == "diverse" else None
        dqn_session = build_dqn_session(self.args) if search_policy == "dqn" else None
        self.dqn_session = dqn_session

        client_names = list(self.gpts.keys())
        if not client_names:
            raise NoClientsConfiguredError

        # Initialize assignment strategy
        from federated_mcts.federation.task_assign import BaseAssignStrategy, RoundRobinStrategy, AssignmentContext
        strategy: BaseAssignStrategy
        match assign_strategy:
            case None:
                strategy = RoundRobinStrategy(eval_client="remote_client")
            case str() as strategy_name:
                strategy = self._get_strategy(strategy_name, task)
            case BaseAssignStrategy() as strategy_instance:
                strategy = strategy_instance
            case legacy if callable(legacy):
                strategy = RoundRobinStrategy(eval_client="remote_client")
                self._legacy_assign_fn = legacy
            case unreachable:
                assert_never(unreachable)

        for step in range(task.steps):
            if to_print:
                print(f"Step {step} of {task.steps} in Task {idx}")

            step_client_times: dict[str, ClientTiming] = {
                name: {"generation": 0.0, "evaluation": 0.0} for name in client_names
            }
                        # Build context and assign tasks to clients
            context: AssignmentContext = {
                "step": step,
                "total_steps": task.steps,
                "task_name": task.__class__.__name__,
            }
            if hasattr(self, "_legacy_assign_fn"):
                task_assignments = self._legacy_assign_fn(client_names, ys)
            else:
                task_assignments = strategy.assign(client_names, ys, context)

            active_count = len([a for a in task_assignments if a["ys"]])
            n_gen = ceil(self.args.n_generate_sample / active_count) if active_count else self.args.n_generate_sample

            all_new_ys, all_values, step_infos, client_times_result = self._run_assignments(
                task, x, step, task_assignments, to_print, n_gen,
            )

            for client_name, times in client_times_result.items():
                step_client_times[client_name]["generation"] += times["generation"]
                step_client_times[client_name]["evaluation"] += times["evaluation"]

            stopped = False
            search_metrics = None
            dqn_transitions = None
            if (diverse_search is not None or dqn_session is not None) and all_new_ys:
                eval_client_name = "remote_client" if "remote_client" in self.gpts else next(
                    assignment["eval_client"] for assignment in task_assignments if assignment["ys"]
                )
            if diverse_search is not None and all_new_ys:
                raw_candidate_count = len(all_new_ys)
                ranking_start = time.time()
                decision = diverse_search.decide(
                    self.args,
                    task,
                    x,
                    all_new_ys,
                    self.gpts[eval_client_name],
                    eval_client_name,
                )
                step_client_times[eval_client_name]["evaluation"] += time.time() - ranking_start
                all_new_ys = decision.candidates
                all_values = decision.values
                select_new_ys = decision.selected
                stopped = decision.stopped
                value_by_state = {
                    state_key(task, x, candidate): value
                    for candidate, value in zip(all_new_ys, all_values)
                }
                for client_info in step_infos:
                    client_info["values"] = [
                        value_by_state.get(state_key(task, x, candidate), 0.0)
                        for candidate in client_info.get("output_ys", [])
                    ]
                search_metrics = {
                    "raw_candidates": raw_candidate_count,
                    "unique_states": len(all_new_ys),
                    "beam_width": len(select_new_ys),
                }

            if dqn_session is not None and all_new_ys:
                outcome = dqn_session.process_step(
                    args=self.args,
                    task=task,
                    x=x,
                    candidates=all_new_ys,
                    client=self.gpts[eval_client_name],
                    evaluator_id=eval_client_name,
                    step=step,
                    total_steps=task.steps,
                    step_tokens=0.0,
                    step_latency=sum(
                        times["generation"] + times["evaluation"]
                        for times in client_times_result.values()
                    ),
                )
                all_new_ys = outcome.candidates
                all_values = outcome.values
                select_new_ys = outcome.selected
                stopped = outcome.stopped
                search_metrics = outcome.search_metrics
                step_client_times[eval_client_name]["evaluation"] += outcome.eval_seconds
                dqn_transitions = outcome.transitions

            self.time_tracker.accumulate_step(step_client_times)

            # Update strategy with per-model rewards
            if not hasattr(self, "_legacy_assign_fn"):
                per_model_rewards = {}
                for ci in step_infos:
                    name = ci.get("solve_client_name", "")
                    vals = ci.get("values", [])
                    if name and vals:
                        per_model_rewards[name] = sum(vals) / len(vals)
                strategy.update(context, per_model_rewards)

            if not all_new_ys:
                all_new_ys = ys.copy()
                all_values = [1.0] * len(ys)

            if diverse_search is None and dqn_session is None:
                ids = list(range(len(all_new_ys)))
                if self.args.method_select == "sample" and sum(all_values) > 0:
                    ps = np.array(all_values) / sum(all_values)
                    select_ids = np.random.choice(ids, size=min(self.args.n_select_sample, len(ids)), p=ps).tolist()
                elif self.args.method_select == "greedy":
                    select_ids = sorted(ids, key=lambda x: all_values[x], reverse=True)[:self.args.n_select_sample]
                else:
                    select_ids = ids[:self.args.n_select_sample]
                select_new_ys = [all_new_ys[select_id] for select_id in select_ids]

            if to_print and all_new_ys and all_values:
                sorted_new_ys, sorted_values = zip(*sorted(zip(all_new_ys, all_values), key=lambda x: x[1], reverse=True))
                print(f"-- new_ys --: {sorted_new_ys[:5]}\n-- sol values --: {sorted_values[:5]}\n-- choices --: {select_new_ys}\n")

            step_info = {
                "step": step, "x": x, "ys": ys, "new_ys": all_new_ys,
                "values": all_values, "select_new_ys": select_new_ys,
                "client_infos": step_infos, "task_assignments": task_assignments,
                "step_client_times": step_client_times,
                "search_metrics": search_metrics,
            }
            if dqn_session is not None:
                step_info["dqn_transitions"] = dqn_transitions if dqn_transitions is not None else []
            infos.append(step_info)
            ys = select_new_ys
            if stopped:
                break

        if dqn_session is not None:
            dqn_session.finish_episode()
            remaining = dqn_session.drain_transitions()
            if remaining and infos:
                infos[-1].setdefault("dqn_transitions", []).extend(remaining)

        if to_print:
            print(f"Final results: {ys}")
        return ys, {"steps": infos}
