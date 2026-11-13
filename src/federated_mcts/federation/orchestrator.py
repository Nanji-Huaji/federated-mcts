"""Federated MCTS orchestrator — manages multi-model reasoning across clients."""

import time
from contextlib import contextmanager
from math import ceil
from typing import Dict, List, Tuple, Callable, Optional
from federated_mcts.utils.exhaustive import assert_never

import numpy as np

from federated_mcts.core.diverse_search import DiverseSearch
from federated_mcts.federation._dqn_finish import build_dqn_session
from federated_mcts.core.search_policy import state_key
from federated_mcts.federation.evaluator_cache import TaskEvaluatorCaches
from federated_mcts.federation.federated_execution import FederatedExecutionMixin, attach_dqn_scores
from federated_mcts.federation.standard_solvers import GenerationClient, StandardSolversMixin
from federated_mcts.federation.task_assign import (
    TaskAssignment,
)
from federated_mcts.federation.time_tracker import ClientTiming, TimeTracker
from federated_mcts.models.usage import get_usage_tracker


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

    def __init__(self, args, *, clients: dict[str, GenerationClient], time_tracker=None):
        """Accept resolved options and clients; construction performs no I/O.

        Real and offline adapters use this same interface. Configuration and
        credential resolution belong to the runner's client factory.
        """
        self.args = args
        self.gpts = dict(clients)
        self.time_tracker = time_tracker if time_tracker is not None else TimeTracker(list(clients))
        self._strategy_cache: dict = {}
        self._evaluator_caches = TaskEvaluatorCaches()
        self.dqn_session = None

    def solve_task(self, task, index: int, *, method="tot", **kwargs):
        from federated_mcts.federation.outcome import finish_task
        return finish_task(self, task, index, method=method, **kwargs)

    @contextmanager
    def deferred_dqn_terminal_flush(self):
        """Keep the DQN terminal record unpublished until :meth:`finish_task`.

        The exact success flag comes from the task verifier after the search
        returns.  Holding the terminal record until then lets it be recorded
        once with its final reward instead of being rewritten afterwards.
        """
        previous = getattr(self, "_defer_dqn_terminal", False)
        self._defer_dqn_terminal = True
        try:
            yield
        finally:
            self._defer_dqn_terminal = previous

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
        self._ensure_evaluator_caches()
        x = task.get_input(idx)
        ys = [""]
        infos = []
        search_policy = getattr(self.args, "search_policy", "baseline")
        diverse_search = DiverseSearch(
            self.args, evaluator_cache_for=self.evaluator_cache,
        ) if search_policy == "diverse" else None
        dqn_session = build_dqn_session(
            self.args, evaluator_cache_for=self.evaluator_cache,
        ) if search_policy == "dqn" else None
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

            # Generation cost is the wall clock of the (parallel) generation
            # phase and this run's scoped usage delta -- never the sum of the
            # per-client durations, which double counts parallel work.
            generation_started = time.perf_counter()
            generation_tokens_before = get_usage_tracker().get_total_tokens()
            all_new_ys, all_values, step_infos, client_times_result = self._run_assignments(
                task, x, step, task_assignments, to_print, n_gen,
            )
            generation_latency = time.perf_counter() - generation_started
            step_tokens = max(
                0, get_usage_tracker().get_total_tokens() - generation_tokens_before)

            for client_name, times in client_times_result.items():
                step_client_times[client_name]["generation"] += times["generation"]
                step_client_times[client_name]["evaluation"] += times["evaluation"]

            if not all_new_ys:
                # Nothing was generated: there is no candidate to score,
                # truncate or select.  Keep the current frontier, record the
                # empty step explicitly, and move on instead of falling
                # through to selection code with an undefined variable.
                self.time_tracker.accumulate_step(step_client_times)
                if not hasattr(self, "_legacy_assign_fn"):
                    strategy.update(context, {})
                step_info = {
                    "step": step, "x": x, "ys": ys, "new_ys": list(ys),
                    "values": [1.0] * len(ys), "select_new_ys": list(ys),
                    "client_infos": step_infos, "task_assignments": task_assignments,
                    "step_client_times": step_client_times,
                    "search_metrics": None,
                }
                if dqn_session is not None:
                    step_info["dqn_transitions"] = []
                infos.append(step_info)
                ys = list(step_info["select_new_ys"])
                continue

            stopped = False
            search_metrics = None
            dqn_transitions = None
            if diverse_search is not None or dqn_session is not None:
                eval_client_name = "remote_client" if "remote_client" in self.gpts else next(
                    assignment["eval_client"] for assignment in task_assignments if assignment["ys"]
                )
            if diverse_search is not None:
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
                    client_info["scored_ys"] = list(client_info.get("output_ys", []))
                    client_info["values"] = [
                        value_by_state.get(state_key(task, x, candidate), 0.0)
                        for candidate in client_info.get("output_ys", [])
                    ]
                search_metrics = {
                    "raw_candidates": raw_candidate_count,
                    "unique_states": len(all_new_ys),
                    "beam_width": len(select_new_ys),
                }

            if dqn_session is not None:
                outcome = dqn_session.process_step(
                    args=self.args,
                    task=task,
                    x=x,
                    candidates=all_new_ys,
                    client=self.gpts[eval_client_name],
                    evaluator_id=eval_client_name,
                    step=step,
                    total_steps=task.steps,
                    step_tokens=float(step_tokens),
                    step_latency=generation_latency,
                )
                all_new_ys = outcome.candidates
                all_values = outcome.values
                select_new_ys = outcome.selected
                stopped = outcome.stopped
                search_metrics = outcome.search_metrics
                step_client_times[eval_client_name]["evaluation"] += outcome.eval_seconds
                dqn_transitions = outcome.transitions
                # Only put scores the session actually produced back on the
                # client infos; truncated/deduplicated candidates stay unscored.
                attach_dqn_scores(task, x, outcome.candidates, outcome.values, step_infos)

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
            dqn_session.finish_episode(
                flush_terminal=not getattr(self, "_defer_dqn_terminal", False))
            remaining = dqn_session.drain_transitions()
            if remaining and infos:
                infos[-1].setdefault("dqn_transitions", []).extend(remaining)

        if to_print:
            print(f"Final results: {ys}")
        return ys, {"steps": infos}
