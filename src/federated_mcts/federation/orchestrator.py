"""Federated MCTS orchestrator — manages multi-model reasoning across clients."""

import itertools
import json
import os
import time
from functools import partial
from math import ceil
from typing import Dict, List, Tuple, Callable, Optional

import numpy as np

from federated_mcts.models.api_client import gpt
from federated_mcts.models import DraftModel, get_model_usage_summary
from federated_mcts.core.generation import get_proposals, get_samples
from federated_mcts.core.evaluation import get_values, get_votes
from federated_mcts.federation.task_assign import (
    TaskAssignment,
    speculative_federated_assign_task,
)
from federated_mcts.federation.time_tracker import TimeTracker
from federated_mcts.utils.uncertainty import Uncertainty


class FederatedSolver:
    """Orchestrates MCTS reasoning across multiple model clients."""

    def __init__(self, args):
        self.args = args
        model_config = args.model_config
        with open(model_config, "r") as f:
            self.model_config = json.load(f)

        for model in self.model_config:
            if "api_key_environment_variable" in model:
                env_var = model["api_key_environment_variable"]
                model["api_key"] = os.environ.get(env_var, "")
            elif "api_key" not in model:
                model["api_key"] = os.environ.get("OPENAI_API_KEY", "")
            else:
                model["api_key"] = ""

            if "api_base" not in model:
                model["api_base"] = "https://try-chatapi.com/v1"

        self.gpts = {}
        for model_cfg in self.model_config:
            client_name = model_cfg["client_name"]
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

    @property
    def latency_dict(self) -> Dict[str, float]:
        return self.time_tracker.latency_dict

    # ── Non-federated solvers ────────────────────────────────────

    def naive_solve(self, task, idx, to_print=True, solve_client="remote_client", **kwargs):
        gpt_fn = self.gpts[solve_client]
        x = task.get_input(idx)
        ys = get_samples(
            self.args, task, x, "", self.args.n_generate_sample,
            self.args.prompt_sample, stop=None
        )
        return ys, {}

    def solve(self, task, idx, to_print=True, solve_client="remote_client", **kwargs):
        gpt_fn = self.gpts[solve_client]
        x = task.get_input(idx)
        ys = [""]
        infos = []
        for step in range(task.steps):
            if self.args.method_generate == "sample":
                new_ys = [
                    get_samples(self.args, task, x, y, self.args.n_generate_sample,
                                prompt_sample=self.args.prompt_sample, stop=task.stops[step],
                                client=gpt_fn)
                    for y in ys
                ]
            elif self.args.method_generate == "propose":
                new_ys = [get_proposals(self.args, step, task, x, y, client=gpt_fn) for y in ys]
            new_ys = list(itertools.chain(*new_ys))
            ids = list(range(len(new_ys)))

            if self.args.method_evaluate == "vote":
                values = get_votes(self.args, task, x, new_ys, self.args.n_evaluate_sample, client=gpt_fn)
            elif self.args.method_evaluate == "value":
                values = get_values(self.args, task, x, new_ys, self.args.n_evaluate_sample, client=gpt_fn)

            if self.args.method_select == "sample":
                ps = np.array(values) / sum(values)
                select_ids = np.random.choice(ids, size=self.args.n_select_sample, p=ps).tolist()
            elif self.args.method_select == "greedy":
                select_ids = sorted(ids, key=lambda x: values[x], reverse=True)[:self.args.n_select_sample]
            select_new_ys = [new_ys[select_id] for select_id in select_ids]

            infos.append({"step": step, "x": x, "ys": ys, "new_ys": new_ys,
                          "values": values, "select_new_ys": select_new_ys})
            ys = select_new_ys

        return ys, {"steps": infos}

    # ── Speculative federated solver ─────────────────────────────

    def speculative_solve(self, task, idx, to_print=True, solve_client=None, eval_client=None, **kwargs):
        local_client = self.gpts.get("local_client", self.gpts[list(self.gpts.keys())[0]])
        remote_client = self.gpts.get("remote_client", self.gpts[list(self.gpts.keys())[-1]])

        x = task.get_input(idx)
        ys = [""]
        infos = []

        for step in range(task.steps):
            solve_gpt = local_client
            eval_gpt = remote_client

            if getattr(self.args, "warm_start", False) and step == 0:
                solve_gpt = remote_client
            if step + 1 == task.steps and getattr(self.args, "last_lm", False):
                solve_gpt = remote_client
            if getattr(self.args, "slm_eval", False):
                eval_gpt = local_client

            start_time = time.time()
            if self.args.method_generate == "sample":
                new_ys = [
                    get_samples(self.args, task, x, y, self.args.n_generate_sample,
                                prompt_sample=self.args.prompt_sample, stop=task.stops[step],
                                client=solve_gpt)
                    for y in ys
                ]
            elif self.args.method_generate == "propose":
                new_ys = [get_proposals(self.args, step, task, x, y, client=solve_gpt) for y in ys]
            new_ys = list(itertools.chain(*new_ys))
            ids = list(range(len(new_ys)))
            self.time_tracker.record_generation("speculative", time.time() - start_time)

            start_time = time.time()
            if self.args.method_evaluate == "vote":
                values = get_votes(self.args, task, x, new_ys, self.args.n_evaluate_sample, client=eval_gpt)
            elif self.args.method_evaluate == "value":
                values = get_values(self.args, task, x, new_ys, self.args.n_evaluate_sample, client=eval_gpt)
            self.time_tracker.record_evaluation("speculative", time.time() - start_time)

            if self.args.method_select == "sample" and sum(values) > 0:
                ps = np.array(values) / sum(values)
                select_ids = np.random.choice(ids, size=self.args.n_select_sample, p=ps).tolist()
            elif self.args.method_select == "greedy":
                select_ids = sorted(ids, key=lambda x: values[x], reverse=True)[:self.args.n_select_sample]
            select_new_ys = [new_ys[select_id] for select_id in select_ids]

            infos.append({"step": step, "x": x, "ys": ys, "new_ys": new_ys,
                          "values": values, "select_new_ys": select_new_ys})
            ys = select_new_ys

        return ys, {"steps": infos}

    # ── Federated solver ─────────────────────────────────────────

    def federated_solve(
        self,
        task,
        idx,
        to_print=True,
        assign_function: Callable = speculative_federated_assign_task,
        **kwargs,
    ):
        x = task.get_input(idx)
        ys = [""]
        infos = []

        client_names = list(self.gpts.keys())
        if not client_names:
            raise ValueError("No clients available")

        for step in range(task.steps):
            step_client_times = {name: {"generation": 0.0, "evaluation": 0.0} for name in client_names}
            task_assignments = assign_function(client_names, ys)

            all_new_ys, all_values, step_infos = [], [], []

            for assignment in task_assignments:
                if assignment["ys"]:
                    n_gen = ceil(self.args.n_generate_sample / len([a for a in task_assignments if a["ys"]]))
                    client_new_ys, client_values, client_info, client_times = self._client_step(
                        task, x, assignment["ys"], step,
                        assignment["solve_client"], assignment["eval_client"],
                        to_print, n_generate_sample=n_gen,
                    )
                    step_client_times[assignment["solve_client"]]["generation"] += client_times["generation"]
                    step_client_times[assignment["eval_client"]]["evaluation"] += client_times["evaluation"]
                    all_new_ys.extend(client_new_ys)
                    all_values.extend(client_values)
                    step_infos.append(client_info)

            self.time_tracker.accumulate_step(step_client_times)

            if not all_new_ys:
                all_new_ys = ys.copy()
                all_values = [1.0] * len(ys)

            ids = list(range(len(all_new_ys)))
            if self.args.method_select == "sample" and sum(all_values) > 0:
                ps = np.array(all_values) / sum(all_values)
                select_ids = np.random.choice(ids, size=min(self.args.n_select_sample, len(ids)), p=ps).tolist()
            elif self.args.method_select == "greedy":
                select_ids = sorted(ids, key=lambda x: all_values[x], reverse=True)[:self.args.n_select_sample]
            else:
                select_ids = ids[:self.args.n_select_sample]

            select_new_ys = [all_new_ys[select_id] for select_id in select_ids]

            infos.append({
                "step": step, "x": x, "ys": ys, "new_ys": all_new_ys,
                "values": all_values, "select_new_ys": select_new_ys,
                "client_infos": step_infos, "task_assignments": task_assignments,
                "step_client_times": step_client_times,
            })
            ys = select_new_ys

        return ys, {"steps": infos}

    # ── Per-client step ──────────────────────────────────────────

    def _client_step(self, task, x, client_ys, step, solve_client_name, eval_client_name,
                     to_print=False, n_generate_sample=None, uncertainty_backoff=False, uncertainty_threshold=0.8):
        uncertainty_calculator = Uncertainty(uncertainty_threshold=uncertainty_threshold, uncertainty_method="entropy")
        solve_gpt = self.gpts[solve_client_name]
        eval_gpt = self.gpts[eval_client_name]
        new_ys = []
        n_gen = n_generate_sample if n_generate_sample else self.args.n_generate_sample

        gen_start = time.time()
        for y in client_ys:
            if self.args.method_generate == "sample":
                result = get_samples(self.args, task, x, y, n_generate_sample=n_gen,
                                     prompt_sample=self.args.prompt_sample, stop=task.stops[step],
                                     client=solve_gpt, get_logprobs=uncertainty_backoff)
                if isinstance(result, tuple):
                    samples_to_add = result[0]
                else:
                    samples_to_add = result
                new_ys.extend(samples_to_add if isinstance(samples_to_add, list) else [samples_to_add])
            elif self.args.method_generate == "propose":
                proposals = get_proposals(self.args, step, task, x, y, client=solve_gpt)
                new_ys.extend(proposals)
        gen_time = time.time() - gen_start

        unique, seen = [], set()
        for y in new_ys:
            if y not in seen:
                unique.append(y)
                seen.add(y)
        new_ys = unique

        eval_start = time.time()
        if new_ys:
            if self.args.method_evaluate == "vote":
                values = get_votes(self.args, task, x, new_ys, self.args.n_evaluate_sample, client=eval_gpt)
            elif self.args.method_evaluate == "value":
                values = get_values(self.args, task, x, new_ys, self.args.n_evaluate_sample, client=eval_gpt)
            else:
                values = [1.0] * len(new_ys)
        else:
            values = []
        eval_time = time.time() - eval_start

        client_info = {
            "solve_client_name": solve_client_name, "eval_client_name": eval_client_name,
            "input_ys": client_ys, "output_ys": new_ys, "values": values, "step": step,
        }
        client_times = {"generation": gen_time, "evaluation": eval_time}

        return new_ys, values, client_info, client_times
