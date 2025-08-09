import itertools
import numpy as np
from functools import partial
from src.tot.models import gpt
import os, re
from src.tot.pattern_match import check_and_fix_last_line
import time
from src.tot.tasks import get_task

import json

from math import ceil

from typing import Tuple, List, Dict, Callable, Union, TypedDict


class TaskAssignment(TypedDict):
    solve_client: str  # the client used
    eval_client: str  # the client used for evaluation
    ys: List[str]  # the output candidates assigned to this client]


def get_value(
    args,
    task,
    x,
    y,
    n_evaluate_sample,
    cache_value=True,
    api_key=None,
    api_base=None,
    model=None,
    client: Callable | None = None,
):
    value_prompt = task.value_prompt_wrap(x, y)
    if cache_value and value_prompt in task.value_cache:
        return task.value_cache[value_prompt]
    if client is None:
        value_outputs = gpt(
            args, value_prompt, n=n_evaluate_sample, stop=None, api_key=api_key, api_base=api_base, model=model
        )
    else:
        value_outputs = client(args, value_prompt, n=n_evaluate_sample, stop=None)
    value = task.value_outputs_unwrap(x, y, value_outputs)
    if cache_value:
        task.value_cache[value_prompt] = value
    return value


def get_values(
    args, task, x, ys, n_evaluate_sample, cache_value=True, api_key=None, api_base=None, model=None, client=None
):
    values = []
    local_value_cache = {}
    for y in ys:  # each partial output
        if y in local_value_cache:  # avoid duplicate candidates
            value = 0
        else:
            # jinyu
            value, final = task.pre_value_check(y, args.eval_rule)
            if value == 0 and final == False:
                count = 0
                while value == 0 and count < 2:
                    value = get_value(
                        args,
                        task,
                        x,
                        y,
                        n_evaluate_sample,
                        cache_value=cache_value,
                        api_key=api_key,
                        api_base=api_base,
                        model=model,
                        client=client,
                    )
                    count += 1
            local_value_cache[y] = value
        values.append(value)
    return values


def get_votes(args, task, x, ys, n_evaluate_sample, api_key=None, api_base=None, model=None, client=None):
    vote_prompt = task.vote_prompt_wrap(x, ys)
    if client is None:
        vote_outputs = gpt(
            args, vote_prompt, n=n_evaluate_sample, stop=None, api_key=api_key, api_base=api_base, model=model
        )
    else:
        vote_outputs = client(args, vote_prompt, n=n_evaluate_sample, stop=None)
    values = task.vote_outputs_unwrap(vote_outputs, len(ys))
    return values


def get_proposals_with_check(args, step, task, x, y, api_key=None, api_base=None, model=None, client=None):
    # jinyu:
    need_generate = task.pre_generate_check(y)
    if need_generate == False:  # no need to generate new proposals
        return [y]

    new_proposal_list, run_times = [], 0
    time_constraint, len_constraint = 6, 4

    while len(new_proposal_list) < len_constraint and run_times < time_constraint:  # Generate at least 4 proposals

        propose_prompt = task.propose_prompt_wrap(x, y)
        if client is None:
            proposals = gpt(
                args,
                propose_prompt,
                n=1,
                stop=None,
                api_key=api_key,
                api_base=api_base,
                model=model,
            )[
                0
            ].split("\n")
        else:
            proposals = client(args, propose_prompt, n=1, stop=None)[0].split("\n")
        # jinyu: check the format
        for pro in proposals:
            is_correct, updated_new_proposal = task.process_generate_result(pro, x, y, args.check_format)
            if is_correct:
                if updated_new_proposal not in new_proposal_list:
                    new_proposal_list.append(updated_new_proposal)
        run_times += 1

    if run_times >= time_constraint:
        print("runtime ", run_times)
    if len(new_proposal_list) == 0:
        return [y]
    return new_proposal_list


def get_proposals_without_check(args, step, task, x, y, api_key=None, api_base=None, model=None, client=None):
    propose_prompt = task.propose_prompt_wrap(x, y)
    if client is None:
        proposals = gpt(
            args,
            propose_prompt,
            n=1,
            stop=None,
            api_key=api_key,
            api_base=api_base,
            model=model,
        )[
            0
        ].split("\n")
    else:
        proposals = client(args, propose_prompt, n=1, stop=None)[0].split("\n")
    return [y + _ + "\n" for _ in proposals]


def get_proposals(args, step, task, x, y, api_key=None, api_base=None, model=None, client=None):
    if args.check_format:
        return get_proposals_with_check(
            args, step, task, x, y, api_key=api_key, api_base=api_base, model=model, client=client
        )
    else:
        return get_proposals_without_check(
            args, step, task, x, y, api_key=api_key, api_base=api_base, model=model, client=client
        )


def get_samples(
    args, task, x, y, n_generate_sample, prompt_sample, stop, api_key=None, api_base=None, model=None, client=None
):
    if prompt_sample == "standard":
        prompt = task.standard_prompt_wrap(x, y)
    elif prompt_sample == "cot":
        prompt = task.cot_prompt_wrap(x, y)
    else:
        raise ValueError(f"prompt_sample {prompt_sample} not recognized")
    if client is None:
        samples = gpt(args, prompt, n=n_generate_sample, stop=stop, api_key=api_key, api_base=api_base, model=model)
    else:
        samples = client(args, prompt, n=n_generate_sample, stop=stop)
    return [y + _ for _ in samples]


def solve(args, task, idx, to_print=True):
    # Errors will occur if the function is deleted.
    global gpt
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    print(gpt)
    x = task.get_input(idx)  # input
    ys = [""]  # current output candidates
    infos = []
    for step in range(task.steps):
        # generation
        if args.method_generate == "sample":
            new_ys = [
                get_samples(
                    args, task, x, y, args.n_generate_sample, prompt_sample=args.prompt_sample, stop=task.stops[step]
                )
                for y in ys
            ]
        elif args.method_generate == "propose":
            new_ys = [get_proposals(args, step, task, x, y) for y in ys]
        new_ys = list(itertools.chain(*new_ys))
        ids = list(range(len(new_ys)))
        # evaluation
        if args.method_evaluate == "vote":
            values = get_votes(args, task, x, new_ys, args.n_evaluate_sample)
        elif args.method_evaluate == "value":
            values = get_values(args, task, x, new_ys, args.n_evaluate_sample)

        # selection
        if args.method_select == "sample":
            ps = np.array(values) / sum(values)
            select_ids = np.random.choice(ids, size=args.n_select_sample, p=ps).tolist()
        elif args.method_select == "greedy":
            select_ids = sorted(ids, key=lambda x: values[x], reverse=True)[: args.n_select_sample]
        select_new_ys = [new_ys[select_id] for select_id in select_ids]

        # log
        if to_print:
            sorted_new_ys, sorted_values = zip(*sorted(zip(new_ys, values), key=lambda x: x[1], reverse=True))
            print(f"-- new_ys --: {sorted_new_ys}\n-- sol values --: {sorted_values}\n-- choices --: {select_new_ys}\n")

        infos.append(
            {"step": step, "x": x, "ys": ys, "new_ys": new_ys, "values": values, "select_new_ys": select_new_ys}
        )
        ys = select_new_ys

    if to_print:
        print(ys)
    return ys, {"steps": infos}


def naive_solve(args, task, idx, to_print=True, model=None):
    if model is None:
        global gpt
        gpt = partial(gpt, model=args.localbackend, temperature=args.temperature)
    print(gpt)
    x = task.get_input(idx)  # input
    ys = get_samples(args, task, x, "", args.n_generate_sample, args.prompt_sample, stop=None)
    return ys, {}


def naive_assign_task(model_list: List, ys: List) -> List[TaskAssignment]:
    """
    Distributes output candidates evenly among a list of clients for both solving and evaluation.

    Args:
        model_list (List[str]): List of client names to which tasks will be assigned.
        ys (List[str]): List of output candidates to be distributed.

    Returns:
        List[TaskAssignment]: A list of dictionaries, each containing:
            - "solve_client": The name of the client assigned to generate outputs.
            - "eval_client": The name of the client assigned to evaluate outputs (same as "solve_client").
            - "ys": The subset of output candidates assigned to this client.

    Notes:
        - Each client receives an approximately equal share of the output candidates.
        - The same client is used for both solving and evaluation.
    """
    if not model_list:
        return []

    # Initialize assignments for each client
    assignments = []
    for client_name in model_list:
        assignments.append(
            TaskAssignment(
                solve_client=client_name, eval_client=client_name, ys=[]  # Same client for both solve and eval
            )
        )

    # Distribute ys evenly among clients using round-robin assignment
    for i, y in enumerate(ys):
        client_idx = i % len(model_list)
        assignments[client_idx]["ys"].append(y)

    return assignments


def speculative_federated_assign_task(model_list: List, ys: List) -> List[TaskAssignment]:
    if not model_list:
        return []

    # Initialize assignments for each client
    assignments = []
    for client_name in model_list:
        assignments.append(
            TaskAssignment(
                solve_client=client_name, eval_client="remote_client", ys=[]  # Same client for both solve and eval
            )
        )

    # Distribute ys evenly among clients using round-robin assignment
    for i, y in enumerate(ys):
        client_idx = i % len(model_list)
        assignments[client_idx]["ys"].append(y)

    return assignments


class ToTMethods:
    def __init__(self, args):
        self.args = args
        model_config = args.model_config
        with open(model_config, "r") as f:
            self.model_config = json.load(f)
        for model in self.model_config:
            print(f"model: {model}")
            if "api_key_environment_variable" in model.keys():
                env_var = model["api_key_environment_variable"]
                model["api_key"] = os.environ.get(env_var, "")
            elif "api_key" not in model.keys():
                model["api_key"] = os.environ.get("OPENAI_API_KEY", "")
            else:
                model["api_key"] = ""
                print(f"Warning: No API key found for {model['client_name']}, using empty string.")
            if "api_base" not in model.keys():
                model["api_base"] = "https://try-chatapi.com/v1"
                print(f"Warning: No API base found for {model['client_name']}, using default base.")
        self.gpts = {
            model["client_name"]: partial(
                gpt,
                model=model["model"],
                temperature=args.temperature,
                api_base=model["api_base"],
                api_key=model["api_key"],
            )
            for model in self.model_config
        }

    def naive_solve(
        self, task, idx, to_print=True, solve_client="remote_client", **kwargs
    ) -> Tuple[List[str], Dict[str, List[Dict[str, str]]]]:
        gpt = self.gpts[solve_client]
        print(gpt)
        x = task.get_input(idx)  # input
        ys = get_samples(self.args, task, x, "", self.args.n_generate_sample, self.args.prompt_sample, stop=None)
        return ys, {}

    def solve(
        self, task, idx, to_print=True, solve_client="remote_client", **kwargs
    ) -> Tuple[List[str], Dict[str, List[Dict[str, str]]]]:
        gpt = self.gpts[solve_client]
        print(gpt)
        x = task.get_input(idx)  # input
        print(f"task: {task}, type: {type(task)}")
        ys = [""]  # current output candidates
        infos = []
        for step in range(task.steps):
            # generation
            if self.args.method_generate == "sample":
                new_ys = [
                    get_samples(
                        self.args,
                        task,
                        x,
                        y,
                        self.args.n_generate_sample,
                        prompt_sample=self.args.prompt_sample,
                        stop=task.stops[step],
                        client=gpt,
                    )
                    for y in ys
                ]
            elif self.args.method_generate == "propose":
                new_ys = [get_proposals(self.args, step, task, x, y, client=gpt) for y in ys]
            new_ys = list(itertools.chain(*new_ys))
            ids = list(range(len(new_ys)))
            # evaluation
            if self.args.method_evaluate == "vote":
                values = get_votes(self.args, task, x, new_ys, self.args.n_evaluate_sample, client=gpt)
            elif self.args.method_evaluate == "value":
                values = get_values(self.args, task, x, new_ys, self.args.n_evaluate_sample, client=gpt)

            # selection
            if self.args.method_select == "sample":
                ps = np.array(values) / sum(values)
                select_ids = np.random.choice(ids, size=self.args.n_select_sample, p=ps).tolist()
            elif self.args.method_select == "greedy":
                select_ids = sorted(ids, key=lambda x: values[x], reverse=True)[: self.args.n_select_sample]
            select_new_ys = [new_ys[select_id] for select_id in select_ids]

            # log
            if to_print:
                sorted_new_ys, sorted_values = zip(*sorted(zip(new_ys, values), key=lambda x: x[1], reverse=True))
                print(
                    f"-- new_ys --: {sorted_new_ys}\n-- sol values --: {sorted_values}\n-- choices --: {select_new_ys}\n"
                )

            infos.append(
                {"step": step, "x": x, "ys": ys, "new_ys": new_ys, "values": values, "select_new_ys": select_new_ys}
            )
            ys = select_new_ys

        if to_print:
            print(ys)
        return ys, {"steps": infos}

    def speculative_solve(
        self, task, idx, to_print=True, solve_client=None, eval_client=None, **kwargs
    ) -> Tuple[List[str], Dict[str, List[Dict[str, str]]]]:
        # Output the imformation of models called
        local_client = self.gpts["solve_client"] if solve_client is None else self.gpts[solve_client]
        remote_client = self.gpts["eval_client"] if eval_client is None else self.gpts[eval_client]

        x = task.get_input(idx)  # input
        ys = [""]  # current output candidates
        infos = []

        for step in range(task.steps):
            # Claim the propose model and value model
            solve_client = local_client
            eval_client = remote_client
            # choose a propose model
            if self.args.warm_start == True and step == 0:
                solve_client = remote_client
                eval_client = remote_client
            elif self.args.slm_generate == False or step + 1 == task.steps and self.args.last_lm:
                solve_client = remote_client
                eval_client = remote_client
            # choose a value model
            eval_client = remote_client
            if self.args.slm_eval:
                eval_client = local_client

            # generation
            if self.args.method_generate == "sample":  # large model for sample
                new_ys = [
                    get_samples(
                        self.args,
                        task,
                        x,
                        y,
                        self.args.n_generate_sample,
                        prompt_sample=self.args.prompt_sample,
                        stop=task.stops[step],
                        client=solve_client,
                    )
                    for y in ys
                ]
            elif self.args.method_generate == "propose":  # large model for propose
                new_ys = [get_proposals(self.args, step, task, x, y, client=solve_client) for y in ys]
            else:
                raise Exception("Not match!")
            new_ys = list(itertools.chain(*new_ys))
            ids = list(range(len(new_ys)))

            # evaluation
            if self.args.method_evaluate == "vote":
                values = get_votes(self.args, task, x, new_ys, self.args.n_evaluate_sample, client=eval_client)
            elif self.args.method_evaluate == "value":
                values = get_values(self.args, task, x, new_ys, self.args.n_evaluate_sample, client=eval_client)
            else:
                raise Exception("Not match!")

            # selection
            if self.args.method_select == "sample":
                ps = np.array(values) / sum(values)
                select_ids = np.random.choice(ids, size=self.args.n_select_sample, p=ps).tolist()
            elif self.args.method_select == "greedy":
                select_ids = sorted(ids, key=lambda x: values[x], reverse=True)[: self.args.n_select_sample]
            select_new_ys = [new_ys[select_id] for select_id in select_ids]

            # log
            if to_print:
                sorted_new_ys, sorted_values = zip(*sorted(zip(new_ys, values), key=lambda x: x[1], reverse=True))
                print(
                    f"-- new_ys --: {sorted_new_ys}\n-- sol values --: {sorted_values}\n-- choices --: {select_new_ys}\n"
                )

            infos.append(
                {
                    "step": step,
                    "x": x,
                    "ys": ys,
                    "new_ys": new_ys,
                    "values": values,
                    "select_new_ys": select_new_ys,
                }
            )
            ys = select_new_ys

        if to_print:
            print(ys)
        return ys, {"steps": infos}

    def federated_solve(
        self,
        task,
        idx,
        to_print=True,
        assign_function: Callable[[List[str], List[str]], List[TaskAssignment]] = speculative_federated_assign_task,
        **kwargs,
    ) -> Tuple[List[str], Dict[str, List[Dict[str, str]]]]:
        """
        Federated solve using multiple clients
        """
        x = task.get_input(idx)  # input
        ys = [""]  # current output candidates
        infos = []

        # Get available clients
        client_names = list(self.gpts.keys())
        if not client_names:
            raise ValueError("No clients available for federated solving")

        for step in range(task.steps):
            if to_print:
                print(f"Step {step} of {task.steps} in Task {idx}")

            # Assign tasks to clients
            task_assignments = assign_function(client_names, ys)

            # Generate proposals from each client
            all_new_ys = []
            all_values = []
            step_infos = []

            for assignment in task_assignments:
                if assignment["ys"]:  # Only process if there are ys assigned
                    if to_print:
                        print(
                            f"Client {assignment['solve_client']} (eval: {assignment['eval_client']}) processing: {assignment['ys']}"
                        )

                    # Generate proposals for this client
                    client_new_ys, client_values, client_info = self._client_step(
                        task,
                        x,
                        assignment["ys"],
                        step,
                        assignment["solve_client"],
                        assignment["eval_client"],
                        to_print,
                        n_generate_sample=ceil(
                            self.args.n_generate_sample / len([a for a in task_assignments if a["ys"]])
                        ),
                    )

                    all_new_ys.extend(client_new_ys)
                    all_values.extend(client_values)
                    step_infos.append(client_info)

            # If no new proposals generated, keep current ys
            if not all_new_ys:
                all_new_ys = ys.copy()
                all_values = [1.0] * len(ys)  # Default values

            # Selection from all proposals
            ids = list(range(len(all_new_ys)))
            if self.args.method_select == "sample" and sum(all_values) > 0:
                ps = np.array(all_values) / sum(all_values)
                select_ids = np.random.choice(ids, size=min(self.args.n_select_sample, len(ids)), p=ps).tolist()
            elif self.args.method_select == "greedy":
                select_ids = sorted(ids, key=lambda x: all_values[x], reverse=True)[: self.args.n_select_sample]
            else:
                select_ids = ids[: self.args.n_select_sample]

            select_new_ys = [all_new_ys[select_id] for select_id in select_ids]

            # Log
            if to_print and all_new_ys and all_values:
                sorted_new_ys, sorted_values = zip(
                    *sorted(zip(all_new_ys, all_values), key=lambda x: x[1], reverse=True)
                )
                print(
                    f"-- new_ys --: {sorted_new_ys[:5]}\n-- sol values --: {sorted_values[:5]}\n-- choices --: {select_new_ys}\n"
                )

            infos.append(
                {
                    "step": step,
                    "x": x,
                    "ys": ys,
                    "new_ys": all_new_ys,
                    "values": all_values,
                    "select_new_ys": select_new_ys,
                    "client_infos": step_infos,
                    "task_assignments": task_assignments,
                }
            )

            ys = select_new_ys

        if to_print:
            print(f"Final results: {ys}")

        return ys, {"steps": infos}

    def _client_step(
        self,
        task,
        x,
        client_ys,
        step,
        solve_client_name,
        eval_client_name,
        to_print=False,
        n_generate_sample=int | None,
    ):
        """
        Execute one step for a specific client

        Args:
            task: The task object
            x: Input string
            client_ys: List of candidate strings assigned to this client
            step: Current step number
            solve_client_name: Name of the client for generation
            eval_client_name: Name of the client for evaluation
            to_print: Whether to print debug information
            n_generate_sample: Number of samples to generate
        """
        solve_gpt = self.gpts[solve_client_name]
        eval_gpt = self.gpts[eval_client_name]
        new_ys = []
        n_generate_sample = n_generate_sample if n_generate_sample else self.args.n_generate_sample

        # Generation
        for y in client_ys:
            if self.args.method_generate == "sample":
                samples = get_samples(
                    self.args,
                    task,
                    x,
                    y,
                    n_generate_sample=n_generate_sample,
                    prompt_sample=self.args.prompt_sample,
                    stop=task.stops[step],
                    client=solve_gpt,
                )
                new_ys.extend(samples)
            elif self.args.method_generate == "propose":
                proposals = get_proposals(self.args, step, task, x, y, client=solve_gpt)
                new_ys.extend(proposals)

        # Remove duplicates while preserving order
        unique_new_ys = []
        seen = set()
        for y in new_ys:
            if y not in seen:
                unique_new_ys.append(y)
                seen.add(y)
        new_ys = unique_new_ys

        # Evaluation
        if new_ys:
            if self.args.method_evaluate == "vote":
                values = get_votes(self.args, task, x, new_ys, self.args.n_evaluate_sample, client=eval_gpt)
            elif self.args.method_evaluate == "value":
                values = get_values(self.args, task, x, new_ys, self.args.n_evaluate_sample, client=eval_gpt)
            else:
                values = [1.0] * len(new_ys)  # Default values
        else:
            values = []

        client_info = {
            "solve_client_name": solve_client_name,
            "eval_client_name": eval_client_name,
            "input_ys": client_ys,
            "output_ys": new_ys,
            "values": values,
            "step": step,
        }

        if to_print:
            print(
                f"Client {solve_client_name} (eval: {eval_client_name}) generated {len(new_ys)} proposals with values {values[:5]}"
            )

        return new_ys, values, client_info
