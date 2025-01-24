import itertools
import numpy as np
from functools import partial
from tot.models import gpt
import os, re
from tot.pattern_match import check_and_fix_last_line
import time

LLM_completion_token = 0
LLM_prompt_token = 0
SLM_completion_token = 0
SLM_prompt_token = 0


local_api_key, local_api_base = (
    "lm-studio",
    "http://127.0.0.1:11451/v1",
)  # "meta-llama-3.1-8b-instruct@q4_k_m" #"http://127.0.0.1:11451/v1","bartowski/Phi-3-medium-128k-instruct-GGUF"
cloud_api_key, cloud_api_base = (
    "lm-studio",
    "http://158.132.255.40:1234/v1",
)  # "Qwen/Qwen2.5-32B-Instruct-GGUF" #"bartowski/Phi-3-medium-128k-instruct-GGUF"
openai_api_key, openai_api_base, openai_model = os.environ.get("OPENAI_API_KEY"), "https://try-chatapi.com/v1", "gpt-4o"


def get_value(args, task, x, y, n_evaluate_sample, cache_value=True, api_key=None, api_base=None, model=None):
    value_prompt = task.value_prompt_wrap(x, y)
    if cache_value and value_prompt in task.value_cache:
        return task.value_cache[value_prompt]
    value_outputs = gpt(args, value_prompt, n=n_evaluate_sample, stop=None, api_key=api_key, api_base=api_base, model=model)  # type: ignore
    value = task.value_outputs_unwrap(x, y, value_outputs)
    if cache_value:
        task.value_cache[value_prompt] = value
    return value


def get_values(args, task, x, ys, n_evaluate_sample, cache_value=True, api_key=None, api_base=None, model=None):
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
                    )
                    count += 1
            local_value_cache[y] = value
        values.append(value)
    return values


def get_votes(args, task, x, ys, n_evaluate_sample, api_key=None, api_base=None, model=None):
    vote_prompt = task.vote_prompt_wrap(x, ys)
    vote_outputs = gpt(args, vote_prompt, n=n_evaluate_sample, stop=None, api_key=api_key, api_base=api_base, model=model)  # type: ignore
    values = task.vote_outputs_unwrap(vote_outputs, len(ys))
    return values


def get_proposals(args, step, task, x, y, api_key=None, api_base=None, model=None):
    # jinyu:
    need_generate = task.pre_generate_check(y)
    if need_generate == False:  # no need to generate new proposals
        return [y]

    new_proposal_list, run_times = [], 0
    time_constraint, len_constraint = 3, 4

    while len(new_proposal_list) < len_constraint and run_times < time_constraint:  # Generate at least 4 proposals

        propose_prompt = task.propose_prompt_wrap(x, y)
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


def get_samples(args, task, x, y, n_generate_sample, prompt_sample, stop, api_key=None, api_base=None, model=None):
    if prompt_sample == "standard":
        prompt = task.standard_prompt_wrap(x, y)
    elif prompt_sample == "cot":
        prompt = task.cot_prompt_wrap(x, y)
    else:
        raise ValueError(f"prompt_sample {prompt_sample} not recognized")
    samples = gpt(args, prompt, n=n_generate_sample, stop=stop, api_key=api_key, api_base=api_base, model=model)  # type: ignore
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


def solve_usingLLM_eval(args, task, idx, to_print=True):
    global gpt
    # Output the imformation of models called
    gpt = partial(gpt, model=args.localbackend, temperature=args.temperature)
    global gpt_evaluator
    gpt_evaluator = partial(
        gpt,
        model="gpt-4o",
        temperature=args.temperature,
        api_base="https://try-chatapi.com/v1",
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    print(gpt)
    print(gpt_evaluator)
    x = task.get_input(idx)  # input
    ys = [""]  # current output candidates
    infos = []
    local_model = args.localbackend
    cloud_model = args.remotebackend

    lat_all, lat_generate, lat_eval, lat_select = [], [], [], []

    for step in range(task.steps):
        step_start_time = time.time()

        # Claim the propose model and value model
        propose_key, propose_base, propose_model = local_api_key, local_api_base, local_model
        # choose a propose model
        if args.warm_start == True and step == 0:
            propose_key, propose_base, propose_model = openai_api_key, openai_api_base, openai_model
        elif args.slm_generate == False or step + 1 == task.steps and args.last_lm:
            propose_key, propose_base, propose_model = cloud_api_key, cloud_api_base, cloud_model
        # choose a value model
        value_key, value_base, value_model = local_api_key, local_api_base, local_model
        if args.slm_eval == False:
            value_key, value_base, value_model = cloud_api_key, cloud_api_base, cloud_model

        # generation
        gen_start_time = time.time()
        if args.method_generate == "sample":  # large model for sample
            new_ys = [
                get_samples(
                    args,
                    task,
                    x,
                    y,
                    args.n_generate_sample,
                    prompt_sample=args.prompt_sample,
                    stop=task.stops[step],
                    api_key=propose_key,
                    api_base=propose_base,
                    model=propose_model,
                )
                for y in ys
            ]
        elif args.method_generate == "propose":  # large model for propose
            new_ys = [
                get_proposals(args, step, task, x, y, api_key=propose_key, api_base=propose_base, model=propose_model)
                for y in ys
            ]
        else:
            raise Exception("Not match!")
        new_ys = list(itertools.chain(*new_ys))
        ids = list(range(len(new_ys)))
        gen_end_time = time.time()
        lat_generate.append(gen_end_time - gen_start_time)

        # evaluation
        eval_start_time = time.time()
        if args.method_evaluate == "vote":
            values = get_votes(
                args, task, x, new_ys, args.n_evaluate_sample, api_key=value_key, api_base=value_base, model=value_model
            )
        elif args.method_evaluate == "value":
            values = get_values(
                args, task, x, new_ys, args.n_evaluate_sample, api_key=value_key, api_base=value_base, model=value_model
            )
        else:
            raise Exception("Not match!")
        eval_end_time = time.time()
        lat_eval.append(eval_end_time - eval_start_time)

        # selection
        sel_start_time = time.time()
        if args.method_select == "sample":
            ps = np.array(values) / sum(values)
            select_ids = np.random.choice(ids, size=args.n_select_sample, p=ps).tolist()
        elif args.method_select == "greedy":
            select_ids = sorted(ids, key=lambda x: values[x], reverse=True)[: args.n_select_sample]
        select_new_ys = [new_ys[select_id] for select_id in select_ids]
        sel_end_time = time.time()
        lat_select.append(sel_end_time - sel_start_time)

        # log
        if to_print:
            sorted_new_ys, sorted_values = zip(*sorted(zip(new_ys, values), key=lambda x: x[1], reverse=True))
            print(f"-- new_ys --: {sorted_new_ys}\n-- sol values --: {sorted_values}\n-- choices --: {select_new_ys}\n")

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
        step_end_time = time.time()
        lat_all.append(step_end_time - step_start_time)

    if to_print:
        print(ys)
    lat_dict = {"all": lat_all, "generate": lat_generate, "eval": lat_eval}
    return ys, {"steps": infos}, lat_dict


def naive_solve(args, task, idx, to_print=True):
    global gpt
    gpt = partial(gpt, model=args.localbackend, temperature=args.temperature)
    print(gpt)
    x = task.get_input(idx)  # input
    ys = get_samples(args, task, x, "", args.n_generate_sample, args.prompt_sample, stop=None)
    return ys, {}


def client_solve(
    args,
    task,
    current_task: str,
    ys: list,
    step: int,
    api_base: str,
    api_key: str,
    model: str,
    client_name: str,
    to_print=True,
) -> tuple:
    """
    This function generates the solve for only 1 step based on the middle steps given by ys and the current task.
    ys: list of str which are the selected output candidates, this list is selected by the assign_task function
    current_task: str, the input of the current (original) task
    step: int, the current step of the task
    """
    global gpt
    gpt = partial(gpt, model=model, temperature=args.temperature, api_base=api_base)
    print(gpt)
    x = current_task
    infos = []
    lat_all, lat_generate, lat_eval, lat_select = [], [], [], []
    step = step
    # generation
    gen_start_time = time.time()
    if args.method_generate == "sample":
        new_ys = [
            get_samples(
                args,
                task,
                x,
                y,
                args.n_generate_sample,
                prompt_sample=args.prompt_sample,
                stop=None,
                api_base=api_base,
                api_key=api_key,
                model=model,
            )
            for y in ys
        ]
    elif args.method_generate == "propose":
        new_ys = [get_proposals(args, 0, task, x, y, api_key=api_key, api_base=api_base, model=model) for y in ys]
    else:
        raise Exception("Not match!")
    new_ys = list(itertools.chain(*new_ys))
    ids = list(range(len(new_ys)))
    gen_end_time = time.time()
    lat_generate.append(gen_end_time - gen_start_time)
    # evaluation
    eval_start_time = time.time()
    if args.method_evaluate == "vote":
        values = get_votes(
            args, task, x, new_ys, args.n_evaluate_sample, api_key=api_key, api_base=api_base, model=model
        )
    elif args.method_evaluate == "value":
        values = get_values(
            args, task, x, new_ys, args.n_evaluate_sample, api_key=api_key, api_base=api_base, model=model
        )
    else:
        raise Exception("Not match!")
    eval_end_time = time.time()
    lat_eval.append(eval_end_time - eval_start_time)

    # selection
    sel_start_time = time.time()
    if args.method_select == "sample":
        ps = np.array(values) / sum(values)
        select_ids = np.random.choice(ids, size=args.n_select_sample, p=ps).tolist()
    elif args.method_select == "greedy":
        select_ids = sorted(ids, key=lambda x: values[x], reverse=True)[: args.n_select_sample]
    select_new_ys = [new_ys[select_id] for select_id in select_ids]
    sel_end_time = time.time()
    lat_select.append(sel_end_time - sel_start_time)

    if to_print:
        sorted_new_ys, sorted_values = zip(*sorted(zip(new_ys, values), key=lambda x: x[1], reverse=True))
        print(f"-- new_ys --: {sorted_new_ys}\n-- sol values --: {sorted_values}\n-- choices --: {select_new_ys}\n")
    infos.append(
        {
            "client_name": client_name,
            "api_base": api_base,
            "model": model,
            "step": step,
            "x": x,
            "ys": ys,
            "new_ys": new_ys,
            "values": values,
            "select_new_ys": select_new_ys,
        }
    )
    ys = select_new_ys

    all_end_time = time.time()
    lat_all.append(all_end_time - gen_start_time)
    lat_dict = {"all": lat_all, "generate": lat_generate, "eval": lat_eval}
    return ys, {"steps": infos}, lat_dict


def assign_task(model_list, ys, **kwargs):
    """
    input: ys: list of str which are the selected output candidates
    **kwargs: a dict of the dicts of api_base, model, api_key, client_name
    output: a list, like that
    [[task1, task2, task3], [task4, task5, task6], [task7, task8, task9]]
    tasks are assigned to different clients with the same index in the model_list
    """
    # 1:1 assignment
    task_list = []
    for i in range(len(model_list)):
        task_list.append([])
    for i in range(len(ys)):
        task_list[i % len(model_list)].append(ys[i])
    return task_list


def thread_solve(args, task, idx, to_print=True, **kwargs):
    """
    kwargs：input：
    "api_base" = base, "model" = model, "api_key" = key, "client_name" = client_name
    """
    # Initialize model
    api_base = kwargs["api_base"]
    model = kwargs["model"]
    api_key = kwargs["api_key"]
    client_name = kwargs["client_name"]
    global gpt
    gpt = partial(gpt, model=model, temperature=args.temperature, api_base=api_base)
    print(gpt)
    x = task.get_input(idx)  # input
    ys = [""]  # current output candidates
    infos = []
    lat_all, lat_generate, lat_eval, lat_select = [], [], [], []
    for step in range(task.steps):
        step_start_time = time.time()

        # generation
        gen_start_time = time.time()
        if args.method_generate == "sample":
            new_ys = [
                get_samples(
                    args,
                    task,
                    x,
                    y,
                    args.n_generate_sample,
                    args.prompt_sample,
                    stop=None,
                    api_base=api_base,
                    api_key=api_key,
                    model=model,
                )
                for y in ys
            ]
        elif args.method_generate == "propose":
            new_ys = [
                get_proposals(args, step, task, x, y, api_key=api_key, api_base=api_base, model=model) for y in ys
            ]
        else:
            raise Exception("Not match!")
        new_ys = list(itertools.chain(*new_ys))
        ids = list(range(len(new_ys)))
        gen_end_time = time.time()
        lat_generate.append(gen_end_time - gen_start_time)

        # evaluation
        eval_start_time = time.time()
        if args.method_evaluate == "vote":
            values = get_votes(
                args, task, x, new_ys, args.n_evaluate_sample, api_key=api_key, api_base=api_base, model=model
            )
        elif args.method_evaluate == "value":
            values = get_values(
                args, task, x, new_ys, args.n_evaluate_sample, api_key=api_key, api_base=api_base, model=model
            )
        else:
            raise Exception("Not match!")
        eval_end_time = time.time()
        lat_eval.append(eval_end_time - eval_start_time)

        # selection
        sel_start_time = time.time()
        if args.method_select == "sample":
            ps = np.array(values) / sum(values)
            select_ids = np.random.choice(ids, size=args.n_select_sample, p=ps).tolist()
        elif args.method_select == "greedy":
            select_ids = sorted(ids, key=lambda x: values[x], reverse=True)[: args.n_select_sample]
        select_new_ys = [new_ys[select_id] for select_id in select_ids]
        sel_end_time = time.time()
        lat_select.append(sel_end_time - sel_start_time)

        if to_print:
            sorted_new_ys, sorted_values = zip(*sorted(zip(new_ys, values), key=lambda x: x[1], reverse=True))
            print(f"-- new_ys --: {sorted_new_ys}\n-- sol values --: {sorted_values}\n-- choices --: {select_new_ys}\n")
        infos.append(
            {
                "client_name": client_name,
                "api_base": api_base,
                "model": model,
                "step": step,
                "x": x,
                "ys": ys,
                "new_ys": new_ys,
                "values": values,
                "select_new_ys": select_new_ys,
            }
        )
        ys = select_new_ys
        step_end_time = time.time()
        lat_all.append(step_end_time - step_start_time)
    if to_print:
        print(ys)
    lat_dict = {"all": lat_all, "generate": lat_generate, "eval": lat_eval}
    return ys, {"steps": infos}, lat_dict
