import itertools
import numpy as np
from functools import partial
from tot.models import gpt
import os, re
from tot.pattern_match import check_and_fix_last_line

LLM_completion_token = 0
LLM_prompt_token = 0
SLM_completion_token = 0
SLM_prompt_token = 0

"""
The part "*_usingLLM" of function name indicates this method using LLM to process tasks.
This naming method comes from that we use SLM to run tasks.
The changes are used to let the LLM process **evaluation** part 
as well as other parts are still handled by SLM.

函数名中的 "*_usingLLM" 表示的是用大模型参与处理任务的方法.
"""

pattern = [
    "(left: 1 24)",
    "(left: 2 12)",
    "(left: 3 8)",
    "(left: 4 6)",
    "(left: 20 4)",
    "(left: 30 6)",
    "(left: 12 12)",
]


def get_value(task, x, y, n_evaluate_sample, cache_value=True):
    value_prompt = task.value_prompt_wrap(x, y)
    if cache_value and value_prompt in task.value_cache:
        return task.value_cache[value_prompt]
    value_outputs = gpt(value_prompt, n=n_evaluate_sample, stop=None)
    value = task.value_outputs_unwrap(x, y, value_outputs)
    if cache_value:
        task.value_cache[value_prompt] = value
    return value


def get_value_usingLLM(task, x, y, n_evaluate_sample, cache_value=True):
    """
    Retrieves a value using gpt-4o instead of SLM based on the provided task and inputs.

    Returns:
        Any: The value obtained from the LLM based on the task and inputs.
    """
    value_prompt = task.value_prompt_wrap(x, y)
    if cache_value and value_prompt in task.value_cache:
        return task.value_cache[value_prompt]
    value_outputs = gpt(
        value_prompt,
        n=n_evaluate_sample,
        stop=None,
        model="gpt-4o",
        api_base="https://try-chatgpt.fun/v1",
        api_key=os.getenv("OPENAI_API_KEY", ""),
    )
    value = task.value_outputs_unwrap(x, y, value_outputs)
    if cache_value:
        task.value_cache[value_prompt] = value
    return value


def get_values(task, x, ys, n_evaluate_sample, cache_value=True):
    values = []
    local_value_cache = {}
    for y in ys:  # each partial output
        if y in local_value_cache:  # avoid duplicate candidates
            value = 0
        else:
            # jinyu
            value = 0
            if "(left: 24)" in y:
                value = 20 + 1
            else:
                for pat in pattern:
                    if pat in y:
                        value = 20
                        break
            if value == 0:
                value = get_value(
                    task, x, y, n_evaluate_sample, cache_value=cache_value
                )
            local_value_cache[y] = value
        values.append(value)
    return values


def get_values_usingLLM(task, x, ys, n_evaluate_sample, cache_value=True):
    values = []
    local_value_cache = {}
    for y in ys:  # each partial output
        if y in local_value_cache:  # avoid duplicate candidates
            value = 0
        else:
            # jinyu
            value = 0
            if "(left: 24)" in y:
                value = 20 + 1
            else:
                for pat in pattern:
                    if pat in y:
                        value = 20
                        break
            if value == 0:
                value = get_value_usingLLM(
                    task, x, y, n_evaluate_sample, cache_value=cache_value
                )

            local_value_cache[y] = value
        values.append(value)
    return values


def get_votes(task, x, ys, n_evaluate_sample):
    vote_prompt = task.vote_prompt_wrap(x, ys)
    vote_outputs = gpt(vote_prompt, n=n_evaluate_sample, stop=None)
    values = task.vote_outputs_unwrap(vote_outputs, len(ys))
    return values


def get_votes_usingLLM(task, x, ys, n_evaluate_sample):
    vote_prompt = task.vote_prompt_wrap(x, ys)
    vote_outputs = gpt(
        vote_prompt,
        n=n_evaluate_sample,
        stop=None,
        model="gpt-4o",
        api_base="https://try-chatgpt.fun/v1",
        api_key=os.getenv("OPENAI_API_KEY", ""),
    )
    values = task.vote_outputs_unwrap(vote_outputs, len(ys))
    return values


def get_proposals(task, x, y):
    # jinyu:
    if "(left: 24)" in y:  # no need to generate new proposals
        return [y]
    new_proposal_list, run_times, time_constraint = [], 0, 6

    while (
        len(new_proposal_list) < 4 and run_times < time_constraint
    ):  # Generate at least 4 proposes

        last_prompt, propose_prompt = task.propose_prompt_wrap(x, y)
        proposals = gpt(
            propose_prompt,
            n=1,
            stop=None,
            api_key="lm-studio",
            api_base="http://127.0.0.1:11451/v1",
            model="bartowski/Phi-3-medium-128k-instruct-GGUF",
        )[0].split("\n")
        # return [y + _ + '\n' for _ in proposals]

        # jinyu: check the format
        for pro in proposals:
            if pro.strip() == "":
                continue
            pro = pro.strip()
            pro = re.sub(r"^[^0-9]+|[^0-9)]+$", "", pro)
            new_proposal = y + pro + "\n"
            if last_prompt:
                is_correct, updated_new_proposal = True, new_proposal
            else:
                is_correct, updated_new_proposal = check_and_fix_last_line(
                    new_proposal, x
                )
            if is_correct:  # or we can directly set Ture
                if updated_new_proposal not in new_proposal_list:
                    new_proposal_list.append(updated_new_proposal)
        if last_prompt:
            break
        run_times += 1

    if run_times >= time_constraint:
        print("runtime ", run_times)
    if len(new_proposal_list) == 0:
        return [y]
    return new_proposal_list


def get_proposals_usingLLM(task, x, y):
    if "(left: 24)" in y:
        return [y]

    last_prompt, propose_prompt = task.propose_prompt_wrap(x, y)
    proposals = gpt(
        propose_prompt,
        n=1,
        stop=None,
        model="gpt-4o",
        api_base="https://try-chatgpt.fun/v1",
        api_key=os.environ.get("OPENAI_API_KEY"),
    )[0].split("\n")
    # return [y + _ + '\n' for _ in proposals]

    # jinyu: Check results
    new_proposal_list = []
    for pro in proposals:
        pro = pro.strip()
        pro = re.sub(r"^[^0-9]+|[^0-9)]+$", "", pro)
        new_proposal = y + pro + "\n"
        if last_prompt:
            is_correct, updated_new_proposal = True, new_proposal
        else:
            is_correct, updated_new_proposal = check_and_fix_last_line(new_proposal, x)
        if is_correct:  # or we can directly set Ture
            if updated_new_proposal not in new_proposal_list:
                new_proposal_list.append(updated_new_proposal)
        if last_prompt:
            break
    return new_proposal_list


def get_samples(task, x, y, n_generate_sample, prompt_sample, stop):
    if prompt_sample == "standard":
        prompt = task.standard_prompt_wrap(x, y)
    elif prompt_sample == "cot":
        prompt = task.cot_prompt_wrap(x, y)
    else:
        raise ValueError(f"prompt_sample {prompt_sample} not recognized")
    samples = gpt(prompt, n=n_generate_sample, stop=stop)
    return [y + _ for _ in samples]


def get_samples_usingLLM(task, x, y, n_generate_sample, prompt_sample, stop):
    if prompt_sample == "standard":
        prompt = task.standard_prompt_wrap(x, y)
    elif prompt_sample == "cot":
        prompt = task.cot_prompt_wrap(x, y)
    else:
        raise ValueError(f"prompt_sample {prompt_sample} not recognized")
    samples = gpt(
        prompt,
        n=n_generate_sample,
        stop=stop,
        model="gpt-4o",
        api_base="https://try-chatgpt.fun/v1",
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    return [y + _ for _ in samples]


def solve(args, task, idx, to_print=True):
    global gpt
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    print(gpt)
    x = task.get_input(idx)  # input
    ys = [""]  # current output candidates
    infos = []
    for step in range(task.steps):
        # generation

        # 我们希望可以使用本地的小模型进行生成, 但是评估部分使用大模型. 同时，尝试使用大模型完全接管最后几步.
        if args.method_generate == "sample" and step < task.steps - 2:
            new_ys = [
                get_samples(
                    task,
                    x,
                    y,
                    args.n_generate_sample,
                    prompt_sample=args.prompt_sample,
                    stop=task.stops[step],
                )
                for y in ys
            ]
        elif args.method_generate == "propose" and True:
            new_ys = [get_proposals(task, x, y) for y in ys]
        elif args.method_generate == "sample" and step >= task.steps - 2:
            new_ys = [
                get_samples_usingLLM(
                    task,
                    x,
                    y,
                    args.n_generate_sample,
                    prompt_sample=args.prompt_sample,
                    stop=task.stops[step],
                )
                for y in ys
            ]
        elif args.method_generate == "propose" and False:
            new_ys = [get_proposals_usingLLM(task, x, y) for y in ys]
        new_ys = list(itertools.chain(*new_ys))
        ids = list(range(len(new_ys)))
        # evaluation
        if args.method_evaluate == "vote":
            values = get_votes(task, x, new_ys, args.n_evaluate_sample)
        elif args.method_evaluate == "value":
            values = get_values(task, x, new_ys, args.n_evaluate_sample)
        # selection
        if args.method_select == "sample":
            ps = np.array(values) / sum(values)
            select_ids = np.random.choice(ids, size=args.n_select_sample, p=ps).tolist()
        elif args.method_select == "greedy":
            select_ids = sorted(ids, key=lambda x: values[x], reverse=True)[
                : args.n_select_sample
            ]
        select_new_ys = [new_ys[select_id] for select_id in select_ids]

        # log
        if to_print:
            sorted_new_ys, sorted_values = zip(
                *sorted(zip(new_ys, values), key=lambda x: x[1], reverse=True)
            )
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


def solve_usingLLM_eval(args, task, idx, to_print=True):
    """
    This function is used to let LLM process the evaluation part,
    as well as SLM process other parts.
    """
    global gpt
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    global gpt_evaluator
    gpt_evaluator = partial(
        gpt,
        model="gpt-4o",
        temperature=args.temperature,
        api_base="https://try-chatgpt.fun/v1",
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    print(gpt)
    print(gpt_evaluator)
    x = task.get_input(idx)  # input
    ys = [""]  # current output candidates
    infos = []
    for step in range(task.steps):
        # generation
        if args.method_generate == "sample" and step != 0:
            new_ys = [
                get_samples(
                    task,
                    x,
                    y,
                    args.n_generate_sample,
                    prompt_sample=args.prompt_sample,
                    stop=task.stops[step],
                )
                for y in ys
            ]
        elif args.method_generate == "propose" and step != 0:  # step != 0
            new_ys = [get_proposals(task, x, y) for y in ys]
        elif args.method_generate == "sample" and step == 0:
            new_ys = [
                get_samples_usingLLM(
                    task,
                    x,
                    y,
                    args.n_generate_sample,
                    prompt_sample=args.prompt_sample,
                    stop=task.stops[step],
                )
                for y in ys
            ]
        elif args.method_generate == "propose" and step == 0:  # step == 0
            new_ys = [get_proposals_usingLLM(task, x, y) for y in ys]
        new_ys = list(itertools.chain(*new_ys))
        ids = list(range(len(new_ys)))
        # evaluation
        if args.method_evaluate == "vote":
            values = get_votes_usingLLM(task, x, new_ys, args.n_evaluate_sample)
        elif args.method_evaluate == "value":
            values = get_values_usingLLM(task, x, new_ys, args.n_evaluate_sample)
        # selection
        if args.method_select == "sample":
            ps = np.array(values) / sum(values)
            select_ids = np.random.choice(ids, size=args.n_select_sample, p=ps).tolist()
        elif args.method_select == "greedy":
            select_ids = sorted(ids, key=lambda x: values[x], reverse=True)[
                : args.n_select_sample
            ]
        select_new_ys = [new_ys[select_id] for select_id in select_ids]
        # log
        if to_print:
            sorted_new_ys, sorted_values = zip(
                *sorted(zip(new_ys, values), key=lambda x: x[1], reverse=True)
            )
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
        end = False
        for i in ys:
            if "(left: 24)" in i:
                end = True
                break
        # if(end):
        #     break
    if to_print:
        print(ys)
    return ys, {"steps": infos}


def naive_solve(args, task, idx, to_print=True):
    global gpt
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    print(gpt)
    x = task.get_input(idx)  # input
    ys = get_samples(task, x, "", args.n_generate_sample, args.prompt_sample, stop=None)
    return ys, {}
