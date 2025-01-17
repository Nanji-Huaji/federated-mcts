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

"""
The part "*_usingLLM" of function name indicates this method using LLM to process tasks.
This naming method comes from that we use SLM to run tasks.
The changes are used to let the LLM process **evaluation** part 
as well as other parts are still handled by SLM.

函数名中的 "*_usingLLM" 表示的是用大模型参与处理任务的方法.
"""

local_api_key, local_api_base ="lm-studio", "http://127.0.0.1:11451/v1" #"meta-llama-3.1-8b-instruct@q4_k_m" #"http://127.0.0.1:11451/v1","bartowski/Phi-3-medium-128k-instruct-GGUF"
cloud_api_key, cloud_api_base ="lm-studio", "http://158.132.255.40:1234/v1" #"Qwen/Qwen2.5-32B-Instruct-GGUF" #"bartowski/Phi-3-medium-128k-instruct-GGUF"
openai_api_key, openai_api_base, openai_model = os.environ.get("OPENAI_API_KEY"), "https://try-chatapi.com/v1", "gpt-4o"

def get_value(args, task, x, y, n_evaluate_sample, cache_value=True, api_key=None, api_base=None, model=None):
    value_prompt = task.value_prompt_wrap(x, y)
    if cache_value and value_prompt in task.value_cache:
        return task.value_cache[value_prompt]
    value_outputs = gpt(args, value_prompt, n=n_evaluate_sample, stop=None, api_key=api_key, api_base=api_base, model=model)
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
            if value == 0 and final==False:
                count=0
                while(value==0 and count<2):
                    value = get_value(args, task, x, y, n_evaluate_sample, cache_value=cache_value, api_key=api_key, api_base=api_base, model=model)
                    count+=1
            local_value_cache[y] = value
        values.append(value)
    return values


def get_votes(args, task, x, ys, n_evaluate_sample, api_key=None, api_base=None, model=None):
    vote_prompt = task.vote_prompt_wrap(x, ys)
    vote_outputs = gpt(args, vote_prompt, n=n_evaluate_sample, stop=None, api_key=api_key, api_base=api_base, model=model)
    values = task.vote_outputs_unwrap(vote_outputs, len(ys))
    return values

def get_proposals(args, step, task, x, y, api_key=None, api_base=None, model=None):
    # jinyu:
    need_generate = task.pre_generate_check(y)
    if need_generate==False:  # no need to generate new proposals
        return [y]
    
    new_proposal_list, run_times = [], 0
    time_constraint, len_constraint = 3, 4

    while (len(new_proposal_list) < len_constraint and run_times < time_constraint):  # Generate at least 4 proposals

        propose_prompt = task.propose_prompt_wrap(x, y)
        proposals = gpt(
            args,
            propose_prompt,
            n=1,
            stop=None,
            api_key=api_key,
            api_base=api_base,
            model=model,
        )[0].split("\n")

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
    samples = gpt(args, prompt, n=n_generate_sample, stop=stop, api_key=api_key, api_base=api_base, model=model)
    return [y + _ for _ in samples]

def solve_usingLLM_eval(args, task, idx, to_print=True):
    """
    This function is used to let LLM process the evaluation part,
    as well as SLM process other parts.
    """
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
        if(args.warm_start == True and step == 0):
            propose_key, propose_base, propose_model = openai_api_key, openai_api_base, openai_model
        elif(args.slm_generate==False or step+1==task.steps and args.last_lm):
            propose_key, propose_base, propose_model = cloud_api_key, cloud_api_base, cloud_model
        # choose a value model
        value_key, value_base, value_model = local_api_key, local_api_base, local_model
        if(args.slm_eval==False):
            value_key, value_base, value_model = cloud_api_key, cloud_api_base, cloud_model
        

        # generation
        gen_start_time = time.time()
        if args.method_generate == "sample": # large model for sample
            new_ys = [get_samples(args, task, x, y, args.n_generate_sample, prompt_sample=args.prompt_sample, stop=task.stops[step], api_key=propose_key, api_base=propose_base, model=propose_model) for y in ys]
        elif args.method_generate == "propose": # large model for propose
            new_ys = [get_proposals(args, step, task, x, y, api_key=propose_key, api_base=propose_base, model=propose_model) for y in ys]
        else:
            raise Exception("Not match!")
        new_ys = list(itertools.chain(*new_ys))
        ids = list(range(len(new_ys)))
        gen_end_time = time.time()
        lat_generate.append(gen_end_time - gen_start_time)

        # evaluation
        eval_start_time = time.time()
        if args.method_evaluate == "vote":
            values = get_votes(args, task, x, new_ys, args.n_evaluate_sample, api_key=value_key, api_base=value_base, model=value_model)
        elif args.method_evaluate == "value":
            values = get_values(args, task, x, new_ys, args.n_evaluate_sample, api_key=value_key, api_base=value_base, model=value_model)
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
            sorted_new_ys, sorted_values = zip(
                *sorted(zip(new_ys, values), key=lambda x: x[1], reverse=True)
            )
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


def federated_solve(args, task, idx, to_print=True, **kwargs):
    '''
    **kwargs：传入：
    model_1={"api_base": "base_1", "model": "model_1", "api_key"="key_1"},
    model_2={"api_base": "base_2", "model: "model2", "api_key"="key_2"},...
    例如：
    local_model={"api_base": "http://127.0.0.1:11451/v1", "model": "bartowski/Phi-3-medium-128k-instruct-GGUF", "api_key": "lm-studio"},
    remote_model={"api_base": "http://158.132.255.110:1234/v1", "model": "meta-llama-3.1-8b-instruct@q4_k_m", "api_key": "lm-studio"},
    '''
    # Initialize federated models
    global gpts
    global models
    gpts = []
    for key, value in kwargs.items():
        gpts.append(partial(gpt, model=value["model"], temperature=args.temperature, api_base=value["api_base"]))
    print(gpts)
    
    x = task.get_input(idx)  # input
    ys = {}  # 输出候选，格式为{model1: [y1, y2, ...], model2: [y1, y2, ...], ...}
    infos = {} # 信息，格式为{model1: {step: 0, x: x, ys: [], new_ys: [], values: [], select_new_ys: []}, model2: {...}, ...}
    lat_all, lat_generate, lat_eval, lat_select = {}, {}, {}, {}
    for key, value in kwargs.items():
        ys[key] = [""]
        infos[key] = []
        lat_all[key], lat_generate[key], lat_eval[key], lat_select[key] = [], [], [], []
        for step in range(task.steps):
            step_start_time = time.time()
            
            # generation
            gen_start_time = time.time()
            if args.method_generate == "sample":
                new_ys = [get_samples(args, task, x, y, args.n_generate_sample, args.prompt_sample, stop=None,
                api_key=value["api_key"], api_base=value["api_base"], model=value["model"]) for y in ys[key]]
            elif args.method_generate == "propose":
                new_ys = [get_proposals(args, step, task, x, y, 
                api_key=value["api_key"], api_base=value["api_base"], model=value["model"]) for y in ys[key]]
                # 若不调用大模型，api_key可任填，但必须有
            else:
                raise Exception("Not match!")
            new_ys = list(itertools.chain(*new_ys))
            ids = list(range(len(new_ys)))
            gen_end_time = time.time()
            lat_generate[key].append(gen_end_time - gen_start_time)
            
            # evaluation
            eval_start_time = time.time()
            if args.method_evaluate == "vote":
                values = get_votes(args, task, x, new_ys, 
                args.n_evaluate_sample, api_key=value["api_key"], api_base=value["api_base"], model=value["model"])
            elif args.method_evaluate == "value":
                values = get_values(args, task, x, new_ys, 
                args.n_evaluate_sample, api_key=value["api_key"], api_base=value["api_base"], model=value["model"])
            else:
                raise Exception("Not match!")
            eval_end_time = time.time()
            lat_eval[key].append(eval_end_time - eval_start_time)
            
            # selection
            sel_start_time = time.time()
            if args.method_select == "sample":
                ps = np.array(values) / sum(values)
                select_ids = np.random.choice(ids, size=args.n_select_sample, p=ps).tolist()
            elif args.method_select == "greedy":
                select_ids = sorted(ids, key=lambda x: values[x], reverse=True)[: args.n_select_sample]
            select_new_ys = [new_ys[select_id] for select_id in select_ids]
            sel_end_time = time.time()
            lat_select[key].append(sel_end_time - sel_start_time)
            
            # log
            if to_print:
                sorted_new_ys, sorted_values = zip(
                    *sorted(zip(new_ys, values), key=lambda x: x[1], reverse=True)
                )
                print(f"-- new_ys --: {sorted_new_ys}\n-- sol values --: {sorted_values}\n-- choices --: {select_new_ys}\n")
            if key not in infos.keys():
                infos.setdefault(key, {})
            infos[key].append(
                {
                    "model": key,
                    "step": step,
                    "x": x,
                    "ys": ys[key],
                    "new_ys": new_ys,
                    "values": values,
                    "select_new_ys": select_new_ys,
                }
            )
            ys[key] = select_new_ys
        if to_print:
            print(ys[key])
        lat_dict = {"all": lat_all, "generate": lat_generate, "eval": lat_eval} 
    return ys, {"steps": infos}, lat_dict
                        



# ---------------------------------------------------
# if (
        #     args.method_generate == "sample"
        #     and args.slm_generate
        #     and (step != 0 and args.warm_start == True or args.warm_start == False)
        # ): # small model for sample
        #     new_ys = [get_samples(task, x, y, args.n_generate_sample, prompt_sample=args.prompt_sample, stop=task.stops[step], api_key=propose_key, api_base=propose_base, model=propose_model) for y in ys]
        # elif (
        #     args.method_generate == "propose"
        #     and args.slm_generate
        #     and (step != 0 and args.warm_start == True or args.warm_start == False)
        # ): # small model for propose
        #     new_ys = [get_proposals(args, step, task, x, y, api_key=propose_key, api_base=propose_base, model=propose_model) for y in ys]
# if args.method_evaluate == "vote" and args.slm_eval:
        #     values = get_votes(task, x, new_ys, args.n_evaluate_sample, api_key=local_api_key, api_base=local_api_base, model=local_model)
        # elif args.method_evaluate == "value" and args.slm_eval:
        #     values = get_values(args, task, x, new_ys, args.n_evaluate_sample, api_key=local_api_key, api_base=local_api_base, model=local_model)
# ---------------------------------------------------

# ---------------------------------------------------
# def get_votes_usingLLM(task, x, ys, n_evaluate_sample):
#     vote_prompt = task.vote_prompt_wrap(x, ys)
#     vote_outputs = gpt(
#         vote_prompt,
#         n=n_evaluate_sample,
#         stop=None,
#         model="gpt-4o",
#         api_base="https://try-chatapi.com/v1",
#         api_key=os.getenv("OPENAI_API_KEY", ""),
#     )
#     values = task.vote_outputs_unwrap(vote_outputs, len(ys))
#     return values
# ---------------------------------------------------

# ---------------------------------------------------
# def get_samples_usingLLM(task, x, y, n_generate_sample, prompt_sample, stop):
#     if prompt_sample == "standard":
#         prompt = task.standard_prompt_wrap(x, y)
#     elif prompt_sample == "cot":
#         prompt = task.cot_prompt_wrap(x, y)
#     else:
#         raise ValueError(f"prompt_sample {prompt_sample} not recognized")
#     samples = gpt(
#         prompt,
#         n=n_generate_sample,
#         stop=stop,
#         model="gpt-4o",
#         api_base="https://try-chatapi.com/v1",
#         api_key=os.environ.get("OPENAI_API_KEY"),
#     )
#     return [y + _ for _ in samples]
# ---------------------------------------------------

# ---------------------------------------------------
# def get_value_usingLLM(task, x, y, n_evaluate_sample, cache_value=True):
#     """
#     Retrieves a value using gpt-4o instead of SLM based on the provided task and inputs.

#     Returns:
#         Any: The value obtained from the LLM based on the task and inputs.
#     """
#     value_prompt = task.value_prompt_wrap(x, y)
#     if cache_value and value_prompt in task.value_cache:
#         return task.value_cache[value_prompt]
#     value_outputs = gpt(
#         value_prompt,
#         n=n_evaluate_sample,
#         stop=None,
#         model="gpt-4o",
#         api_base="https://try-chatapi.com/v1",
#         api_key=os.getenv("OPENAI_API_KEY", ""),
#     )
#     value = task.value_outputs_unwrap(x, y, value_outputs)
#     if cache_value:
#         task.value_cache[value_prompt] = value
#     return value
# ---------------------------------------------------

# def get_values_usingLLM(args, task, x, ys, n_evaluate_sample, cache_value=True):
#     values = []
#     local_value_cache = {}
#     for y in ys:  # each partial output
#         if y in local_value_cache:  # avoid duplicate candidates
#             value = 0
#         else:
#             # jinyu
#             pattern_final = r"\(left: -?\d+\)"
#             value, final = 0, False
#             if args.eval_rule == True:  # get the value with some rules
#                 if "(left: 24)" in y:
#                     value = 20 + 2
#                 elif re.search(pattern_final, y):
#                     final = True
#                 else:
#                     for pat in pattern:
#                         if pat in y:
#                             value = 20 + 1
#                             break
#             if value == 0 and final==False:
#                 count=0
#                 while(value==0 and count<3):
#                     value = get_value(task, x, y, n_evaluate_sample, cache_value=False, api_key=os.getenv("OPENAI_API_KEY", ""), api_base="https://try-chatapi.com/v1", model="gpt-4o")
#                     count+=1

#             local_value_cache[y] = value
#         values.append(value)
#     return values

# ---------------------------------------------------
# def get_proposals_usingLLM(args, step, task, x, y):
#     need_generate = task.pre_generate_check(y)
#     if need_generate==False:  # no need to generate new proposals
#         return [y]

#     new_proposal_list, run_times = [], 0
#     time_constraint, len_constraint = 3, 4

#     while len(new_proposal_list) < len_constraint and run_times < time_constraint:
#         last_prompt, propose_prompt = task.propose_prompt_wrap(x, y)
#         proposals = gpt(
#             propose_prompt,
#             n=1,
#             stop=None,
#             model="gpt-4o",
#             api_base="https://try-chatapi.com/v1",
#             api_key=os.environ.get("OPENAI_API_KEY"),
#         )[0].split("\n")
#         # return [y + _ + '\n' for _ in proposals]

#         # jinyu: Check results
#         for pro in proposals:
#             is_correct, updated_new_proposal = task.process_generate_result(pro, y, args.check_format)
#             if is_correct:
#                 if updated_new_proposal not in new_proposal_list:
#                     new_proposal_list.append(updated_new_proposal)
#         run_times += 1

#     return new_proposal_list
# ---------------------------------------------------

# ---------------------------------------------------
# def solve(args, task, idx, to_print=True):
#     global gpt
#     gpt = partial(gpt, model=args.localbackend, temperature=args.temperature)
#     print(gpt)
#     x = task.get_input(idx)  # input
#     ys = [""]  # current output candidates
#     infos = []
#     for step in range(task.steps):
#         # generation

#         # 我们希望可以使用本地的小模型进行生成, 但是评估部分使用大模型. 同时，尝试使用大模型完全接管最后几步.
#         if args.method_generate == "sample" and step < task.steps - 2:
#             new_ys = [
#                 get_samples(
#                     task,
#                     x,
#                     y,
#                     args.n_generate_sample,
#                     prompt_sample=args.prompt_sample,
#                     stop=task.stops[step],
#                 )
#                 for y in ys
#             ]
#         elif args.method_generate == "propose" and True:
#             new_ys = [get_proposals(task, x, y) for y in ys]
#         elif args.method_generate == "sample" and step >= task.steps - 2:
#             new_ys = [
#                 get_samples_usingLLM(
#                     task,
#                     x,
#                     y,
#                     args.n_generate_sample,
#                     prompt_sample=args.prompt_sample,
#                     stop=task.stops[step],
#                 )
#                 for y in ys
#             ]
#         elif args.method_generate == "propose" and False:
#             new_ys = [get_proposals_usingLLM(task, x, y) for y in ys]
#         new_ys = list(itertools.chain(*new_ys))
#         ids = list(range(len(new_ys)))
#         # evaluation
#         if args.method_evaluate == "vote":
#             values = get_votes(task, x, new_ys, args.n_evaluate_sample)
#         elif args.method_evaluate == "value":
#             values = get_values(task, x, new_ys, args.n_evaluate_sample)
#         # selection
#         if args.method_select == "sample":
#             ps = np.array(values) / sum(values)
#             select_ids = np.random.choice(ids, size=args.n_select_sample, p=ps).tolist()
#         elif args.method_select == "greedy":
#             select_ids = sorted(ids, key=lambda x: values[x], reverse=True)[
#                 : args.n_select_sample
#             ]
#         select_new_ys = [new_ys[select_id] for select_id in select_ids]

#         # log
#         if to_print:
#             sorted_new_ys, sorted_values = zip(
#                 *sorted(zip(new_ys, values), key=lambda x: x[1], reverse=True)
#             )
#             print(
#                 f"-- new_ys --: {sorted_new_ys}\n-- sol values --: {sorted_values}\n-- choices --: {select_new_ys}\n"
#             )

#         infos.append(
#             {
#                 "step": step,
#                 "x": x,
#                 "ys": ys,
#                 "new_ys": new_ys,
#                 "values": values,
#                 "select_new_ys": select_new_ys,
#             }
#         )
#         ys = select_new_ys

#     if to_print:
#         print(ys)
#     return ys, {"steps": infos}
# ---------------------------------------------------
