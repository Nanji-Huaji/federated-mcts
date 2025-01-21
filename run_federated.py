import os
import json
import argparse
from run_usingLLM import parse_args

from tot.tasks import get_task
from tot.methods.bfs import naive_solve, solve_usingLLM_eval, federated_solve
from tot.models import gpt_usage

import openai


"""
多Client联合推理方案
"""

"""
base_command = [
    "--task", "game24",
    "--task_start_index", "900",
    "--task_end_index", "909",
    "--method_generate", "propose",
    "--method_evaluate", "value",
    "--method_select", "greedy",
    "--n_evaluate_sample", "3",
    "--n_select_sample", "5",
    "--temperature", "0.9",
    "--localbackend", "meta-llama-3.1-8b-instruct@q4_k_m",
]
add_command = [
    "--slm_generate",
    "--check_format",
    "--eval_rule",
    "--slm_eval",
    "--warm_start"]

"""


"""
当前为朴素的多端协作推理，类似于Self-Consistency
federated_run会在多端以相同的方式进行一次推理
"""




def federated_run(args):
    task = get_task(args.task)
    logs, cnt_avg, cnt_any = {}, 0, 0
    lat_all, lat_generate, lat_eval = 0, 0, 0
    federated_client = {"local_client": {"api_base": "http://127.0.0.1:11451/v1", "api_key": "lm-studio", "model": args.localbackend},
                    "remote_client" : {"api_base": "http://158.132.255.40:1234/v1", "api_key": "lm-studio", "model": args.remotebackend}}
    if args.naive_run:
        file = f"./logs/federated/{args.task}/{args.localbackend}_{args.remotebackend}/{args.temperature}_naive_{args.prompt_sample}_sample_{args.n_generate_sample}_start{args.task_start_index}_end{args.task_end_index}_federated"
    else:
        file = f"./logs/federated/{args.task}/{args.localbackend}_{args.remotebackend}/{args.temperature}_{args.prompt_sample}_sample_{args.n_generate_sample}_start{args.task_start_index}_end{args.task_end_index}_federated"
    print("file:", file)
    if not os.path.exists(file):
        os.makedirs(file)
    for i in range(args.task_start_index, args.task_end_index + 1):
        print('Solve task ', i)
        # solve
        if args.naive_run:
            ys, info = naive_solve(args, task, i)
        else:
            ys, info, lat_dict = federated_solve(args, task, i, to_print=True, **federated_client)

        print("ys ", ys)
        # 收到ys, info, lat_dict
        # 此时ys和info的意义如下：
        '''
        ys = {}  # 输出候选，格式为{model1: [y1, y2, ...], model2: [y1, y2, ...], ...}
        infos = {} # 信息，格式为{model1: {step: 0, x: x, ys: [], new_ys: [], values: [], select_new_ys: []}, model2: {...}, ...}
        '''
        # log
        new_output = {}
        infos, output_list = {}, {}
        for model in federated_client.keys():
            # Set default to avoid key error
            if model not in logs.keys():
                logs.setdefault(model, [])
            if model not in infos.keys():
                infos.setdefault(model, [])
            if model not in output_list.keys():
                output_list.setdefault(model, []) 
            if model not in new_output.keys():
                new_output.setdefault(model, [])
            for y in ys[model]:
                r, new_output[model] = task.test_output_modfiy(i, y)
                if model not in new_output.keys():
                    new_output.setdefault(model, [])
                if model not in output_list.keys():
                    output_list.setdefault(model, [])
                if(new_output[model] not in output_list[model]):  # Avoid duplication of outputs
                    output_list[model].append(new_output)
                else:
                    r = {"r": 0}  # Do not count twice
                infos[model].append(r)
            token_consumption = 0 # 暂时未完成
            if model not in info.keys():
                info.setdefault(model, {})
            info[model].update(
                {
                    "model": model,
                    "idx": i,
                    "ys": ys,
                    "infos": infos[model],
                    "usage_so_far": token_consumption,
                }
            )
            info[model].update(lat_dict)
            # lat_all[model], lat_generate[model], lat_eval[model] = (
            # lat_all[model] + sum(lat_dict[model]["all"]),
            # lat_generate[model] + sum(lat_dict[model]["generate"]),
            # lat_eval[model] + sum(lat_dict[model]["eval"]),
        # )
        logs[model].append(info[model])
    print("========推理结束========")
    print("log为空") if not logs else print(logs)
    # with open(file + ".json", "w") as f:
    #     json.dump(logs, f, indent=4)
    import sys
    print(sys.argv[0])
    with open(file + ".json", "w") as f:
        json.dump(logs, f, indent=4)






def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument(
        "--localbackend",
        type=str,
        choices=[
            "gpt-4",
            "gpt-3.5-turbo",
            "gpt-4o",
            "gpt-4o-mini",
            "lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF",
            "bartowski/Phi-3-medium-128k-instruct-GGUF",
            "meta-llama-3.1-8b-instruct@q4_k_m",
            "Qwen/Qwen2.5-32B-Instruct-GGUF",
            "phi-3.1-mini-128k-instruct",
        ],
        default="phi-3.1-mini-128k-instruct",
    )
    args.add_argument(
        "--remotebackend",
        type=str,
        choices=[
            "gpt-4",
            "gpt-3.5-turbo",
            "gpt-4o",
            "gpt-4o-mini",
            "lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF",
            "bartowski/Phi-3-medium-128k-instruct-GGUF",
            "meta-llama-3.1-8b-instruct@q4_k_m",
            "Qwen/Qwen2.5-32B-Instruct-GGUF",
            "qwen2.5-32b-instruct"
        ],
        default="qwen2.5-32b-instruct",
    )
    args.add_argument("--temperature", type=float, default=0.9)

    args.add_argument(
        "--task", type=str, required=True, choices=["game24", "text", "crosswords"]
    )
    args.add_argument("--task_start_index", type=int, default=900)
    args.add_argument("--task_end_index", type=int, default=1000)

    args.add_argument("--naive_run", action="store_true")
    args.add_argument(
        "--prompt_sample", type=str, choices=["standard", "cot"]
    )  # only used when method_generate = sample, or naive_run

    args.add_argument("--method_generate", type=str, choices=["sample", "propose"])
    args.add_argument("--method_evaluate", type=str, choices=["value", "vote"])
    args.add_argument(
        "--method_select", type=str, choices=["sample", "greedy"], default="greedy"
    )
    args.add_argument(
        "--n_generate_sample", type=int, default=1
    )  # only thing needed if naive_run
    args.add_argument("--n_evaluate_sample", type=int, default=1)
    args.add_argument("--n_select_sample", type=int, default=1)

    # jinyu
    # args.add_argument(
    #     "--slm_generate", action="store_true", help="use small lm for generation"
    # )
    # args.add_argument(
    #     "--slm_eval", action="store_true", help="use small lm for evaluation"
    # )
    args.add_argument(
        "--check_format",
        action="store_true",
        help="check the format and correctness of the generated contents",
    )
    args.add_argument(
        "--eval_rule", action="store_true", help="use rules for evaluation"
    )
    # args.add_argument(
    #     "--warm_start",
    #     action="store_true",
    #     help="step 0 uses large model for generation",
    # )
    args.add_argument(
        "--inference_idx", type=int, default=0, help="Do multiple experiments"
    )
    # args.add_argument(
    #     "--last_lm", action="store_true", help="Use the large model for the last step"
    # )

    # args.add_argument("--filter", action="store_true", help="Enable filtering for specific runs.")



    args = args.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)
    federated_run(args)
