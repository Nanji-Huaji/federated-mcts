import os
import json
import argparse

from tot.tasks import get_task
from tot.methods.bfs import naive_solve, solve_usingLLM_eval
from tot.models import gpt_usage

import openai


def run(args):
    task = get_task(args.task)
    logs, cnt_avg, cnt_any = [], 0, 0
    lat_all, lat_generate, lat_eval = 0, 0, 0
    if args.naive_run:
        file = f"./logs/{args.task}/{args.localbackend}/{args.remotebackend}/{args.temperature}_naive_{args.prompt_sample}_sample_{args.n_generate_sample}_start{args.task_start_index}_end{args.task_end_index}_usingLLM"
    else:
        file = f"./logs/{args.task}/{args.localbackend}/{args.remotebackend}/{args.temperature}_{args.method_generate}{args.n_generate_sample}_{args.method_evaluate}{args.n_evaluate_sample}_{args.method_select}{args.n_select_sample}_start{args.task_start_index}_end{args.task_end_index}_smg_{args.slm_generate}_sme_{args.slm_eval}_check_{args.check_format}_rule_{args.eval_rule}_warm_{args.warm_start}_last_{args.last_lm}_idx_{args.inference_idx}"
    os.makedirs(os.path.dirname(file + ".json"), exist_ok=True)

    for i in range(args.task_start_index, args.task_end_index):
        print("Solve task ", i)
        # solve
        if args.naive_run:
            ys, info = naive_solve(args, task, i)
        else:
            ys, info, lat_dict = solve_usingLLM_eval(args, task, i)

        print(f"task.step为{task.steps}.")

        # log
        print("ys ", ys)
        infos, output_list = [], []
        for y in ys:
            r, new_output = task.test_output_modfiy(i, y)  # type: ignore
            if new_output not in output_list:  # Avoid duplication of outputs
                output_list.append(new_output)
            else:
                r = {"r": 0}  # Do not count twice
            infos.append(r)
        token_consumption = gpt_usage(args.localbackend)
        info.update(
            {
                "idx": i,
                "ys": ys,
                "infos": infos,
                "usage_so_far": token_consumption,
            }  # type: ignore
        )  # type: ignore
        info.update(lat_dict)  # jinyu: update the latency
        lat_all, lat_generate, lat_eval = (
            lat_all + sum(lat_dict["all"]),
            lat_generate + sum(lat_dict["generate"]),
            lat_eval + sum(lat_dict["eval"]),
        )
        logs.append(info)
        with open(file + ".json", "w") as f:
            json.dump(logs, f, indent=4)

        # log main metric
        accs = [info["r"] for info in infos]
        cnt_avg += sum(accs)  # / len(accs) #jinyu: counting the sum
        cnt_any += any(accs)
        print(i, "sum(accs)", sum(accs), "cnt_avg", cnt_avg, "cnt_any", cnt_any, "\n")

    n = args.task_end_index - args.task_start_index
    print("The average sum is ", cnt_avg / n, ". The accuracy is: ", cnt_any / n)
    print("Token consumption: ", token_consumption)
    print("Latency: ", lat_all, ", ", lat_generate, ", ", lat_eval)
    res_json = {
        "avg_sum": cnt_avg / n,
        "acc": cnt_any / n,
        "lat": lat_all,
        "lat_generate": lat_generate,
        "lat_eval": lat_eval,
        "sm": args.localbackend,
        "llm": args.remotebackend,
    }
    res_json.update(token_consumption)

    with open(file + "_performance.json", "w") as f:
        json.dump(res_json, f, indent=4)


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
        default="bartowski/Phi-3-medium-128k-instruct-GGUF",
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
            "qwen2.5-32b-instruct",
        ],
        default="qwen2.5-32b-instruct",
    )
    args.add_argument("--temperature", type=float, default=0.9)
    args.add_argument("--task", type=str, required=True, choices=["game24", "text", "crosswords"])
    args.add_argument("--task_start_index", type=int, default=900)
    args.add_argument("--task_end_index", type=int, default=1000)
    args.add_argument("--naive_run", action="store_true")
    args.add_argument(
        "--prompt_sample", type=str, choices=["standard", "cot"]
    )  # only used when method_generate = sample, or naive_run
    args.add_argument("--method_generate", type=str, choices=["sample", "propose"])
    args.add_argument("--method_evaluate", type=str, choices=["value", "vote"])
    args.add_argument("--method_select", type=str, choices=["sample", "greedy"], default="greedy")
    args.add_argument("--n_generate_sample", type=int, default=1)  # only thing needed if naive_run
    args.add_argument("--n_evaluate_sample", type=int, default=1)
    args.add_argument("--n_select_sample", type=int, default=1)

    # jinyu
    args.add_argument("--slm_generate", action="store_true", help="use small lm for generation")
    args.add_argument("--slm_eval", action="store_true", help="use small lm for evaluation")
    args.add_argument(
        "--check_format",
        action="store_true",
        help="check the format and correctness of the generated contents",
    )
    args.add_argument("--eval_rule", action="store_true", help="use rules for evaluation")
    args.add_argument(
        "--warm_start",
        action="store_true",
        help="step 0 uses large model for generation",
    )
    args.add_argument("--inference_idx", type=int, default=0, help="Do multiple experiments")
    args.add_argument("--last_lm", action="store_true", help="Use the large model for the last step")

    args.add_argument("--filter", action="store_true", help="Enable filtering for specific runs.")

    args = args.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)
    print(f"api_key is set to {openai.api_key}, api_base is set to{openai.api_base}.")
    run(args)
