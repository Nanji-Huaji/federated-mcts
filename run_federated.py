import os
import json
import argparse
from run_usingLLM import parse_args

from tot.tasks import get_task
from tot.methods.bfs import naive_solve, client_solve, assign_task
from tot.models import gpt_usage
import time

import openai

# model_weight is used to judge the inference capability of each model
model_weight = {
    "meta-llama-3.1-8b-instruct@q4_k_m": 1,
    "qwen2.5-32b-instruct": 1,
    "gpt-4o": 5,
}

# model_list is a list of models that we want to use in the federated learning
model_list = [
    {
        "client_name": "local_client",
        "api_base": "http://127.0.0.1:11451/v1",
        "api_key": "lm-studio",
        "model": "meta-llama-3.1-8b-instruct@q4_k_m",
    },
    {
        "client_name": "remote_client",
        "api_base": "http://158.132.255.40:1234/v1",
        "api_key": "lm-studio",
        "model": "qwen2.5-32b-instruct",
    },
]


def client_solve_wrapper(args, task, current_task, ys, model_dict: dict, step: int, to_print=True):
    """
    model_info: dict, "client_name": {"model": model_name, "api_base": api_base, "api_key": api_key}
    """
    return client_solve(
        args,
        task,
        current_task,
        ys,
        step,
        model_dict["api_base"],
        model_dict["api_key"],
        model_dict["model"],
        model_dict["client_name"],
        to_print,
    )


def run(args):
    global file
    task = get_task(args.task)
    logs, cnt_avg, cnt_any = [], 0, 0
    lat_all, lat_generate, lat_eval = 0, 0, 0
    simple_task, hard_task = [], []
    ys = []  # all the ys from all the models, this is used for communication between models
    # Define the log file path
    model_name = [model["model"] for model in model_list]
    model_name_str = "_".join(model_name)
    time_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
    file = f"./logs/federated/thread/{args.task}/{model_name_str}/{args.temperature}_{args.prompt_sample}_sample_{args.n_generate_sample}_start{args.task_start_index}_end{args.task_end_index}"
    file += "_" + time_str
    os.makedirs(os.path.dirname(file + ".json"), exist_ok=True)

    for i in range(args.task_start_index, args.task_end_index):
        task = get_task(args.task)
        current_task = task.get_input(i)
        info = []
        for step in range(task.steps):
            # TODO
            # 0. initialize ys if it is empty(ok)
            # 1. assign task to client(ok)
            # 2. solve task on each client(ok?)
            # 3. aggregate results
            step_ys = []
            if (not ys) or (not ys[0]):
                # Do the first inference using the 0th model if ys is empty
                new_ys, new_info, lat_dict = client_solve_wrapper(
                    args, task, current_task, [], model_list[0], step, to_print=True
                )
                ys.append(new_ys)
            else:  # if ys is not empty
                # Assign task to client
                task_list = assign_task(model_list, ys)
                # Solve task on each client
                for i in range(min(len(model_list), len(task_list))):
                    new_ys, new_info, lat_dict = client_solve_wrapper(
                        args, task, current_task, ys, model_list[i], step, to_print=True
                    )
                    step_ys.append(new_ys)
                    info.append(new_info)
                    lat_all += lat_dict["all"]
                    lat_generate += lat_dict["generate"]
                    lat_eval += lat_dict["eval"]
            # Aggregate results
            ys = step_ys if step_ys else ys
        # log
        infos, output_list = [], []
        print("ys ", ys)
        for y in ys:
            r, new_output = task.test_output_modfiy(i, y)  # type: ignore
            if new_output not in output_list:  # Avoid duplication of outputs
                output_list.append(new_output)
            else:
                r = {"r": 0}  # Do not count twice
            infos.append(r)
        # token_consumption = gpt_usage(args.localbackend)
        # TODO: Add token consumption to the log
        info.update(  # type: ignore
            {
                "idx": i,
                "ys": ys,
                "infos": infos,
                "usage_so_far": token_consumption,  # type: ignore
            }
        )
        # TODO: Save logs
        info.update(lat_dict)  # type: ignore # jinyu: update the latency
        lat_all, lat_generate, lat_eval = (
            lat_all + sum(lat_dict["all"]),
            lat_generate + sum(lat_dict["generate"]),
            lat_eval + sum(lat_dict["eval"]),
        )
        logs.append(info)
        with open(file + ".json", "w") as f:
            json.dump(logs, f, indent=4)
