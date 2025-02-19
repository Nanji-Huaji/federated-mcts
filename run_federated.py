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
        "model": "phi-3-medium-4k-instruct",
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


def list_merge(ys):
    """
    Merge a list of lists into one list
    """
    return [item for sublist in ys for item in sublist]


def run(args):
    global file
    task = get_task(args.task)
    logs, cnt_avg, cnt_any = [], 0, 0
    lat_all, lat_generate, lat_eval = 0, 0, 0
    simple_task, hard_task = [], []
    # Define the log file path
    model_name = [model["model"] for model in model_list]
    model_name_str = "_".join(model_name)
    time_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
    if args.naive_run:
        file = f"./logs/{args.task}/{args.localbackend}/{args.remotebackend}/{args.temperature}_naive_{args.prompt_sample}_sample_{args.n_generate_sample}_start{args.task_start_index}_end{args.task_end_index}"
    else:
        file = f"./logs/federated/{args.task}/{model_name_str}/{args.temperature}_{args.prompt_sample}_sample_{args.n_generate_sample}_start{args.task_start_index}_end{args.task_end_index}"
    file += "_" + time_str
    os.makedirs(os.path.dirname(file + ".json"), exist_ok=True)
    if args.naive_run:
        raise NotImplementedError("Naive run is not implemented yet")
    print(f"The models is defined as {model_list}")
    info = []  # info is a dictionary that stores the information of the task

    for i in range(args.task_start_index, args.task_end_index):
        ys = [""]  # ys is a list of outputs from each model
        print(f"Task {i}")
        task = get_task(args.task)
        current_task = task.get_input(i)
        for step in range(task.steps):
            print(f"Step {step} of {task.steps} in Task {i}")  # 在第1个task中的第2个step卡住了，一直输出runtime  3
            # TODO
            # 3. aggregate results
            step_ys = []
            if (not ys) or (not ys[0]):
                # Do the first inference using the 0th model if ys is empty
                new_ys, new_info, lat_dict = client_solve_wrapper(
                    args, task, current_task, [""], model_list[0], step, to_print=True
                )
                step_ys.extend(new_ys)
                info.append(new_info)
                print(f"初始化完成！new_ys为{new_ys}")
            else:  # if ys is not empty
                print("ys非空，开始分配任务")
                print(f"ys为{ys}，ys的长度为{len(ys)}")
                # Assign task to client
                task_list = assign_task(model_list, ys)
                # Remove the " " in the task_list
                task_list = [task for task in task_list if task != " "]
                print(f"分配任务完成，task_list为{task_list}")
                # Solve task on each client
                for i in range(min(len(model_list), len(task_list))):
                    print(f"在{model_list[i]}上推理{task_list[i]}")
                    new_ys, new_info, lat_dict = client_solve_wrapper(
                        args, task, current_task, task_list[i], model_list[i], step, to_print=True
                    )
                    step_ys += new_ys
                    print(f"推理完成，step_ys为{step_ys}")
                    print(f"new_ys: {new_ys}，new_info: {new_info}")
                    info.append(new_info)
                    print(f"new_info: {new_info}")
                    # TODO: Update the latency
                    # lat_all += lat_dict["all"]
                    # lat_generate += lat_dict["generate"]
                    # lat_eval += lat_dict["eval"]
            # Aggregate results
            ys = list_merge(ys)  # Convert the data structure from list of list to list
            ys = step_ys.copy()

        # log
        infos, output_list = [], []
        for y in ys:
            r, new_output = task.test_output_modfiy(i, y)  # type: ignore
            if new_output not in output_list:  # Avoid duplication of outputs
                output_list.append(new_output)
            else:
                r = {"r": 0}  # Do not count twice
            infos.append(r)
        # token_consumption = gpt_usage(args.localbackend)
        token_consumption = 0
        # TODO: Add token consumption to the log

        # TODO: Save logs
        info.append(lat_dict)  # type: ignore
        # lat_all, lat_generate, lat_eval = (
        #     lat_all + sum(lat_dict["all"]),
        #     lat_generate + sum(lat_dict["generate"]),
        #     lat_eval + sum(lat_dict["eval"]),
        # )
        logs.append(info)
        logs.append(infos)
        with open(file + ".json", "w") as f:
            json.dump(logs, f, indent=4)


if __name__ == "__main__":
    args = parse_args()
    run(args)
    print("Done")
