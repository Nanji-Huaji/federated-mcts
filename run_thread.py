import os
import json
import argparse
from run_edge_cloud import parse_args

from tot.tasks import get_task
from tot.methods.bfs import naive_solve, thread_solve
from tot.models import gpt_usage
import run

import openai

import time
import threading

from collections import defaultdict

file = ""

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


def read_model_list(file_path) -> list:
    with open(file_path, "r") as f:
        model_list = json.load(f)
    return model_list


def run_thread(args):
    global file
    task = get_task(args.task)
    logs, cnt_avg, cnt_any = [], 0, 0
    lat_all, lat_generate, lat_eval = 0, 0, 0
    simple_task, hard_task = [], []
    model_name = [model["model"] for model in model_list]
    model_name_str = "_".join(model_name)
    time_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
    file = f"./logs/federated/thread/{args.task}/{model_name_str}/{args.temperature}_{args.prompt_sample}_sample_{args.n_generate_sample}_start{args.task_start_index}_end{args.task_end_index}"
    file += "_" + time_str
    os.makedirs(os.path.dirname(file + ".json"), exist_ok=True)

    def thread_solve_wrapper(model_dict: dict, i: int):
        """
        model_info: dict, "client_name": {"model": model_name, "api_base": api_base, "api_key": api_key}
        """
        return thread_solve(
            args,
            task,
            i,
            api_base=model_dict["api_base"],
            api_key=model_dict["api_key"],
            model=model_dict["model"],
            client_name=model_dict["client_name"],
        )

    results = {}

    # Multi-threading Definition
    threads = []
    for i in range(args.task_start_index, args.task_end_index):
        for model_info in model_list:

            def thread_function(model_info=model_info, i=i):
                result = thread_solve_wrapper(model_info, i)
                results[(model_info["model"], i)] = result

            t = threading.Thread(target=thread_function)
            threads.append(t)
            print(f"Thread {t} started.")
            t.start()

    for t in threads:
        t.join()
        print(f"Thread {t} finished.")

    print("All threads finished.")

    # Log
    for i in range(args.task_start_index, args.task_end_index):
        for model_info in model_list:
            result = results[(model_info["model"], i)]
            ys, info, lat_dict = result
            # log
            print("ys ", ys)
            infos, output_list = [], []
            for y in ys:
                r, new_output = task.test_output_modfiy(i, y)  # type: ignore
                if new_output not in output_list:
                    output_list.append(new_output)
                else:
                    r = {"r": 0}
                infos.append(r)
            token_consumption = gpt_usage(args.localbackend)
            info.update(
                {
                    "idx": i,
                    "ys": ys,
                    "infos": infos,
                    "usage_so_far": token_consumption,
                }
            )
            info.update(lat_dict)
            lat_all, lat_generate, lat_eval = (
                lat_all + sum(lat_dict["all"]),
                lat_generate + sum(lat_dict["generate"]),
                lat_eval + sum(lat_dict["eval"]),
            )
            logs.append(info)
            with open(file + ".json", "w") as f:
                json.dump(logs, f, indent=4)
            print(i, "sum(accs)", "cnt_avg", cnt_avg, "cnt_any", cnt_any, "\n")


def merge_results():
    # Merge the results of different clients produced into a json file.
    global file
    file_path = file + ".json"
    # Read the json file
    with open(file_path, "r") as log_file:
        data = json.load(log_file)
    # Create a dict to store the infos by idx
    infos_by_idx = defaultdict(list)
    for item in data:
        idx = item["idx"]
        infos_by_idx[idx].extend(item["infos"])
        if ("idx: " + str(idx)) not in infos_by_idx[idx]:
            infos_by_idx[idx].insert(0, "idx: " + str(idx))
    logs = []
    # Merge the results
    for idx, infos in infos_by_idx.items():
        logs.append(infos)  # type: ignore
        # Write the merged infos to a new json file
    with open(file + "_merged.json", "w") as f:
        json.dump(logs, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    args = parse_args()
    print(args)
    if args.naive_run:
        run.run(args)  # Call the original run function
    else:
        run_thread(args)
        merge_results()
