import os
import json
from tot.tasks import get_task
from tot.methods.bfs import assign_task
import time
from tot.methods.bfs import federated_solve
from run_usingLLM import parse_args
from functools import partial


federated_token_usage = {}

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


def run(args):
    global file, federated_token_usage
    federated_token_usage = {model: 0 for model in map(lambda m: m["model"], model_list)}
    task = get_task(args.task)
    logs, cnt_avg, cnt_any = [], 0, 0
    lat_all, lat_generate, lat_eval = 0, 0, 0
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

    # generation
    gen_start = time.time()
    for i in range(args.task_start_index, args.task_end_index):
        print(f"Solving task {i}")
        # solve
        if args.naive_run:
            raise NotImplementedError("Naive run is not implemented yet")
        else:
            ys, info, lat_dict = federated_solve(
                args,
                task,
                i,
                model_list,
                assign_task,
                to_print=True,
            )

        # log
        print("ys: ", ys)
        infos, output_list = [], []
        for y in ys:
            r, new_output = task.test_output_modfiy(i, y)
            if new_output not in output_list:
                output_list.append(new_output)
            else:
                r = {"r": 0}
            infos.append(r)
        info.update(
            {
                "idx": i,
                "ys": ys,
                "infos": infos,
                "usage_so_far": federated_token_usage,
            }  # type: ignore
        )  # type: ignore
        logs.append(info)
        with open(file + ".json", "w") as f:
            json.dump(logs, f, indent=4)

        # log main metric
        accs = [info["r"] for info in infos]
        cnt_avg += sum(accs)
        cnt_any += any(accs)
        print(i, "sum(accs)", sum(accs), "cnt_avg", cnt_avg, "cnt_any", cnt_any, "\n")
        n = args.task_end_index - args.task_start_index
    print("The average sum is ", cnt_avg / n, ". The accuracy is: ", cnt_any / n)
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
    with open(file + "_performance.json", "w") as f:
        json.dump(res_json, f, indent=4)


if __name__ == "__main__":

    args = parse_args()
    run(args)
