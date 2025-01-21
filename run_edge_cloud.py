import os
import json
import argparse
from run_usingLLM import parse_args

from tot.tasks import get_task
from tot.methods.bfs import naive_solve, solve_usingLLM_eval
from tot.models import gpt_usage

import openai


def run(args):
    task = get_task(args.task)
    logs, cnt_avg, cnt_any = [], 0, 0
    lat_all, lat_generate, lat_eval = 0, 0, 0
    simple_task, hard_task = [], []

    file_edge = f"./logs/{args.task}/{args.localbackend}/{args.remotebackend}/{args.temperature}_{args.method_generate}{args.n_generate_sample}_{args.method_evaluate}{args.n_evaluate_sample}_{args.method_select}{args.n_select_sample}_start{args.task_start_index}_end{args.task_end_index}_last_{args.last_lm}_edge_model_idx_{args.inference_idx}"
    os.makedirs(os.path.dirname(file_edge + ".json"), exist_ok=True)
    file_cloud = f"./logs/{args.task}/{args.localbackend}/{args.remotebackend}/{args.temperature}_{args.method_generate}{args.n_generate_sample}_{args.method_evaluate}{args.n_evaluate_sample}_{args.method_select}{args.n_select_sample}_start{args.task_start_index}_end{args.task_end_index}_last_{args.last_lm}_cloud_model_idx_{args.inference_idx}"
    os.makedirs(os.path.dirname(file_cloud + ".json"), exist_ok=True)
    file_performance = f"./logs/{args.task}/{args.localbackend}/{args.remotebackend}/{args.temperature}_{args.method_generate}{args.n_generate_sample}_{args.method_evaluate}{args.n_evaluate_sample}_{args.method_select}{args.n_select_sample}_start{args.task_start_index}_end{args.task_end_index}_last_{args.last_lm}_edge_cloud_idx_{args.inference_idx}_performance.json"
    os.makedirs(os.path.dirname(file_performance + ".json"), exist_ok=True)

    # edge model
    args.slm_generate, args.slm_eval = True, True
    for i in range(args.task_start_index, args.task_end_index):
        print('Solve task ', i)
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
            r, new_output = task.test_output_modfiy(i, y) # type: ignore
            if(new_output not in output_list):  # Avoid duplication of outputs
                output_list.append(new_output)
            else:
                r = {"r": 0}  # Do not count twice
            infos.append(r)
        token_consumption_edge = gpt_usage(args.localbackend)
        info.update(
            {
                "idx": i,
                "ys": ys,
                "infos": infos,
                "usage_so_far": token_consumption_edge,
            } # type: ignore
        ) # type: ignore
        info.update(lat_dict)  # jinyu: update the latency
        lat_all, lat_generate, lat_eval = (
            lat_all + sum(lat_dict["all"]),
            lat_generate + sum(lat_dict["generate"]),
            lat_eval + sum(lat_dict["eval"]),
        )
        logs.append(info)
        with open(file_edge + ".json", "w") as f:
            json.dump(logs, f, indent=4)

        # log main metric
        accs = [info["r"] for info in infos]
        cnt_avg += sum(accs)  # / len(accs) #jinyu: counting the sum
        cnt_any += any(accs)
        print(i, "sum(accs)", sum(accs), "cnt_avg", cnt_avg, "cnt_any", cnt_any, "\n")

        if(sum(accs)>0):
            simple_task.append(i)
        else:
            hard_task.append(i)

    n = args.task_end_index - args.task_start_index
    print("The average sum is ", cnt_avg / n, ". The accuracy is: ", cnt_any / n)
    print("Token consumption: ", token_consumption_edge)
    print("Latency: ", lat_all, ", ", lat_generate, ", ", lat_eval)
    # res_json = {
    #     "avg_sum": cnt_avg / n,
    #     "acc": cnt_any / n,
    #     "lat": lat_all,
    #     "lat_generate": lat_generate,
    #     "lat_eval": lat_eval,
    # }
    # res_json.update(token_consumption)
    # with open(file_edge + "_performance.json", "w") as f:
    #     json.dump(res_json, f, indent=4)


    logs = []
    # cloud model
    print('hard task', hard_task)
    args.slm_generate, args.slm_eval = False, False
    for i in hard_task:
        print('Solve task ', i, ', hard task', hard_task)
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
            r, new_output = task.test_output_modfiy(i, y)
            if(new_output not in output_list):  # Avoid duplication of outputs
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
            }
        )
        info.update(lat_dict)  # jinyu: update the latency
        lat_all, lat_generate, lat_eval = (
            lat_all + sum(lat_dict["all"]),
            lat_generate + sum(lat_dict["generate"]),
            lat_eval + sum(lat_dict["eval"]),
        )
        logs.append(info)
        with open(file_cloud + ".json", "w") as f:
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
    }
    res_json.update(token_consumption)
    with open(file_performance, "w") as f:
        json.dump(res_json, f, indent=4)


if __name__ == "__main__":
    args = parse_args()
    print(args)
    print(f"api_key为{openai.api_key}, api_base为{openai.api_base}.")
    run(args)
