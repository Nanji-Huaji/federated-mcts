import os
import json
import argparse
from run_edge_cloud import parse_args

from tot.tasks import get_task
from tot.methods.bfs import naive_solve, thread_solve
from tot.models import gpt_usage

import openai

import time
import thread
import threading

model_list = {"local_client": {"api_base": "http://127.0.0.1:11451/v1", "api_key": "lm-studio", "model": "meta-llama-3.1-8b-instruct@q4_k_m"},
                "remote_client": {"api_base": "http://158.132.255.40:1234/v1", "api_key": "lm-studio", "model": "qwen2.5-32b-instruct"}}



def run(args):
    task = get_task(args.task)
    logs, cnt_avg, cnt_any = [], 0, 0
    lat_all, lat_generate, lat_eval = 0, 0, 0
    simple_task, hard_task = [], []
    model_name = [model["model"] for model in model_list.values()]
    model_name_str = "_".join(model_name)
    time_str = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    file = f"./logs/federated/thread/{args.task}/{model_name_str}/{args.temperature}_{args.prompt_sample}_sample_{args.n_generate_sample}_start{args.task_start_index}_end{args.task_end_index}"
    file += ("_" + time_str)
    os.makedirs(os.path.dirname(file + ".json"), exist_ok=True)

    def thread_solve_wrapper(model_info, i):
        '''
        model_info: dict, {"model": model_name, "api_base": api_base, "api_key": api_key}
        '''
        return thread_solve(args, task, i, api_base=model_info["api_base"], api_key=model_info["api_key"], model=model_info["model"])   
    
    results = {}

    # Multi-threading Definition
    threads = []
    for i in range(args.task_start_index, args.task_end_index):
        for model_info in model_list.values():
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
        for model_info in model_list.values():
            result = results[(model_info["model"], i)]
            ys, info, lat_dict = result
            # log
            print("ys ", ys)
            infos, output_list = [], []
            for y in ys:
                r, new_output = task.test_output_modfiy(i, y)
                if(new_output not in output_list):
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
            print("=========================================")
            print(logs)

    

if __name__ == '__main__':
    args = parse_args()
    run(args)