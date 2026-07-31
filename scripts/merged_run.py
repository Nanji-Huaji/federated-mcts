import os
import json
import argparse

from federated_mcts.env_loader import load_dotenv
load_dotenv()


from federated_mcts.tasks import get_task

from federated_mcts.models import get_model_usage_summary

import openai


from federated_mcts.federation import FederatedSolver






def file_name_generater(args):
    if hasattr(args, "model_config"):
        with open(args.model_config, "r") as f:
            model_config = json.load(f)
        args.localbackend = list(
            filter(lambda model_config_dict: model_config_dict["client_name"] == "local_client", model_config)
        )[0]["model"]
        args.remotebackend = list(
            filter(lambda model_config_dict: model_config_dict["client_name"] == "remote_client", model_config)
        )[0]["model"]
        print(f"Local backend: {args.localbackend}, Remote backend: {args.remotebackend}")
    if args.naive_run:
        file = f"./logs/{args.task}/{args.localbackend}/{args.remotebackend}/{args.temperature}_naive_{args.prompt_sample}_sample_{args.n_generate_sample}_start{args.task_start_index}_end{args.task_end_index}_usingLLM"
    else:
        policy_suffix = {
            "diverse": "_policy_diverse",
            "dqn": "_policy_dqn",
        }.get(getattr(args, "search_policy", "baseline"), "")
        file = f"./logs/{args.task}/{args.solve_method}/{args.remotebackend}/{args.temperature}_{args.method_generate}_n_generate_sample_{args.n_generate_sample}_{args.method_evaluate}_n_evaluate_sample_{args.n_evaluate_sample}_method_select_{args.method_select}_n_select_sample_{args.n_select_sample}{policy_suffix}_start{args.task_start_index}_end{args.task_end_index}_smg_{args.slm_generate}_sme_{args.slm_eval}_check_{args.check_format}_rule_{args.eval_rule}_warm_{args.warm_start}_last_{args.last_lm}_idx_{args.inference_idx}"
    os.makedirs(os.path.dirname(file + ".json"), exist_ok=True)
    print(f"File name: {file}.json")
    return file


def run(args, solve_function):
    file_name = file_name_generater(args)

    solver = FederatedSolver(args)
    function_map = {
        "naive": solver.naive_solve,
        "tot": solver.solve,
        "speculative_solve": solver.speculative_solve,
        "federated_solve": solver.federated_solve,
    }
    solve_function = function_map[solve_function]

    logs, cnt_avg, cnt_any = [], 0, 0

    task_indices = range(args.task_start_index, args.task_end_index)
    if getattr(args, "task_split", None):
        import json
        with open(f"data/splits/{args.task}.json") as f:
            split = json.load(f)[args.task_split]
        task_indices = list(split)
        print("Using %s split: %d tasks, indices %d..%d" % (args.task_split, len(task_indices), task_indices[0], task_indices[-1]))
    for i in task_indices:
        ys = [""]
        print(f"Task {i}")
        task = get_task(args.task)

        ys, info = solve_function(task, i, assign_strategy=args.assign_strategy)

        # log
        print("ys ", ys)
        infos, output_list = [], []
        for y in ys:
            r, new_output = task.test_output_modify(i, y)  # type: ignore
            if new_output not in output_list:  # Avoid duplication of outputs
                output_list.append(new_output)
            else:
                r = {"r": 0}  # Do not count twice
            infos.append(r)

        # Re-attribute the terminal DQN reward with the exact task reward (the solver only sees the success-state heuristic).
        if getattr(args, "search_policy", "baseline") == "dqn":
            session = getattr(solver, "dqn_session", None)
            if session is not None:
                accs = [info["r"] for info in infos]
                session.finalize(any(accs))

        token_consumption = get_model_usage_summary()
        time_consumption = solver.latency_dict
        info.update(
            {
                "idx": i,
                "ys": ys,
                "infos": infos,
                "usage_so_far": token_consumption,
                "time_consumption": time_consumption,
            }
        )

        logs.append(info)

        with open(file_name + ".json", "w") as f:
            json.dump(logs, f, indent=4)

        # log main metric
        accs = [info["r"] for info in infos]
        cnt_avg += sum(accs)  # / len(accs) #jinyu: counting the sum
        cnt_any += any(accs)
        print(i, "sum(accs)", sum(accs), "cnt_avg", cnt_avg, "cnt_any", cnt_any, "\n")

    n = len(task_indices)
    print("The average sum is ", cnt_avg / n, ". The accuracy is: ", cnt_any / n)
    print("Token consumption: ", token_consumption)
    res_json = {
        "avg_sum": cnt_avg / n,
        "acc": cnt_any / n,
        "sm": args.localbackend,
        "llm": args.remotebackend,
    }
    res_json.update(token_consumption)

    with open(file_name + "_performance.json", "w") as f:
        json.dump(res_json, f, indent=4)
        print(f"Performance results saved to {file_name}_performance.json")


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
    args.add_argument("--task", type=str, required=True, choices=["game24", "text", "crosswords", "gsm8k", "wikilogic", "hyblogic", "blocksworld"])
    args.add_argument("--task_split", type=str, default=None, choices=["train", "val", "test"],
                        help="use predefined split from data/splits/{task}.json (overrides --task_start/--task_end)")
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
    args.add_argument("--search_policy", choices=["baseline", "diverse", "dqn"], default="baseline")
    args.add_argument("--diverse_joint_rank", action=argparse.BooleanOptionalAction, default=True)
    args.add_argument("--diversity_weight", type=float, default=0.35)
    args.add_argument("--beam_expand", type=int, default=2)
    args.add_argument("--beam_uncertainty_margin", type=float, default=0.1)
    args.add_argument("--beam_confidence_margin", type=float, default=0.8)
    args.add_argument("--dqn_seed", type=int, default=0, help="seed for the DQN controller")
    args.add_argument("--dqn_epsilon", type=float, default=1.0, help="exploration rate; clamped to >= 0.5 while collecting")
    args.add_argument("--dqn_state_dim", type=int, default=12)
    args.add_argument("--dqn_capacity", type=int, default=10000)
    args.add_argument("--dqn_token_budget", type=float, default=5000.0)
    args.add_argument("--dqn_latency_budget", type=float, default=60.0)
    args.add_argument(
        "--dqn_checkpoint", type=str, default=None,
        help="path to a trained DQN checkpoint; missing file starts collection-only mode",
    )
    args.add_argument(
        "--dqn_jsonl", type=str, default=None,
        help="path to append collected DQN transitions as JSONL",
    )
    args.add_argument(
        "--solve_method", type=str, choices=["naive", "tot", "speculative_solve", "federated_solve"], default="tot"
    )
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
    args.add_argument(
        "--model_config", type=str, default="configs/model_config.json", help="Path to the model configuration file"
    )
    args.add_argument("--inference_idx", type=int, default=0, help="Do multiple experiments")
    args.add_argument("--last_lm", action="store_true", help="Use the large model for the last step")

    args.add_argument("--assign_strategy", type=str, default="round_robin",
                        choices=["round_robin", "difficulty", "bandit"],
                        help="Task assignment strategy (default: round_robin)")
    args.add_argument("--filter", action="store_true", help="Enable filtering for specific runs.")

    args = args.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.filter:
        print("Filtering enabled. Only runs with specific criteria will be executed.")
    else:
        print("No filtering applied. All runs will be executed.")

    run(args, args.solve_method)
