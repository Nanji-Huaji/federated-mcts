import itertools
import numpy as np
from functools import partial

from federated_mcts.models.api_client import gpt
from federated_mcts.core.generation import get_proposals, get_samples
from federated_mcts.core.evaluation import get_values, get_votes


def solve(args, task, idx, to_print=True):
    # Errors will occur if the function is deleted.
    global gpt
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    print(gpt)
    x = task.get_input(idx)  # input
    ys = [""]  # current output candidates
    infos = []
    for step in range(task.steps):
        # generation
        if args.method_generate == "sample":
            candidate_batches = [
                get_samples(
                    args, task, x, y, args.n_generate_sample, prompt_sample=args.prompt_sample, stop=task.stops[step]
                )
                for y in ys
            ]
        elif args.method_generate == "propose":
            candidate_batches = [get_proposals(args, step, task, x, y) for y in ys]
        new_ys: list[str] = list(itertools.chain.from_iterable(candidate_batches))
        ids = list(range(len(new_ys)))
        # evaluation
        if args.method_evaluate == "vote":
            values = get_votes(args, task, x, new_ys, args.n_evaluate_sample)
        elif args.method_evaluate == "value":
            values = get_values(args, task, x, new_ys, args.n_evaluate_sample)

        # selection
        if args.method_select == "sample":
            ps = np.array(values) / sum(values)
            select_ids = np.random.choice(ids, size=args.n_select_sample, p=ps).tolist()
        elif args.method_select == "greedy":
            select_ids = sorted(ids, key=lambda x: values[x], reverse=True)[: args.n_select_sample]
        select_new_ys = [new_ys[select_id] for select_id in select_ids]

        # log
        if to_print:
            sorted_new_ys, sorted_values = zip(*sorted(zip(new_ys, values), key=lambda x: x[1], reverse=True))
            print(f"-- new_ys --: {sorted_new_ys}\n-- sol values --: {sorted_values}\n-- choices --: {select_new_ys}\n")

        infos.append(
            {"step": step, "x": x, "ys": ys, "new_ys": new_ys, "values": values, "select_new_ys": select_new_ys}
        )
        ys = select_new_ys

    if to_print:
        print(ys)
    return ys, {"steps": infos}


def naive_solve(args, task, idx, to_print=True, model=None):
    if model is None:
        global gpt
        gpt = partial(gpt, model=args.localbackend, temperature=args.temperature)
    print(gpt)
    x = task.get_input(idx)  # input
    ys = get_samples(args, task, x, "", args.n_generate_sample, args.prompt_sample, stop=None)
    return ys, {}
