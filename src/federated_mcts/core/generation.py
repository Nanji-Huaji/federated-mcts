import itertools
import numpy as np
from functools import partial
from federated_mcts.models.api_client import gpt  # keep old import for now
import os
import re
from federated_mcts.utils.format_check import check_and_fix_last_line
import time
from federated_mcts.tasks import get_task
import json
from math import ceil
from typing import Literal, Tuple, List, Dict, Callable, Union, overload
import federated_mcts.utils.uncertainty as uncertainty


def get_proposals_with_check(
    args,
    step,
    task,
    x,
    y,
    api_key=None,
    api_base=None,
    model=None,
    client=None,
    get_logprobs=False,
    n_generate=4,
) -> list[str]:
    # jinyu:
    need_generate = task.pre_generate_check(y) if hasattr(task, "pre_generate_check") else True
    if need_generate == False:  # no need to generate new proposals
        return [y]

    new_proposal_list: list[str] = []
    run_times = 0
    time_constraint, len_constraint = 2, 4

    while len(new_proposal_list) < len_constraint and run_times < time_constraint:  # Generate at least 4 proposals

        propose_prompt = task.propose_prompt_wrap(x, y)
        if client is None:
            completions = gpt(
                args,
                propose_prompt,
                n=n_generate,
                stop=None,
                api_key=api_key,
                api_base=api_base,
                model=model,
            )
        else:
            completions = client(args, propose_prompt, n=n_generate, stop=None)
        # jinyu: iterate all completions, split each by line
        for completion in completions:
            for pro in completion.split("\n"):
                if hasattr(task, "process_generate_result") and args.check_format:
                    is_correct, updated_new_proposal = task.process_generate_result(pro, x, y, args.check_format)
                    if is_correct:
                        if updated_new_proposal not in new_proposal_list:
                            new_proposal_list.append(updated_new_proposal)
        run_times += 1

    if run_times >= time_constraint:
        print("runtime ", run_times)
    if len(new_proposal_list) == 0:
        # Optional hook: blocksworld returns [] so invalid branches die; the
        # historical [y] fallback stays for tasks without the hook.
        no_candidate_hook = getattr(task, "on_no_valid_candidates", None)
        return no_candidate_hook(y) if no_candidate_hook is not None else [y]
    return new_proposal_list


def get_proposals_without_check(
    args,
    step,
    task,
    x,
    y,
    api_key=None,
    api_base=None,
    model=None,
    client=None,
    get_logprobs=False,
) -> list[str]:
    propose_prompt = task.propose_prompt_wrap(x, y)
    if client is None:
        proposals = gpt(
            args,
            propose_prompt,
            n=1,
            stop=None,
            api_key=api_key,
            api_base=api_base,
            model=model,
        )[0].split("\n")
    else:
        proposals = client(args, propose_prompt, n=1, stop=None)[0].split("\n")
    return [y + _ + "\n" for _ in proposals]


def get_proposals_with_logits(args, step, task, x, y, draft_model):
    pass


def get_proposals(
    args,
    step,
    task,
    x,
    y,
    api_key=None,
    api_base=None,
    model=None,
    client=None,
    get_logprobs=False,
    n_generate=4,
)-> list[str]:
    if args.check_format:
        return get_proposals_with_check(
            args,
            step,
            task,
            x,
            y,
            api_key=api_key,
            api_base=api_base,
            model=model,
            client=client,
            get_logprobs=get_logprobs,
            n_generate=n_generate,
        )
    return get_proposals_without_check(
        args,
        step,
        task,
        x,
        y,
        api_key=api_key,
        api_base=api_base,
        model=model,
        client=client,
        get_logprobs=get_logprobs,
    )


@overload
def get_samples(
    args,
    task,
    x,
    y,
    n_generate_sample,
    prompt_sample,
    stop,
    api_key=None,
    api_base=None,
    model=None,
    client=None,
    get_logprobs: Literal[False] = False,
) -> list[str]: ...


@overload
def get_samples(
    args,
    task,
    x,
    y,
    n_generate_sample,
    prompt_sample,
    stop,
    api_key=None,
    api_base=None,
    model=None,
    client=None,
    get_logprobs: Literal[True] = True,
) -> tuple[list[str], list[float]]: ...


def get_samples(
    args,
    task,
    x,
    y,
    n_generate_sample,
    prompt_sample,
    stop,
    api_key=None,
    api_base=None,
    model=None,
    client=None,
    get_logprobs: bool = False,
) -> list[str] | tuple[list[str], list[float]]:
    if prompt_sample == "standard":
        prompt = task.standard_prompt_wrap(x, y)
    elif prompt_sample == "cot":
        prompt = task.cot_prompt_wrap(x, y)
    else:
        raise ValueError(f"prompt_sample {prompt_sample} not recognized")

    if client is None:
        samples_result = gpt(
            args,
            prompt,
            n=n_generate_sample,
            stop=stop,
            api_key=api_key,
            api_base=api_base,
            model=model,
            get_logprobs=get_logprobs,
        )
    else:
        samples_result = client(
            args, prompt, n=n_generate_sample, stop=stop, get_logprobs=get_logprobs
        )

    if get_logprobs:
        if isinstance(samples_result, tuple):
            samples, logprobs = samples_result
        else:
            # compatibility handling
            samples = samples_result
            logprobs = [0.0] * len(samples)

        # return concatenated samples and logprobs
        final_samples = [y + _ for _ in samples]
        return final_samples, logprobs
    else:
        if isinstance(samples_result, tuple):
            samples, logprobs = samples_result
        else:
            samples = samples_result

        return [y + _ for _ in samples]
