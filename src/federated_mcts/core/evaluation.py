from typing import Callable, List, Tuple, Union
from federated_mcts.models.api_client import gpt
from federated_mcts.evaluation.game24_instrumentation import (
    VoteCallRecord,
    VoteRecorder,
    parse_vote_response,
)
from federated_mcts.evaluation.game24_vote_recording import record_vote_step


def get_value(
    args,
    task,
    x,
    y,
    n_evaluate_sample,
    cache_value=True,
    api_key=None,
    api_base=None,
    model=None,
    client: Callable | None = None,
):
    value_prompt = task.value_prompt_wrap(x, y)
    if cache_value and value_prompt in task.value_cache:
        return task.value_cache[value_prompt]
    if client is None:
        value_outputs = gpt(
            args, value_prompt, n=n_evaluate_sample, stop=None, api_key=api_key, api_base=api_base, model=model
        )
    else:
        value_outputs = client(args, value_prompt, n=n_evaluate_sample, stop=None)
    if isinstance(value_outputs, tuple):
        value_outputs = value_outputs[0]
    value = task.value_outputs_unwrap(x, y, value_outputs)
    if cache_value:
        task.value_cache[value_prompt] = value
    return value


def get_values(
    args, task, x, ys, n_evaluate_sample, cache_value=True, api_key=None, api_base=None, model=None, client=None
):
    values = []
    local_value_cache = {}
    for y in ys:  # each partial output
        if y in local_value_cache:  # avoid duplicate candidates
            value = 0
        else:
            # jinyu
            value, final = task.pre_value_check(y, args.eval_rule) if hasattr(task, "pre_value_check") else (0, False)
            if value == 0 and final == False:
                count = 0
                while value == 0 and count < 2:
                    value = get_value(
                        args,
                        task,
                        x,
                        y,
                        n_evaluate_sample,
                        cache_value=cache_value,
                        api_key=api_key,
                        api_base=api_base,
                        model=model,
                        client=client,
                    )
                    count += 1
            local_value_cache[y] = value
        values.append(value)
    return values


def get_votes(
    args, task, x, ys, n_evaluate_sample, api_key=None, api_base=None,
    model=None, client=None, recorder: VoteRecorder | None = None,
    evaluator_mode: str = "vote", evaluator_identity: str = "", depth: int = 0,
):
    vote_prompt = task.vote_prompt_wrap(x, ys)
    if client is None:
        vote_outputs = gpt(
            args, vote_prompt, n=n_evaluate_sample, stop=None, api_key=api_key, api_base=api_base, model=model
        )
    else:
        vote_outputs = client(args, vote_prompt, n=n_evaluate_sample, stop=None)
    if isinstance(vote_outputs, tuple):
        vote_outputs = vote_outputs[0]
    values = task.vote_outputs_unwrap(vote_outputs, len(ys))
    if recorder is not None:
        calls: list[VoteCallRecord] = []
        for call_index, raw_output in enumerate(vote_outputs):
            parsed_index, parse_success, failure_reason = parse_vote_response(
                raw_output)
            in_range = (parsed_index is not None and
                        0 <= parsed_index < len(ys))
            if parse_success and not in_range:
                failure_reason = "out_of_range"
            calls.append(VoteCallRecord(
                call_index=call_index,
                evaluator_mode=evaluator_mode,
                evaluator_identity=evaluator_identity,
                raw_output=raw_output,
                parsed_index=parsed_index,
                parse_success=parse_success,
                failure_reason=failure_reason,
                in_range=in_range,
            ))
        record_vote_step(
            recorder, depth, evaluator_mode, evaluator_identity, len(ys),
            values, calls)
    return values
