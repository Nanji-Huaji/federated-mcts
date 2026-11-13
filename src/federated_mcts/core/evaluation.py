from typing import Callable, List, Tuple, Union
from federated_mcts.models.api_client import gpt
from federated_mcts.core.observation import (
    VoteCallRecord,
    VoteObservationSink,
    parse_vote_choice,
    record_vote_step,
)


class EvaluatorValueCache:
    """Value cache keyed by ``(evaluator identity, value prompt)``.

    The historical ``Task.value_cache`` was keyed by prompt text alone.  A
    search may evaluate the same prompt with more than one evaluator (and the
    prompt does not even carry the puzzle ``x``), so a prompt-only key let one
    evaluator's score leak into another evaluator's decision.  Putting the
    evaluator identity into the key keeps caching safe while preserving the
    "same evaluator, same prompt" hit that the cache exists for.

    Mirrors the evaluator-scoped key used by ``core.joint_ranking``.
    """

    def __init__(self) -> None:
        self._values: dict[tuple[str, str], float] = {}

    def get(self, evaluator_identity: str, value_prompt: str) -> float | None:
        return self._values.get((evaluator_identity, value_prompt))

    def put(
        self, evaluator_identity: str, value_prompt: str, value: float,
    ) -> None:
        self._values[(evaluator_identity, value_prompt)] = value

    def clear(self) -> None:
        self._values.clear()

    def __len__(self) -> int:
        return len(self._values)


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
    cache: EvaluatorValueCache | None = None,
    evaluator_identity: str = "",
):
    """Score one candidate with the value evaluator.

    ``cache``/``evaluator_identity`` scope the cross-call cache explicitly;
    without a cache the call is never served from, nor written to, the
    task-level ``value_cache`` (that prompt-only cache leaked scores across
    evaluators and is no longer used).  ``cache_value`` and the remaining
    parameters keep their historical meaning and positions.
    """
    value_prompt = task.value_prompt_wrap(x, y)
    if cache_value and cache is not None:
        cached = cache.get(evaluator_identity, value_prompt)
        if cached is not None:
            return cached
    if client is None:
        value_outputs = gpt(
            args, value_prompt, n=n_evaluate_sample, stop=None, api_key=api_key, api_base=api_base, model=model
        )
    else:
        value_outputs = client(args, value_prompt, n=n_evaluate_sample, stop=None)
    if isinstance(value_outputs, tuple):
        value_outputs = value_outputs[0]
    value = task.value_outputs_unwrap(x, y, value_outputs)
    if cache_value and cache is not None:
        cache.put(evaluator_identity, value_prompt, float(value))
    return value


def get_values(
    args, task, x, ys, n_evaluate_sample, cache_value=True, api_key=None, api_base=None, model=None, client=None,
    cache: EvaluatorValueCache | None = None, evaluator_identity: str = "",
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
                        cache=cache,
                        evaluator_identity=evaluator_identity,
                    )
                    count += 1
            local_value_cache[y] = value
        values.append(value)
    return values


def get_votes(
    args, task, x, ys, n_evaluate_sample, api_key=None, api_base=None,
    model=None, client=None, recorder: VoteObservationSink | None = None,
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
            parsed_index, parse_success, failure_reason = parse_vote_choice(
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
