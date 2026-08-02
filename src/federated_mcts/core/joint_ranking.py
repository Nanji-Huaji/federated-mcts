import json
import math
from typing import Hashable

from federated_mcts.core.evaluation import get_values
from federated_mcts.core.search_policy import state_key


CacheKey = tuple[str, str, Hashable]


def parse_ranked_scores(payload: str, candidate_count: int) -> list[float] | None:
    start = payload.find("{")
    end = payload.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        document = json.loads(payload[start : end + 1])
        ranking = document["ranking"]
        parsed = {int(item["id"]): float(item["score"]) for item in ranking}
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
    expected = set(range(candidate_count))
    if len(ranking) != candidate_count or set(parsed) != expected:
        return None
    scores = [parsed[index] for index in range(candidate_count)]
    if not all(math.isfinite(score) and 0.0 <= score <= 1.0 for score in scores):
        return None
    return scores


def _joint_scores(args, task, x: str, candidates: list[str], client) -> list[float] | None:
    prompt_hook = getattr(task, "joint_rank_prompt_wrap", None)
    if prompt_hook is None:
        return None
    outputs = client(args, prompt_hook(x, candidates), n=1, stop=None)
    if isinstance(outputs, tuple):
        outputs = outputs[0]
    return parse_ranked_scores(outputs[0], len(candidates)) if outputs else None


def evaluate_ranked_candidates(
    args,
    task,
    x: str,
    candidates: list[str],
    client,
    evaluator_id: str,
    cache: dict[CacheKey, float],
    joint_rank: bool,
) -> list[float]:
    values: list[float | None] = [None] * len(candidates)
    missing_ids: list[int] = []
    missing_candidates: list[str] = []
    task_id = type(task).__name__
    for index, candidate in enumerate(candidates):
        key = (task_id, evaluator_id, state_key(task, x, candidate))
        if key in cache:
            values[index] = cache[key]
        else:
            missing_ids.append(index)
            missing_candidates.append(candidate)

    scores = _joint_scores(args, task, x, missing_candidates, client) if joint_rank and missing_candidates else None
    if missing_candidates and scores is None:
        scores = get_values(
            args, task, x, missing_candidates, args.n_evaluate_sample, client=client,
        )

    for index, score in zip(missing_ids, scores or []):
        if score is None:
            numeric_score = 0.0
        else:
            numeric_score = float(score)
        values[index] = numeric_score
        cache[(task_id, evaluator_id, state_key(task, x, candidates[index]))] = numeric_score
    return [0.0 if value is None else float(value) for value in values]
