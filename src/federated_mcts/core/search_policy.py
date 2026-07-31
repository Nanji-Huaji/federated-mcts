from collections import Counter
from dataclasses import dataclass
from typing import Hashable, Protocol


class StateTask(Protocol):
    def canonical_state_key(self, x: str, y: str) -> Hashable: ...

    def is_success_state(self, x: str, y: str) -> bool: ...


@dataclass(frozen=True, slots=True)
class DiverseSearchConfig:
    base_width: int
    max_expansion: int
    diversity_weight: float
    uncertainty_margin: float
    confidence_margin: float
    joint_rank: bool

    @classmethod
    def from_args(cls, args) -> "DiverseSearchConfig":
        return cls(
            base_width=max(1, args.n_select_sample),
            max_expansion=max(0, getattr(args, "beam_expand", 2)),
            diversity_weight=max(0.0, getattr(args, "diversity_weight", 0.35)),
            uncertainty_margin=max(0.0, getattr(args, "beam_uncertainty_margin", 0.1)),
            confidence_margin=max(0.0, getattr(args, "beam_confidence_margin", 0.8)),
            joint_rank=getattr(args, "diverse_joint_rank", True),
        )


def state_key(task, x: str, candidate: str) -> Hashable:
    hook = getattr(task, "canonical_state_key", None)
    return hook(x, candidate) if hook is not None else candidate


def deduplicate_candidates(task, x: str, candidates: list[str]) -> tuple[list[str], list[Hashable]]:
    unique: list[str] = []
    states: list[Hashable] = []
    seen: set[Hashable] = set()
    for candidate in candidates:
        state = state_key(task, x, candidate)
        if state in seen:
            continue
        seen.add(state)
        unique.append(candidate)
        states.append(state)
    return unique, states


def adaptive_beam_width(
    values: list[float],
    base_width: int,
    max_expansion: int,
    uncertainty_margin: float,
    confidence_margin: float,
) -> int:
    if not values:
        return 0
    candidate_count = len(values)
    base = min(max(1, base_width), candidate_count)
    ordered = sorted(values, reverse=True)
    if len(ordered) > 1 and ordered[0] - ordered[1] >= confidence_margin:
        return max(1, base - 1)
    if base < candidate_count and ordered[base - 1] - ordered[base] <= uncertainty_margin:
        return min(candidate_count, base + max_expansion)
    return base


def _state_similarity(left: Hashable, right: Hashable) -> float:
    if left == right:
        return 1.0
    left_items = tuple(left) if isinstance(left, tuple) else (left,)
    right_items = tuple(right) if isinstance(right, tuple) else (right,)
    left_counts = Counter(left_items)
    right_counts = Counter(right_items)
    overlap = sum((left_counts & right_counts).values())
    total = sum((left_counts | right_counts).values())
    return overlap / total if total else 0.0


def diverse_select(
    candidates: list[str],
    values: list[float],
    states: list[Hashable],
    width: int,
    diversity_weight: float,
) -> list[int]:
    remaining = set(range(len(candidates)))
    selected: list[int] = []
    while remaining and len(selected) < width:
        def score(index: int) -> tuple[float, float, int]:
            similarity = max(
                (_state_similarity(states[index], states[chosen]) for chosen in selected),
                default=0.0,
            )
            mmr = values[index] - diversity_weight * similarity
            return mmr, values[index], -index

        chosen = max(remaining, key=score)
        selected.append(chosen)
        remaining.remove(chosen)
    return selected


def successful_candidate(task, x: str, candidates: list[str]) -> str | None:
    hook = getattr(task, "is_success_state", None)
    if hook is None:
        return None
    return next((candidate for candidate in candidates if hook(x, candidate)), None)
