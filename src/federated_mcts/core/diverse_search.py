from dataclasses import dataclass

from federated_mcts.core.joint_ranking import CacheKey, evaluate_ranked_candidates
from federated_mcts.core.search_policy import (
    DiverseSearchConfig,
    adaptive_beam_width,
    deduplicate_candidates,
    diverse_select,
    successful_candidate,
)


@dataclass(frozen=True, slots=True)
class SearchDecision:
    candidates: list[str]
    values: list[float]
    selected: list[str]
    stopped: bool


class DiverseSearch:
    def __init__(self, args):
        self.config = DiverseSearchConfig.from_args(args)
        self.cache: dict[CacheKey, float] = {}

    def decide(self, args, task, x: str, candidates: list[str], client, evaluator_id: str) -> SearchDecision:
        unique, states = deduplicate_candidates(task, x, candidates)
        success = successful_candidate(task, x, unique)
        if success is not None:
            values = [1.0 if candidate == success else 0.0 for candidate in unique]
            return SearchDecision(unique, values, [success], True)
        values = evaluate_ranked_candidates(
            args,
            task,
            x,
            unique,
            client,
            evaluator_id,
            self.cache,
            self.config.joint_rank,
        )
        width = adaptive_beam_width(
            values,
            self.config.base_width,
            self.config.max_expansion,
            self.config.uncertainty_margin,
            self.config.confidence_margin,
        )
        selected_ids = diverse_select(
            unique,
            values,
            states,
            width,
            self.config.diversity_weight,
        )
        return SearchDecision(unique, values, [unique[index] for index in selected_ids], False)
