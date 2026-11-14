"""DQN search session: episode bookkeeping between the solvers and the
Budget-Aware DQN controller.

The session computes the pre-decision state, applies the controller's action
(beam x joint-rank), evaluates candidates accordingly, selects the beam-width
subset, and records one transition per decision step.  Rewards are sparse:
intermediate transitions carry a zero correctness base; only the final
transition receives the exact correctness reward (finalize) minus bounded
token/latency penalties.  The action is always chosen BEFORE the current
step's joint ranking, from candidate/dedup structure, previous-step value
stats and cumulative budgets.

Exact verification belongs to the caller (``finish_task``): the terminal
transition is held until :meth:`DqnSearchSession.finalize` supplies the real
success flag, so the recorded reward is final on its first write.
"""

from __future__ import annotations

import time
from typing import Callable

from federated_mcts.core.dqn.episode import DqnEpisode
from federated_mcts.core.dqn.step import DqnStepOutcome
from federated_mcts.core.evaluation import EvaluatorValueCache
from federated_mcts.core.joint_ranking import evaluate_ranked_candidates
from federated_mcts.core.search_policy import deduplicate_candidates, successful_candidate
from federated_mcts.models.usage import get_usage_tracker


class DqnSearchSession:
    def __init__(
        self,
        controller,
        *,
        token_budget: float = 5000.0,
        latency_budget: float = 60.0,
        max_tokens: int = 1000,
        budget_seconds: float = 10.0,
        jsonl_path: str | None = None,
        oracle_distance_reward_enabled: bool = False,
        oracle_distance_scale: float = 0.25,
        evaluator_cache_for: Callable[[object, str, object, str], EvaluatorValueCache] | None = None,
    ):
        self.controller = controller
        self.token_budget = float(token_budget)
        self.latency_budget = float(latency_budget)
        self.max_tokens = max_tokens
        self.budget_seconds = budget_seconds
        self._episode = DqnEpisode(
            max_tokens=max_tokens, budget_seconds=budget_seconds,
            distance_scale=oracle_distance_scale, jsonl_path=jsonl_path,
        )
        self.oracle_distance_reward_enabled = oracle_distance_reward_enabled
        self.oracle_distance_scale = oracle_distance_scale
        self._oracle_distance: int | None = None
        self._oracle_optimal: int | None = None
        self._previous_beam_width: int | None = None
        # The joint-ranking score cache is partitioned by the evaluator's
        # client *instance* and identity, then reset when the solved task or
        # its input changes.  One session therefore never mixes scores from
        # two tasks, two adapters, or two labels.
        self._value_caches: dict[tuple[int, str], dict] = {}
        self._client_refs: dict[int, object] = {}
        self._task = None
        self._input: str | None = None
        self._evaluator_cache_for = evaluator_cache_for
        self.last_values: list[float] | None = None
        self.last_joint_rank: bool | None = None
        self._cumulative_tokens = 0.0
        self._cumulative_latency = 0.0

    def drain_transitions(self) -> list[dict]:
        return self._episode.drain()

    def finish_episode(self, success: bool = False, *, flush_terminal: bool = True) -> None:
        """Close any dangling pending transition as the terminal one.

        ``flush_terminal=False`` leaves the terminal record unpublished until
        :meth:`finalize` learns the exact verification result.
        """
        self._episode.finish(bool(success), flush_terminal=flush_terminal)

    def finalize(self, terminal_success: bool) -> None:
        """Attribute the exact task reward to the final transition and publish it."""
        self._episode.finalize(bool(terminal_success))

    # -- evaluator-scoped caches -------------------------------------------

    def _ranking_cache(self, task, x: str, client, evaluator_id: str) -> dict:
        if task is not self._task or x != self._input:
            self._task, self._input = task, x
            self._value_caches.clear()
            self._client_refs.clear()
        client_key = id(client)
        self._client_refs.setdefault(client_key, client)
        return self._value_caches.setdefault((client_key, evaluator_id), {})

    def _value_cache(self, task, x: str, client, evaluator_id: str) -> EvaluatorValueCache | None:
        if self._evaluator_cache_for is None:
            return None
        return self._evaluator_cache_for(task, evaluator_id, client, x)

    # -- per-step processing ------------------------------------------------

    def process_step(
        self,
        *,
        args,
        task,
        x: str,
        candidates: list[str],
        client,
        evaluator_id: str,
        step: int,
        total_steps: int,
        step_tokens: float = 0.0,
        step_latency: float = 0.0,
    ) -> DqnStepOutcome:
        unique, states = deduplicate_candidates(task, x, candidates)
        ranking_cache = self._ranking_cache(task, x, client, evaluator_id)
        from federated_mcts.core.dqn.oracle_session import (
            initial_distance,
            oracle_distance_enabled,
            selected_distance,
        )

        oracle_enabled = oracle_distance_enabled(task, self.oracle_distance_reward_enabled)
        if oracle_enabled and self._oracle_distance is None:
            self._oracle_distance = initial_distance(x)
        success = successful_candidate(task, x, unique)

        decision_state = None
        if success is None:
            if oracle_enabled and self._oracle_optimal is None:
                from federated_mcts.core.dqn.oracle_distance import min_remaining_distance
                self._oracle_optimal = min_remaining_distance(x, [""])
            decision_state = self.controller.decide(
                candidates=unique,
                states=states,
                previous_values=self.last_values,
                step=step,
                total_steps=total_steps,
                tokens_consumed=self._cumulative_tokens,
                token_budget=self.token_budget,
                latency_consumed=self._cumulative_latency,
                latency_budget=self.latency_budget,
                previous_joint_rank=self.last_joint_rank,
                task_optimal_length=self._oracle_optimal if oracle_enabled else None,
                current_remaining_distance=self._oracle_distance,
                previous_beam_width=self._previous_beam_width,
            )
            self._episode.close(next_state=decision_state.state, done=False, success=False)
            self._episode.open(decision_state.state, decision_state.action, self._oracle_distance if oracle_enabled else None)
            joint_rank = decision_state.joint_rank
            self.last_joint_rank = joint_rank
        else:
            joint_rank = self.last_joint_rank if self.last_joint_rank is not None else False

        evaluation_tokens_before = get_usage_tracker().get_total_tokens()
        if success is not None:
            values = [1.0 if candidate == success else 0.0 for candidate in unique]
            select_new_ys = [success]
            stopped = True
            eval_seconds = 0.0
        else:
            if decision_state is None:
                raise RuntimeError("DQN decision state is required for evaluation")
            eval_budget = {2: 2, 3: 4, 4: 6}.get(decision_state.beam, len(unique))
            if eval_budget < len(unique):
                unique = unique[:eval_budget]
            ranking_start = time.time()
            values = evaluate_ranked_candidates(
                args,
                task,
                x,
                unique,
                client,
                evaluator_id,
                ranking_cache,
                joint_rank=joint_rank,
                value_cache=self._value_cache(task, x, client, evaluator_id),
            )
            eval_seconds = time.time() - ranking_start
            width = max(1, min(decision_state.beam, len(unique)))
            selected_ids = sorted(
                range(len(unique)), key=lambda index: values[index], reverse=True
            )[:width]
            select_new_ys = [unique[index] for index in selected_ids]
            stopped = False

        eval_tokens = max(0, get_usage_tracker().get_total_tokens() - evaluation_tokens_before)
        total_step_tokens = step_tokens + eval_tokens
        self._episode.add_costs(tokens=total_step_tokens, latency=step_latency + eval_seconds)
        self._cumulative_tokens += total_step_tokens
        self._cumulative_latency += step_latency + eval_seconds

        if oracle_enabled and self._episode.pending is not None:
            self._episode.pending["distance_after"] = selected_distance(x, select_new_ys, stopped)
            self._oracle_distance = self._episode.pending["distance_after"]
            self._previous_beam_width = len(select_new_ys)
        if stopped:
            self._episode.close(next_state=None, done=True, success=True)
        self.last_values = values

        action = None if decision_state is None else decision_state.action
        search_metrics = {
            "raw_candidates": len(candidates),
            "unique_states": len(unique),
            "beam_width": len(select_new_ys),
            "dqn_action": action,
            "dqn_beam": None if decision_state is None else decision_state.beam,
            "dqn_joint_rank": joint_rank,
            "step_tokens": float(step_tokens),
            "eval_tokens": float(eval_tokens),
            "total_step_tokens": float(total_step_tokens),
            "step_latency": float(step_latency),
            "eval_seconds": float(eval_seconds),
        }
        return DqnStepOutcome(
            candidates=unique,
            values=values,
            selected=select_new_ys,
            stopped=stopped,
            transitions=self.drain_transitions(),
            search_metrics=search_metrics,
            eval_seconds=eval_seconds,
        )
