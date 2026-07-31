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
"""

from __future__ import annotations

import time

from federated_mcts.core.dqn.episode import DqnEpisode
from federated_mcts.core.dqn.step import DqnStepOutcome
from federated_mcts.core.dqn.transition_rewards import build_reward_components
from federated_mcts.core.joint_ranking import evaluate_ranked_candidates
from federated_mcts.core.search_policy import deduplicate_candidates, successful_candidate


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
        self._value_cache: dict = {}
        self.last_values = None
        self.last_joint_rank: bool | None = None
        self._cumulative_tokens = 0.0
        self._cumulative_latency = 0.0
    def drain_transitions(self) -> list[dict]:
        return self._episode.drain()

    def finish_episode(self, success: bool = False) -> None:
        """Close any dangling pending transition as the terminal one."""
        self._episode.finish(bool(success))

    def finalize(self, terminal_success: bool) -> None:
        """Re-attribute the exact task reward to the final transition,
        replacing the provisional success-state heuristic used at solve time
        (the runner only knows the exact reward after task.test_output_modify)."""
        self._episode.finalize(bool(terminal_success))

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
            )
            self._episode.close(next_state=decision_state.state, done=False, success=False)
            self._episode.open(decision_state.state, decision_state.action, self._oracle_distance if oracle_enabled else None)
            joint_rank = decision_state.joint_rank
            self.last_joint_rank = joint_rank
        else:
            joint_rank = self.last_joint_rank if self.last_joint_rank is not None else False

        if success is not None:
            values = [1.0 if candidate == success else 0.0 for candidate in unique]
            select_new_ys = [success]
            stopped = True
            eval_seconds = 0.0
        else:
            ranking_start = time.time()
            values = evaluate_ranked_candidates(
                args,
                task,
                x,
                unique,
                client,
                evaluator_id,
                self._value_cache,
                joint_rank=joint_rank,
            )
            eval_seconds = time.time() - ranking_start
            width = max(1, min(decision_state.beam, len(unique)))
            selected_ids = sorted(
                range(len(unique)), key=lambda index: values[index], reverse=True
            )[:width]
            select_new_ys = [unique[index] for index in selected_ids]
            stopped = False

        self._episode.add_costs(tokens=step_tokens, latency=step_latency + eval_seconds)
        self._cumulative_tokens += step_tokens
        self._cumulative_latency += step_latency + eval_seconds

        if oracle_enabled and self._episode.pending is not None:
            self._episode.pending["distance_after"] = selected_distance(x, select_new_ys, stopped)
            self._oracle_distance = self._episode.pending["distance_after"]
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
