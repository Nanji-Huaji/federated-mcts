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

from federated_mcts.core.dqn.actions import beam_and_joint_rank
from federated_mcts.core.dqn.controller import _plain
from federated_mcts.core.dqn.recorder import JSONLTransitionRecorder
from federated_mcts.core.dqn.rewards import correctness_reward, latency_penalty, token_penalty
from federated_mcts.core.dqn.step import DqnStepOutcome
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
    ):
        self.controller = controller
        self.token_budget = float(token_budget)
        self.latency_budget = float(latency_budget)
        self.max_tokens = max_tokens
        self.budget_seconds = budget_seconds
        self.recorder = JSONLTransitionRecorder(jsonl_path) if jsonl_path else None
        self._value_cache: dict = {}
        self.last_values = None
        self.last_joint_rank: bool | None = None
        self._cumulative_tokens = 0.0
        self._cumulative_latency = 0.0
        self._pending: dict | None = None
        self._all_transitions: list[dict] = []
        self._pending_transitions: list[dict] = []
        self._last_terminal_costs = (0.0, 0.0)

    # -- episode lifecycle -------------------------------------------------

    def _open_pending(self, state, action: int) -> None:
        beam, joint_rank = beam_and_joint_rank(action)
        self._pending = {
            "state": _plain(state),
            "action": int(action),
            "beam": int(beam),
            "joint_rank": bool(joint_rank),
            "tokens": 0.0,
            "latency": 0.0,
        }

    def _close_pending(self, *, next_state, done: bool, success: bool) -> dict | None:
        if self._pending is None:
            return None
        pending = self._pending
        self._pending = None
        base = correctness_reward(success) if done else 0.0
        reward = (
            base
            - token_penalty(pending["tokens"], self.max_tokens)
            - latency_penalty(pending["latency"], self.budget_seconds)
        )
        transition = {
            "state": pending["state"],
            "action": pending["action"],
            "reward": float(reward),
            "next_state": None if next_state is None else _plain(next_state),
            "done": bool(done),
            "beam": pending["beam"],
            "joint_rank": pending["joint_rank"],
        }
        self._all_transitions.append(transition)
        self._pending_transitions.append(transition)
        if done:
            self._last_terminal_costs = (pending["tokens"], pending["latency"])
        if self.recorder is not None:
            self.recorder.record(transition)
        return transition

    def _add_costs(self, *, tokens: float, latency: float) -> None:
        if self._pending is None:
            return
        self._pending["tokens"] += float(tokens)
        self._pending["latency"] += float(latency)

    def drain_transitions(self) -> list[dict]:
        drained = self._pending_transitions
        self._pending_transitions = []
        return drained

    def finish_episode(self, success: bool = False) -> None:
        """Close any dangling pending transition as the terminal one."""
        if self._pending is not None:
            self._close_pending(next_state=None, done=True, success=bool(success))

    def finalize(self, terminal_success: bool) -> None:
        """Re-attribute the exact task reward to the final transition,
        replacing the provisional success-state heuristic used at solve time
        (the runner only knows the exact reward after task.test_output_modify)."""
        if not self._all_transitions:
            return
        final = self._all_transitions[-1]
        if not final["done"]:
            return
        tokens, latency = self._last_terminal_costs
        final["reward"] = float(
            correctness_reward(bool(terminal_success))
            - token_penalty(tokens, self.max_tokens)
            - latency_penalty(latency, self.budget_seconds)
        )
        if self.recorder is not None:
            self.recorder.replace_last(final)

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
            self._close_pending(next_state=decision_state.state, done=False, success=False)
            self._open_pending(decision_state.state, decision_state.action)
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

        self._add_costs(tokens=step_tokens, latency=step_latency + eval_seconds)
        self._cumulative_tokens += step_tokens
        self._cumulative_latency += step_latency + eval_seconds

        if stopped:
            self._close_pending(next_state=None, done=True, success=True)
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
