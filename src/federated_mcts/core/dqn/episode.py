"""DQN episode transition lifecycle and recorder coordination.

A transition is the unit a trainer consumes.  Non-terminal transitions are
published as soon as their costs are known.  The terminal transition is the
only one whose correctness component depends on the task's exact verifier, so
it is held unpublished until :meth:`DqnEpisode.finalize` receives that verdict.
This keeps the JSONL recorder append-only: a terminal record is written once,
with the reward the caller will report, instead of being written provisionally
and rewritten afterwards.
"""

from __future__ import annotations

from federated_mcts.core.dqn.actions import beam_and_joint_rank
from federated_mcts.core.dqn.controller import _plain
from federated_mcts.core.dqn.recorder import JSONLTransitionRecorder
from federated_mcts.core.dqn.transition_rewards import build_reward_components


class DqnEpisode:
    """Own pending transition state and the transitions emitted for one episode."""

    def __init__(
        self,
        *,
        max_tokens: int,
        budget_seconds: float,
        distance_scale: float,
        jsonl_path: str | None,
    ):
        self.max_tokens = max_tokens
        self.budget_seconds = budget_seconds
        self.distance_scale = distance_scale
        self.recorder = JSONLTransitionRecorder(jsonl_path) if jsonl_path else None
        self.pending: dict | None = None
        self.all_transitions: list[dict] = []
        self.pending_transitions: list[dict] = []
        self.last_terminal_costs = (0.0, 0.0)
        self._terminal_transition: dict | None = None
        self._terminal_success = False
        self._terminal_published = False

    def open(self, state, action: int, distance_before: int | None) -> None:
        beam, joint_rank = beam_and_joint_rank(action)
        self.pending = {
            "state": _plain(state), "action": int(action), "beam": int(beam),
            "joint_rank": bool(joint_rank), "tokens": 0.0, "latency": 0.0,
            "distance_before": distance_before, "distance_after": None,
        }

    def add_costs(self, *, tokens: float, latency: float) -> None:
        if self.pending is not None:
            self.pending["tokens"] += float(tokens)
            self.pending["latency"] += float(latency)

    def close(self, *, next_state, done: bool, success: bool) -> dict | None:
        if self.pending is None:
            return None
        pending = self.pending
        self.pending = None
        components = build_reward_components(
            success=success, done=done,
            distance_before=pending["distance_before"], distance_after=pending["distance_after"],
            distance_scale=self.distance_scale, tokens=pending["tokens"], latency=pending["latency"],
            max_tokens=self.max_tokens, budget_seconds=self.budget_seconds,
        )
        before, after = pending["distance_before"], pending["distance_after"]
        transition = {
            "state": pending["state"], "action": pending["action"],
            "reward": float(components["total"]),
            "next_state": None if next_state is None else _plain(next_state),
            "done": bool(done), "beam": pending["beam"], "joint_rank": pending["joint_rank"],
        }
        if before is not None or after is not None:
            transition.update({
                "distance_before": before, "distance_after": after,
                "distance_delta": None if before is None or after is None else before - after,
                "distance_reward": components["distance"],
                "correctness_reward": components["correctness"],
                "token_penalty": components["token_penalty"],
                "latency_penalty": components["latency_penalty"],
                 "reward_components": {
                     "correctness": components["correctness"],
                     "distance": components["distance"],
                     "token_penalty": components["token_penalty"],
                     "latency_penalty": components["latency_penalty"],
                 },
            })
        self.all_transitions.append(transition)
        self.pending_transitions.append(transition)
        if done:
            self.last_terminal_costs = (pending["tokens"], pending["latency"])
            self._terminal_transition = transition
            self._terminal_success = bool(success)
            self._terminal_published = False
        elif self.recorder is not None:
            self.recorder.record(transition)
        return transition

    def drain(self) -> list[dict]:
        transitions, self.pending_transitions = self.pending_transitions, []
        return transitions

    def finish(self, success: bool, *, flush_terminal: bool = True) -> None:
        """Close a dangling transition and (optionally) publish the terminal one.

        ``flush_terminal=False`` is used by ``finish_task``: the terminal
        record stays in memory until :meth:`finalize` receives the exact
        verification verdict, so the recorder never needs a rewrite.
        """
        if self.pending is not None:
            self.close(next_state=None, done=True, success=success)
        if flush_terminal:
            self._publish_terminal(self._terminal_success)

    def finalize(self, success: bool) -> None:
        """Publish the terminal transition once, with the exact task reward."""
        self._publish_terminal(bool(success))

    def _publish_terminal(self, success: bool) -> None:
        transition = self._terminal_transition
        if transition is None or self._terminal_published:
            return
        components = transition.get("reward_components")
        before = transition.get("distance_before") if components is not None else None
        after = transition.get("distance_after") if components is not None else None
        tokens, latency = self.last_terminal_costs
        reward = build_reward_components(
            success=success, done=True, distance_before=before, distance_after=after,
            distance_scale=self.distance_scale, tokens=tokens, latency=latency,
            max_tokens=self.max_tokens, budget_seconds=self.budget_seconds,
        )
        transition["reward"] = float(reward["total"])
        if components is not None:
            transition["correctness_reward"] = reward["correctness"]
            transition["token_penalty"] = reward["token_penalty"]
            transition["latency_penalty"] = reward["latency_penalty"]
            transition["reward_components"] = {
                "correctness": reward["correctness"],
                "distance": reward["distance"],
                "token_penalty": reward["token_penalty"],
                "latency_penalty": reward["latency_penalty"],
            }
        if self.recorder is not None:
            self.recorder.record(transition)
        self._terminal_published = True
