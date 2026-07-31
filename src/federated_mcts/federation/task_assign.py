"""Task assignment strategies for federated MCTS.

Strategies determine which model handles which candidate solutions at each step.
All strategies follow the BaseAssignStrategy interface with assign() + update().

Built-in strategies:
- RoundRobinStrategy          — evenly distribute candidates (baseline / legacy)
- DifficultyBasedStrategy     — step-aware: more local on easy steps, more remote on hard
- ContextualBanditStrategy    — UCB-based online learning per (model, step)

Usage:
    strategy = get_strategy("difficulty", total_steps=3)
    assignments = strategy.assign(models, ys, context)
    # ... run step ...
    strategy.update(context, per_model_rewards)
"""

import math
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict, Union

import numpy as np


# ── Types ─────────────────────────────────────────────────────────────

class TaskAssignment(TypedDict):
    solve_client: str
    eval_client: str
    ys: List[str]


class AssignmentContext(TypedDict, total=False):
    """Context passed to assign() at each MCTS step."""
    step: int
    total_steps: int
    task_name: str
    prev_values: Optional[List[float]]  # evaluation scores from previous step
    n_candidates: int


# ── Base ───────────────────────────────────────────────────────────────

class BaseAssignStrategy(ABC):
    """Abstract base for all assignment strategies."""

    @abstractmethod
    def assign(
        self,
        model_list: List[str],
        ys: List[str],
        context: AssignmentContext,
    ) -> List[TaskAssignment]:
        """Produce per-model assignments for the current MCTS step."""
        ...

    def update(self, context: AssignmentContext, rewards: Dict[str, float]) -> None:
        """Optional: update internal state after step completes."""
        pass

    def reset(self) -> None:
        """Reset internal state (e.g., between experiments)."""
        pass


# ── Round-Robin (Baseline) ────────────────────────────────────────────

class RoundRobinStrategy(BaseAssignStrategy):
    """Evenly distribute candidates among all models. Same as old naive_assign_task."""

    def __init__(self, eval_client: Optional[str] = None, **kwargs):
        self.eval_client = eval_client

    def assign(
        self,
        model_list: List[str],
        ys: List[str],
        context: AssignmentContext,
    ) -> List[TaskAssignment]:
        if not model_list:
            return []

        assignments: List[TaskAssignment] = []
        for client_name in model_list:
            assignments.append(TaskAssignment(
                solve_client=client_name,
                eval_client=self.eval_client if self.eval_client else client_name,
                ys=[],
            ))

        if len(ys) == 1 and ys[0] == "":
            # Sole root candidate: replicate to every model so each
            # independently explores from the root.
            for a in assignments:
                a["ys"] = [ys[0]]
        else:
            for i, y in enumerate(ys):
                client_idx = i % len(model_list)
                assignments[client_idx]["ys"].append(y)

        return assignments


# ── Difficulty-Based ──────────────────────────────────────────────────

class DifficultyBasedStrategy(BaseAssignStrategy):
    """Route candidates based on MCTS depth: easy steps -> local, hard steps -> remote.

    Assumes model_list is ordered: [local_model, remote_model].
    Difficulty = step / (total_steps - 1): 0 = easy (early exploration), 1 = hard (final answer).

    Args:
        total_steps: total MCTS steps for the task (e.g., 3 for Game24)
        local_ratio_at_easy: fraction of candidates to local model at step 0 (default 0.8)
        local_ratio_at_hard: fraction of candidates to local model at last step (default 0.0)
        local_is_eval: whether the local model also evaluates (default False)
    """

    def __init__(
        self,
        total_steps: int = 3,
        local_ratio_at_easy: float = 0.8,
        local_ratio_at_hard: float = 0.0,
        local_is_eval: bool = False,
    ):
        self.total_steps = total_steps
        self.local_easy = local_ratio_at_easy
        self.local_hard = local_ratio_at_hard
        self.local_is_eval = local_is_eval

    def assign(
        self,
        model_list: List[str],
        ys: List[str],
        context: AssignmentContext,
    ) -> List[TaskAssignment]:
        if not model_list:
            return []
        if not ys:
            return []

        if len(ys) == 1 and ys[0] == "":
            remote_model = model_list[1] if len(model_list) >= 2 else model_list[0]
            eval_client = model_list[0] if self.local_is_eval else remote_model
            assignments = []
            for client_name in model_list:
                assignments.append(TaskAssignment(
                    solve_client=client_name,
                    eval_client=eval_client,
                    ys=[ys[0]],
                ))
            return assignments

        step = context.get("step", 0)
        total = context.get("total_steps", self.total_steps)

        # Linear interpolation: step=0 -> local_easy, step=total-1 -> local_hard
        if total <= 1:
            alpha = 0.0
        else:
            alpha = step / (total - 1)
        local_ratio = self.local_easy + alpha * (self.local_hard - self.local_easy)
        local_ratio = max(0.0, min(1.0, local_ratio))

        local_model = model_list[0] if len(model_list) >= 1 else None
        remote_model = model_list[1] if len(model_list) >= 2 else model_list[0]

        n_local = max(1, round(len(ys) * local_ratio)) if local_ratio > 0 else 0
        n_local = min(n_local, len(ys))

        eval_client = local_model if self.local_is_eval else remote_model

        assignments: List[TaskAssignment] = []
        if local_model and n_local > 0:
            assignments.append(TaskAssignment(
                solve_client=local_model,
                eval_client=eval_client,
                ys=ys[:n_local],
            ))
        if remote_model and n_local < len(ys):
            assignments.append(TaskAssignment(
                solve_client=remote_model,
                eval_client=remote_model,
                ys=ys[n_local:],
            ))

        return assignments


# ── Contextual Bandit ─────────────────────────────────────────────────

class ContextualBanditStrategy(BaseAssignStrategy):
    """UCB-based online learning for per-step model selection.

    Maintains a bandit arm for each (model, step) pair. Uses UCB to balance
    exploration vs exploitation. Reward = average evaluation score from that step.

    Args:
        total_steps: total MCTS steps for the task
        exploration_weight: UCB exploration constant C (default 2.0)
        min_samples: minimum pulls before using UCB; random before that
        decay_factor: exponentially decay old rewards (0=no decay)
    """

    def __init__(
        self,
        total_steps: int = 3,
        exploration_weight: float = 2.0,
        min_samples: int = 5,
        decay_factor: float = 0.1,
    ):
        self.total_steps = total_steps
        self.C = exploration_weight
        self.min_samples = min_samples
        self.decay = decay_factor

        # Per (model, step): {count, sum_reward}
        self._arms: Dict[Tuple[str, int], Dict[str, float]] = defaultdict(
            lambda: {"count": 0.0, "sum": 0.0}
        )
        self._total_pulls = 0.0

    def _ucb(self, model: str, step: int) -> float:
        arm = self._arms[(model, step)]
        if arm["count"] < self.min_samples:
            return float("inf")  # explore unknown arms first
        avg = arm["sum"] / arm["count"]
        bonus = self.C * math.sqrt(2.0 * math.log(max(1.0, self._total_pulls)) / arm["count"])
        return avg + bonus

    def assign(
        self,
        model_list: List[str],
        ys: List[str],
        context: AssignmentContext,
    ) -> List[TaskAssignment]:
        if not model_list:
            return []
        if not ys:
            return []

        if len(ys) == 1 and ys[0] == "":
            remote_model = model_list[1] if len(model_list) >= 2 else model_list[0]
            assignments = []
            for client_name in model_list:
                assignments.append(TaskAssignment(
                    solve_client=client_name,
                    eval_client=remote_model,
                    ys=[ys[0]],
                ))
            return assignments

        step = context.get("step", 0)

        # Compute UCB scores for each model at this step
        scores = {m: self._ucb(m, step) for m in model_list}

        # Softmax -> allocation proportions
        finite_scores = {m: s for m, s in scores.items() if s != float("inf")}
        if not finite_scores:
            # All unknown -> uniform
            props = {m: 1.0 / len(model_list) for m in model_list}
        else:
            max_score = max(finite_scores.values())
            exps = {}
            for m in model_list:
                s = scores[m]
                if s == float("inf"):
                    exps[m] = math.exp(10.0)  # give exploring arms a strong boost
                else:
                    exps[m] = math.exp(s - max_score)
            total_exp = sum(exps.values())
            props = {m: e / total_exp for m, e in exps.items()}

        # Allocate ys proportionally
        remote_model = model_list[1] if len(model_list) >= 2 else model_list[0]
        n_total = len(ys)
        assignments: List[TaskAssignment] = []
        offset = 0

        # Ensure minimum allocation per model to prevent lockout
        min_per_model = max(1, n_total // len(model_list))
        for i, model in enumerate(model_list):
            if i == len(model_list) - 1:
                n_model = n_total - offset
            else:
                n_model = max(min_per_model, round(n_total * props[model]))
                n_model = min(n_model, n_total - offset - min_per_model * (len(model_list) - 1 - i))
            n_model = min(n_model, n_total - offset)

            if n_model > 0:
                assignments.append(TaskAssignment(
                    solve_client=model,
                    eval_client=remote_model,
                    ys=ys[offset:offset + n_model],
                ))
                offset += n_model

        return assignments

    def update(self, context: AssignmentContext, rewards: Dict[str, float]) -> None:
        """Update bandit with per-model reward from the completed step."""
        step = context.get("step", 0)

        if self.decay > 0:
            for key in self._arms:
                self._arms[key]["count"] *= (1.0 - self.decay)
                self._arms[key]["sum"] *= (1.0 - self.decay)
            self._total_pulls *= (1.0 - self.decay)

        for model, reward in rewards.items():
            arm = self._arms[(model, step)]
            arm["count"] += 1.0
            arm["sum"] += reward
            self._total_pulls += 1.0

    def get_stats(self) -> Dict[str, Any]:
        """Return bandit statistics for inspection."""
        stats = {}
        for (model, step), arm in self._arms.items():
            key = f"{model}_step{step}"
            avg = arm["sum"] / arm["count"] if arm["count"] > 0 else 0.0
            stats[key] = {"count": int(arm["count"]), "avg_reward": round(avg, 4)}
        return stats

    def reset(self) -> None:
        self._arms.clear()
        self._total_pulls = 0.0


# ── Strategy Factory ───────────────────────────────────────────────────

STRATEGY_REGISTRY: Dict[str, type] = {
    "round_robin": RoundRobinStrategy,
    "difficulty": DifficultyBasedStrategy,
    "bandit": ContextualBanditStrategy,
}


def get_strategy(name: str, **kwargs) -> BaseAssignStrategy:
    """Create a strategy by name. Extra kwargs passed to constructor.

    Available: round_robin, difficulty, bandit
    """
    if name not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown strategy: {name}. Available: {list(STRATEGY_REGISTRY.keys())}"
        )
    return STRATEGY_REGISTRY[name](**kwargs)


# ── Legacy Compatibility ──────────────────────────────────────────────

_default_rr = RoundRobinStrategy()
_default_spec = RoundRobinStrategy(eval_client="remote_client")


def naive_assign_task(
    model_list: List[str], ys: List[str], context: Optional[AssignmentContext] = None
) -> List[TaskAssignment]:
    """Legacy: evenly distribute, each model evals its own candidates."""
    ctx = context if context else {}
    return _default_rr.assign(model_list, ys, ctx)


def speculative_federated_assign_task(
    model_list: List[str], ys: List[str], context: Optional[AssignmentContext] = None
) -> List[TaskAssignment]:
    """Legacy: evenly distribute, all eval on remote_client."""
    ctx = context if context else {}
    return _default_spec.assign(model_list, ys, ctx)
