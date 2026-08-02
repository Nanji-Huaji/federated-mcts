import itertools
import time
from typing import Literal, Protocol
from federated_mcts.utils.exhaustive import assert_never

import numpy as np

from federated_mcts.core.diverse_search import DiverseSearch
from federated_mcts.core.dqn.factory import build_dqn_session
from federated_mcts.core.evaluation import get_values, get_votes
from federated_mcts.core.generation import get_proposals, get_samples
from federated_mcts.federation.time_tracker import TimeTracker


class StandardSolverArgs(Protocol):
    """CLI options consumed by the standard solver methods."""

    n_generate_sample: int
    prompt_sample: Literal["standard", "cot"] | None
    search_policy: Literal["baseline", "diverse", "dqn"]
    method_generate: Literal["sample", "propose"]
    game24_exact_prune: bool
    method_evaluate: Literal["vote", "value"]
    n_evaluate_sample: int
    method_select: Literal["sample", "greedy"]
    n_select_sample: int
    warm_start: bool
    last_lm: bool
    slm_eval: bool


class GenerationClient(Protocol):
    """Callable interface shared by remote and local generation clients."""

    def __call__(
        self,
        args: StandardSolverArgs,
        prompt: str,
        n: int = 1,
        stop: str | list[str] | None = None,
        get_logprobs: bool = False,
    ) -> list[str]: ...


class StandardSolversMixin:
    args: StandardSolverArgs
    gpts: dict[str, GenerationClient]
    time_tracker: TimeTracker

    def naive_solve(self, task, idx, to_print=True, solve_client="remote_client", **kwargs):
        x = task.get_input(idx)
        ys = get_samples(
            self.args, task, x, "", self.args.n_generate_sample,
            self.args.prompt_sample, stop=None,
        )
        return ys, {}

    def solve(self, task, idx, to_print=True, solve_client="remote_client", **kwargs):
        gpt_fn = self.gpts[solve_client]
        x = task.get_input(idx)
        ys = [""]
        infos = []
        search_policy = getattr(self.args, "search_policy", "baseline")
        diverse_search = DiverseSearch(self.args) if search_policy == "diverse" else None
        dqn_session = build_dqn_session(self.args) if search_policy == "dqn" else None
        self.dqn_session = dqn_session
        for step in range(task.steps):
            candidate_batches: list[list[str]]
            solve_method_generate: Literal["sample", "propose"] = self.args.method_generate
            match solve_method_generate:
                case "sample":
                    candidate_batches = [
                        get_samples(
                            self.args, task, x, y, self.args.n_generate_sample,
                            prompt_sample=self.args.prompt_sample, stop=task.stops[step],
                            client=gpt_fn,
                        )
                        for y in ys
                    ]
                case "propose":
                    candidate_batches = [get_proposals(self.args, step, task, x, y, client=gpt_fn) for y in ys]
                case unreachable_generate:
                    assert_never(unreachable_generate)
            new_ys = list(itertools.chain.from_iterable(candidate_batches))
            # ---- Game24 exact dead-end pruning (non-DQN only) ----
            prune_metrics: dict[str, int] = {}
            if getattr(self.args, "game24_exact_prune", False) and dqn_session is None:
                from federated_mcts.tasks.game24 import Game24Task
                if isinstance(task, Game24Task):
                    from federated_mcts.tasks.game24_exact_prune import prune_unreachable
                    prune_proposed = len(new_ys)
                    new_ys, _, prune_pruned = prune_unreachable(new_ys, task, x)
                    prune_metrics = {
                        "exact_prune_proposed": prune_proposed,
                        "exact_prune_pruned": prune_pruned,
                        "exact_prune_retained": len(new_ys),
                    }
            if not new_ys:
                break
            ids = list(range(len(new_ys)))

            if diverse_search is not None:
                raw_candidate_count = len(new_ys)
                decision = diverse_search.decide(
                    self.args, task, x, new_ys, gpt_fn, solve_client,
                )
                infos.append({
                    "step": step, "x": x, "ys": ys, "new_ys": decision.candidates,
                    "values": decision.values, "select_new_ys": decision.selected,
                    "search_metrics": {
                        "raw_candidates": raw_candidate_count,
                        "unique_states": len(decision.candidates),
                        "beam_width": len(decision.selected),
                        **prune_metrics,
                    },
                })
                ys = decision.selected
                if decision.stopped:
                    break
                continue

            if dqn_session is not None and new_ys:
                outcome = dqn_session.process_step(
                    args=self.args,
                    task=task,
                    x=x,
                    candidates=new_ys,
                    client=gpt_fn,
                    evaluator_id=solve_client,
                    step=step,
                    total_steps=task.steps,
                )
                step_info = {
                    "step": step, "x": x, "ys": ys, "new_ys": outcome.candidates,
                    "values": outcome.values, "select_new_ys": outcome.selected,
                    "search_metrics": outcome.search_metrics,
                }
                step_info["dqn_transitions"] = outcome.transitions
                infos.append(step_info)
                ys = outcome.selected
                if outcome.stopped or not ys:
                    break
                continue

            solve_method_evaluate: Literal["vote", "value"] = self.args.method_evaluate
            match solve_method_evaluate:
                case "vote":
                    values = get_votes(self.args, task, x, new_ys, self.args.n_evaluate_sample, client=gpt_fn)
                case "value":
                    values = get_values(self.args, task, x, new_ys, self.args.n_evaluate_sample, client=gpt_fn)
                case unreachable_evaluate:
                    assert_never(unreachable_evaluate)

            solve_method_select: Literal["sample", "greedy"] = self.args.method_select
            match solve_method_select:
                case "sample" if sum(values) > 0:
                    probabilities = np.array(values) / sum(values)
                    select_ids = np.random.choice(ids, size=self.args.n_select_sample, p=probabilities).tolist()
                case "sample" | "greedy":
                    select_ids = sorted(ids, key=lambda index: values[index], reverse=True)[:self.args.n_select_sample]
                case unreachable_select:
                    assert_never(unreachable_select)
            select_new_ys = [new_ys[select_id] for select_id in select_ids]

            if to_print:
                sorted_new_ys, sorted_values = zip(*sorted(zip(new_ys, values), key=lambda item: item[1], reverse=True))
                print(f"-- new_ys --: {sorted_new_ys}\n-- sol values --: {sorted_values}\n-- choices --: {select_new_ys}\n")

            infos.append({
                "step": step, "x": x, "ys": ys, "new_ys": new_ys,
                "values": values, "select_new_ys": select_new_ys,
                "search_metrics": prune_metrics,
            })
            ys = select_new_ys

        if dqn_session is not None:
            dqn_session.finish_episode()
            remaining = dqn_session.drain_transitions()
            if remaining and infos:
                infos[-1].setdefault("dqn_transitions", []).extend(remaining)

        if to_print:
            print(ys)
        return ys, {"steps": infos}

    def speculative_solve(self, task, idx, to_print=True, solve_client=None, eval_client=None, **kwargs):
        local_client = self.gpts["local_client"] if solve_client is None else self.gpts[solve_client]
        remote_client = self.gpts["remote_client"] if eval_client is None else self.gpts[eval_client]
        print(f"local_client: {local_client}, remote_client: {remote_client}")

        x = task.get_input(idx)
        ys = [""]
        infos = []

        for step in range(task.steps):
            solve_gpt = local_client
            eval_gpt = remote_client

            if self.args.warm_start and step == 0:
                solve_gpt = remote_client
                eval_gpt = remote_client
            elif step + 1 == task.steps and self.args.last_lm:
                solve_gpt = remote_client
                eval_gpt = remote_client

            if self.args.slm_eval:
                eval_gpt = local_client
            print(f"Step {step} of {task.steps} in Task {idx}, solve_client: {solve_gpt}, eval_client: {eval_gpt}")

            start_time = time.time()
            candidate_batches: list[list[str]]
            speculative_method_generate: Literal["sample", "propose"] = self.args.method_generate
            match speculative_method_generate:
                case "sample":
                    candidate_batches = [
                        get_samples(
                            self.args, task, x, y, self.args.n_generate_sample,
                            prompt_sample=self.args.prompt_sample, stop=task.stops[step],
                            client=solve_gpt,
                        )
                        for y in ys
                    ]
                case "propose":
                    candidate_batches = [get_proposals(self.args, step, task, x, y, client=solve_gpt) for y in ys]
                case unreachable_generate_speculative:
                    assert_never(unreachable_generate_speculative)
            new_ys = list(itertools.chain.from_iterable(candidate_batches))
            ids = list(range(len(new_ys)))
            self.time_tracker.record_generation("speculative", time.time() - start_time)

            start_time = time.time()
            speculative_method_evaluate: Literal["vote", "value"] = self.args.method_evaluate
            match speculative_method_evaluate:
                case "vote":
                    values = get_votes(self.args, task, x, new_ys, self.args.n_evaluate_sample, client=eval_gpt)
                case "value":
                    values = get_values(self.args, task, x, new_ys, self.args.n_evaluate_sample, client=eval_gpt)
                case unreachable_evaluate_speculative:
                    assert_never(unreachable_evaluate_speculative)
            self.time_tracker.record_evaluation("speculative", time.time() - start_time)

            speculative_method_select: Literal["sample", "greedy"] = self.args.method_select
            match speculative_method_select:
                case "sample" if sum(values) > 0:
                    probabilities = np.array(values) / sum(values)
                    select_ids = np.random.choice(ids, size=self.args.n_select_sample, p=probabilities).tolist()
                case "sample" | "greedy":
                    select_ids = sorted(ids, key=lambda index: values[index], reverse=True)[:self.args.n_select_sample]
                case unreachable_select_speculative:
                    assert_never(unreachable_select_speculative)
            select_new_ys = [new_ys[select_id] for select_id in select_ids]

            if to_print:
                sorted_new_ys, sorted_values = zip(*sorted(zip(new_ys, values), key=lambda item: item[1], reverse=True))
                print(f"-- new_ys --: {sorted_new_ys}\n-- sol values --: {sorted_values}\n-- choices --: {select_new_ys}\n")

            infos.append({
                "step": step, "x": x, "ys": ys, "new_ys": new_ys,
                "values": values, "select_new_ys": select_new_ys,
            })
            ys = select_new_ys

        if to_print:
            print(ys)
        return ys, {"steps": infos}
