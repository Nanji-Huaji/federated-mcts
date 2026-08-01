import itertools
import time
from federated_mcts.utils.exhaustive import assert_never

import numpy as np

from federated_mcts.core.diverse_search import DiverseSearch
from federated_mcts.core.dqn.factory import build_dqn_session
from federated_mcts.core.evaluation import get_values, get_votes
from federated_mcts.core.generation import get_proposals, get_samples


class StandardSolversMixin:
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
            match self.args.method_generate:
                case "sample":
                    new_ys = [
                        get_samples(
                            self.args, task, x, y, self.args.n_generate_sample,
                            prompt_sample=self.args.prompt_sample, stop=task.stops[step],
                            client=gpt_fn,
                        )
                        for y in ys
                    ]
                case "propose":
                    new_ys = [get_proposals(self.args, step, task, x, y, client=gpt_fn) for y in ys]
                case unreachable:
                    assert_never(unreachable)
            new_ys = list(itertools.chain(*new_ys))
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
                if outcome.stopped:
                    break
                continue

            match self.args.method_evaluate:
                case "vote":
                    values = get_votes(self.args, task, x, new_ys, self.args.n_evaluate_sample, client=gpt_fn)
                case "value":
                    values = get_values(self.args, task, x, new_ys, self.args.n_evaluate_sample, client=gpt_fn)
                case unreachable:
                    assert_never(unreachable)

            match self.args.method_select:
                case "sample" if sum(values) > 0:
                    probabilities = np.array(values) / sum(values)
                    select_ids = np.random.choice(ids, size=self.args.n_select_sample, p=probabilities).tolist()
                case "sample" | "greedy":
                    select_ids = sorted(ids, key=lambda index: values[index], reverse=True)[:self.args.n_select_sample]
                case unreachable:
                    assert_never(unreachable)
            select_new_ys = [new_ys[select_id] for select_id in select_ids]

            if to_print:
                sorted_new_ys, sorted_values = zip(*sorted(zip(new_ys, values), key=lambda item: item[1], reverse=True))
                print(f"-- new_ys --: {sorted_new_ys}\n-- sol values --: {sorted_values}\n-- choices --: {select_new_ys}\n")

            infos.append({
                "step": step, "x": x, "ys": ys, "new_ys": new_ys,
                "values": values, "select_new_ys": select_new_ys,
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
            match self.args.method_generate:
                case "sample":
                    new_ys = [
                        get_samples(
                            self.args, task, x, y, self.args.n_generate_sample,
                            prompt_sample=self.args.prompt_sample, stop=task.stops[step],
                            client=solve_gpt,
                        )
                        for y in ys
                    ]
                case "propose":
                    new_ys = [get_proposals(self.args, step, task, x, y, client=solve_gpt) for y in ys]
                case unreachable:
                    assert_never(unreachable)
            new_ys = list(itertools.chain(*new_ys))
            ids = list(range(len(new_ys)))
            self.time_tracker.record_generation("speculative", time.time() - start_time)

            start_time = time.time()
            match self.args.method_evaluate:
                case "vote":
                    values = get_votes(self.args, task, x, new_ys, self.args.n_evaluate_sample, client=eval_gpt)
                case "value":
                    values = get_values(self.args, task, x, new_ys, self.args.n_evaluate_sample, client=eval_gpt)
                case unreachable:
                    assert_never(unreachable)
            self.time_tracker.record_evaluation("speculative", time.time() - start_time)

            match self.args.method_select:
                case "sample" if sum(values) > 0:
                    probabilities = np.array(values) / sum(values)
                    select_ids = np.random.choice(ids, size=self.args.n_select_sample, p=probabilities).tolist()
                case "sample" | "greedy":
                    select_ids = sorted(ids, key=lambda index: values[index], reverse=True)[:self.args.n_select_sample]
                case unreachable:
                    assert_never(unreachable)
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
