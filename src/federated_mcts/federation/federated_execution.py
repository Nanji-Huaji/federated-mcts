import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from federated_mcts.utils.exhaustive import assert_never

from federated_mcts.core.evaluation import get_values, get_votes
from federated_mcts.core.generation import get_proposals, get_samples
from federated_mcts.utils.uncertainty import Uncertainty


class FederatedExecutionMixin:
    def _run_assignments(self, task, x, step, task_assignments, to_print, n_gen):
        all_new_ys = []
        all_values = []
        step_infos = []
        step_client_times = {}

        jobs = [(index, assignment) for index, assignment in enumerate(task_assignments) if assignment["ys"]]
        if not jobs:
            return all_new_ys, all_values, step_infos, step_client_times

        positions = {original: dense for dense, (original, _) in enumerate(jobs)}
        futures = {}
        with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
            for original, assignment in jobs:
                future = executor.submit(
                    self._client_step,
                    task, x, assignment["ys"], step,
                    assignment["solve_client"], assignment["eval_client"],
                    to_print, n_generate_sample=n_gen,
                )
                futures[future] = positions[original]

            results = [None] * len(jobs)
            for future in as_completed(futures):
                results[futures[future]] = future.result()

        for (_, assignment), result in zip(jobs, results):
            client_new_ys, client_values, client_info, client_times = result
            all_new_ys.extend(client_new_ys)
            all_values.extend(client_values)
            step_infos.append(client_info)
            solve_client = assignment["solve_client"]
            eval_client = assignment["eval_client"]
            step_client_times.setdefault(solve_client, {"generation": 0.0, "evaluation": 0.0})
            step_client_times.setdefault(eval_client, {"generation": 0.0, "evaluation": 0.0})
            step_client_times[solve_client]["generation"] += client_times["generation"]
            step_client_times[eval_client]["evaluation"] += client_times["evaluation"]

        return all_new_ys, all_values, step_infos, step_client_times

    def _client_step(
        self, task, x, client_ys, step, solve_client_name, eval_client_name,
        to_print=False, n_generate_sample=None, uncertainty_backoff=False,
        uncertainty_threshold=0.8,
    ):
        uncertainty_calculator = Uncertainty(
            uncertainty_threshold=uncertainty_threshold,
            uncertainty_method="entropy",
        )
        solve_gpt = self.gpts[solve_client_name]
        eval_gpt = self.gpts[eval_client_name]
        new_ys = []
        n_gen = n_generate_sample if n_generate_sample else self.args.n_generate_sample

        generation_start = time.time()
        for y in client_ys:
            match self.args.method_generate:
                case "sample":
                    result = get_samples(
                        self.args, task, x, y, n_generate_sample=n_gen,
                        prompt_sample=self.args.prompt_sample, stop=task.stops[step],
                        client=solve_gpt, get_logprobs=uncertainty_backoff,
                    )
                    if uncertainty_backoff:
                        samples, logprobs = result
                        uncertainty_calculator.calculate_uncertainty(logprobs)
                        samples_to_add = samples
                    else:
                        samples_to_add = result
                    new_ys.extend(samples_to_add if isinstance(samples_to_add, list) else [samples_to_add])
                case "propose":
                    proposals = get_proposals(
                        self.args, step, task, x, y, client=solve_gpt,
                        get_logprobs=uncertainty_backoff, n_generate=n_gen,
                    )
                    if uncertainty_backoff and isinstance(proposals, tuple):
                        proposals = proposals[0]
                    new_ys.extend(proposals)
                case unreachable:
                    assert_never(unreachable)
        generation_time = time.time() - generation_start

        new_ys = list(dict.fromkeys(new_ys))
        evaluation_start = time.time()
        if getattr(self.args, "search_policy", "baseline") in ("diverse", "dqn"):
            values = [0.0] * len(new_ys)
        elif new_ys and self.args.method_evaluate == "vote":
            values = get_votes(self.args, task, x, new_ys, self.args.n_evaluate_sample, client=eval_gpt)
        elif new_ys and self.args.method_evaluate == "value":
            values = get_values(self.args, task, x, new_ys, self.args.n_evaluate_sample, client=eval_gpt)
        elif new_ys:
            values = [1.0] * len(new_ys)
        else:
            values = []
        evaluation_time = time.time() - evaluation_start

        client_info = {
            "solve_client_name": solve_client_name, "eval_client_name": eval_client_name,
            "input_ys": client_ys, "output_ys": new_ys, "values": values, "step": step,
        }
        client_times = {"generation": generation_time, "evaluation": evaluation_time}

        if to_print:
            print(
                f"Client {solve_client_name} (eval: {eval_client_name}) "
                f"generated {len(new_ys)} proposals with values {values[:5]}"
            )

        return new_ys, values, client_info, client_times
