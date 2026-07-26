"""Task assignment functions for federated MCTS — legacy interface."""

from typing import List, TypedDict


class TaskAssignment(TypedDict):
    solve_client: str
    eval_client: str
    ys: List[str]


def naive_assign_task(model_list: List[str], ys: List[str]) -> List[TaskAssignment]:
    """Distribute candidates evenly among clients. Each client evals its own."""
    if not model_list:
        return []

    assignments: List[TaskAssignment] = []
    for client_name in model_list:
        assignments.append(TaskAssignment(
            solve_client=client_name, eval_client=client_name, ys=[]
        ))

    for i, y in enumerate(ys):
        client_idx = i % len(model_list)
        assignments[client_idx]["ys"].append(y)

    return assignments


def speculative_federated_assign_task(model_list: List[str], ys: List[str]) -> List[TaskAssignment]:
    """Distribute candidates evenly, all eval on remote_client."""
    if not model_list:
        return []

    assignments: List[TaskAssignment] = []
    for client_name in model_list:
        assignments.append(TaskAssignment(
            solve_client=client_name, eval_client="remote_client", ys=[]
        ))

    for i, y in enumerate(ys):
        client_idx = i % len(model_list)
        assignments[client_idx]["ys"].append(y)

    return assignments
