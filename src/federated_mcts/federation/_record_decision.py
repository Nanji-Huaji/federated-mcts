"""Task-recorder integration for completed baseline solver steps."""

from __future__ import annotations

import time


def record_baseline_decision(
    task_recorder,
    step: int,
    ys: list[str],
    candidate_batches: list[list[str]],
    new_ys: list[str],
    values: list[float],
    select_ids: list[int],
    selected: list[str],
    started_at: float,
    task,
    x: str,
    generation_recorder,
    vote_recorder,
) -> None:
    """Persist one completed baseline choice with lineage parent references."""
    parent_lineage_ids = (
        [""] * len(ys) if step == 0
        else task_recorder.selected_lineage_ids)
    candidate_parent_lineage_ids = [
        parent_lineage_id
        for parent_lineage_id, candidates in zip(
            parent_lineage_ids, candidate_batches)
        for _ in candidates
    ]
    task_recorder.record_decision(
        depth=step,
        parent_count=len(ys),
        candidates=new_ys,
        candidate_parent_lineage_ids=candidate_parent_lineage_ids,
        values=values,
        selected_indices=select_ids,
        selected=selected,
        is_stopping=step + 1 >= task.steps,
        elapsed_ms=(time.time() - started_at) * 1000,
        task=task,
        x=x,
        gen_rec=generation_recorder,
        vote_rec=vote_recorder,
    )
