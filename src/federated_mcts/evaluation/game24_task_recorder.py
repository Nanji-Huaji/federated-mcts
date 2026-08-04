"""Task-level recorder: lineage, telemetry, audit, dispositions.
Six exhaustive dispositions: none, selected, expanded,
terminal_solved, terminal_unsolved, terminal_invalid.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from federated_mcts.evaluation.game24_decision_helpers import (
    _SolverTask,
    apply_expanded_disposition,
    build_failed_step_events,
    compute_decision_audit,
    detect_disposition,
)
from federated_mcts.evaluation.game24_instrumentation import (
    GenerationRecorder,
    VoteRecorder,
)
from federated_mcts.evaluation.game24_metrics import (
    build_lineage_id,
    make_candidate_event,
    make_step_metrics,
    make_task_metrics,
)
from federated_mcts.evaluation.game24_metrics_types import (
    CandidateEvent,
    TaskCandidateMetrics,
)


@dataclass
class TaskSessionRecorder:
    task_id: str
    task_input: str
    decision_policy: str
    enable_audit: bool = False
    _step_records: list[dict] = field(default_factory=list)
    _prev_usage_snapshot: dict | None = None
    _usage_deltas: list[dict] = field(default_factory=list)
    _elapsed_ms_total: float = 0.0
    _selected_lineage_ids: list[str] = field(default_factory=list)
    _global_ordinal: int = 0

    def new_generation_recorder(self) -> GenerationRecorder:
        return GenerationRecorder()

    def new_vote_recorder(self) -> VoteRecorder:
        return VoteRecorder()

    @property
    def selected_lineage_ids(self) -> list[str]:
        return list(self._selected_lineage_ids)

    def record_decision(
        self,
        *,
        depth: int,
        parent_count: int,
        candidates: list[str],
        candidate_parent_lineage_ids: list[str],
        values: list[float],
        selected_indices: list[int],
        selected: list[str],
        is_stopping: bool,
        elapsed_ms: float,
        task: _SolverTask,
        x: str,
        gen_rec: GenerationRecorder | None = None,
        vote_rec: VoteRecorder | None = None,
    ) -> dict:
        self._elapsed_ms_total += elapsed_ms
        sel_set = frozenset(selected_indices)
        is_last_step = (depth + 1 >= task.steps)

        parent_lid_set: frozenset[str] = frozenset(
            lid for lid in candidate_parent_lineage_ids if lid)
        apply_expanded_disposition(self._step_records, parent_lid_set)

        events: list[CandidateEvent] = []
        new_selected_lineages: list[str] = []

        for ci, candidate in enumerate(candidates):
            parent_lid = (
                candidate_parent_lineage_ids[ci]
                if ci < len(candidate_parent_lineage_ids) else ""
            )
            lid = build_lineage_id(
                self.task_id, depth,
                parent_lineage_id=str(depth - 1) if depth > 0 else "",
                generation_ordinal=self._global_ordinal,
            )
            self._global_ordinal += 1

            disposition, is_terminal = detect_disposition(
                ci=ci, candidate=candidate,
                selected_indices=sel_set,
                is_last_step=is_last_step,
                task=task, x=x,
            )

            if ci in sel_set:
                new_selected_lineages.append(lid)

            evaluator_score = float(values[ci]) if ci < len(values) else None

            events.append(make_candidate_event(
                task_id=self.task_id, depth=depth,
                parent_lineage_id=parent_lid,
                generation_ordinal=ci,
                stage=("evaluator_scored"
                       if ci < len(values) else "proposed"),
                disposition=disposition,
                raw_content=candidate,
                canonical_text=candidate,
                evaluator_score=evaluator_score,
                evaluator_mode="vote",
                is_selected=ci in sel_set,
                is_terminal=is_terminal,
            ))

        self._selected_lineage_ids = new_selected_lineages

        audit_hash = compute_decision_audit(
            enable_audit=self.enable_audit,
            task_id=self.task_id,
            task_input=x,
            depth=depth,
            candidates=candidates,
            values=values,
            selected_indices=selected_indices,
            events=events,
            is_stopping=is_stopping,
            decision_policy=self.decision_policy,
        )

        delta: dict = {}
        try:
            from federated_mcts.models import get_model_usage_summary
            cur = get_model_usage_summary()
        except Exception:
            cur = {}
        if self._prev_usage_snapshot and cur:
            for m, stats in cur.items():
                p = self._prev_usage_snapshot.get(m, {})
                for k in ("completion_tokens", "prompt_tokens",
                          "total_tokens", "total_calls"):
                    cv = int(stats.get(k, 0))
                    pv = int(p.get(k, 0))
                    if cv != pv:
                        delta[f"{m}.{k}"] = cv - pv
        self._prev_usage_snapshot = cur
        self._usage_deltas.append(delta)

        gen_steps = gen_rec.steps if gen_rec is not None else []
        vote_steps = vote_rec.steps if vote_rec is not None else []

        step_record: dict = {
            "depth": depth,
            "parent_count": parent_count,
            "candidates": events,
            "values": list(values),
            "selected_indices": list(selected_indices),
            "selected": selected,
            "is_stopping": is_stopping,
            "elapsed_ms": elapsed_ms,
            "audit_hash": audit_hash,
            "usage_delta": delta,
            "gen_instrumentation": [dict(s) for s in gen_steps],
            "vote_instrumentation": [dict(s) for s in vote_steps],
        }
        self._step_records.append(step_record)
        return step_record

    def record_failed_step(
        self, *, depth: int, parent_count: int, gen_rec: GenerationRecorder,
        elapsed_ms: float, task: _SolverTask, x: str,
    ) -> dict:
        """Record a terminal step with zero valid candidates (dead-end expansion)."""
        self._elapsed_ms_total += elapsed_ms
        events = build_failed_step_events(gen_rec, self.task_id, depth)
        step_record: dict = {
            "depth": depth, "parent_count": parent_count,
            "candidates": events, "values": [], "selected_indices": [],
            "selected": [], "is_stopping": True, "elapsed_ms": elapsed_ms,
            "audit_hash": "", "usage_delta": {},
            "gen_instrumentation": [dict(s) for s in gen_rec.steps],
            "vote_instrumentation": []}
        self._step_records.append(step_record)
        return step_record

    def to_task_metrics(self) -> TaskCandidateMetrics:
        step_details = [
            make_step_metrics(
                depth=sr["depth"],
                parent_count=sr["parent_count"],
                candidates=sr["candidates"],
            ) for sr in self._step_records
        ]
        return make_task_metrics(
            task_id=self.task_id,
            task_input=self.task_input,
            step_details=step_details,
        )

    @property
    def total_elapsed_ms(self) -> float:
        return self._elapsed_ms_total

    @property
    def usage_deltas(self) -> list[dict]:
        return list(self._usage_deltas)
