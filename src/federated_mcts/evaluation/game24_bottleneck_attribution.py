"""Post-hoc oracle attribution over recorded candidate events.

Why this module exists
----------------------
The oracle audit *runs* during a Game24 experiment, but its annotations are
discarded and its per-candidate verdicts are never persisted: the
``oracle_is_reachable`` / ``oracle_exact_pruned`` fields declared on
``CandidateEvent`` are ``None`` on every candidate of every task, and selected
candidates keep ``stage='evaluator_scored'``.  Consequently the depth-level
attribution required by the Todo 8 acceptance criteria cannot be read from any
artifact.

This module recomputes it from data that *is* recorded.  Each candidate event
stores ``canonical_text`` (e.g. ``"1 + 1 = 2 (left: 1 2 8)"``), which contains
the multiset of numbers remaining after that move.  Reachability — whether that
remaining multiset can still be combined to 24 — is a pure deterministic
function of those numbers, so it can be recovered exactly, with no API calls and
no dependence on the discarded annotations.

Recomputation is a *read-only* analysis over an existing run.  It does not
modify the run, its artifacts, or any decision.

Attribution semantics
---------------------
For each task the *first loss step* is the shallowest step at which no selected
candidate was reachable.  That step is then classified:

``no_candidate_generated``
    Nothing was generated at all -- a failure before generation even applies.
``generation_recall``
    Candidates were generated, but **none** was reachable.  The correct move was
    never proposed, so no selection rule could have saved the task.
``selection_regret``
    A reachable candidate existed and was not selected.  The information was
    present in the candidate set and the selector discarded it.
``terminal_unreached``
    Every step selected at least one reachable candidate, yet no selected
    candidate ever reached the terminal state (a single remaining number equal
    to 24).  Every individual decision looked solvable; the chain still did not
    terminate on target, which is what running out of depth looks like from the
    selector's point of view.

The distinction is the point of the exercise: ``generation_recall`` calls for
better generation (beam, diversity), while ``selection_regret`` calls for a
better evaluator or ranking.  They are not interchangeable fixes.

Solved tasks are classified ``none`` and excluded from the loss tallies but
retained in the per-depth aggregate, so reachable-candidate density can be read
across all steps.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Iterable

from federated_mcts.evaluation.game24_decision_helpers import (
    _parse_canonical_numbers,
)
from federated_mcts.evaluation.game24_oracle_audit import (
    _can_reach_target_numbers,
)

# First-loss classification labels.
LOSS_NONE = "none"
LOSS_NO_CANDIDATE_GENERATED = "no_candidate_generated"
LOSS_GENERATION_RECALL = "generation_recall"
LOSS_SELECTION_REGRET = "selection_regret"
LOSS_TERMINAL_UNREACHED = "terminal_unreached"

FIRST_LOSS_LABELS = (
    LOSS_NONE,
    LOSS_NO_CANDIDATE_GENERATED,
    LOSS_GENERATION_RECALL,
    LOSS_SELECTION_REGRET,
    LOSS_TERMINAL_UNREACHED,
)

# Loss classes reported as headline attribution, in descending severity.
REPORTED_LOSS_LABELS = (
    LOSS_GENERATION_RECALL,
    LOSS_SELECTION_REGRET,
    LOSS_TERMINAL_UNREACHED,
    LOSS_NO_CANDIDATE_GENERATED,
)

_TARGET = Fraction(24)


@dataclass(frozen=True, slots=True)
class DepthAttribution:
    """Reachability aggregate for one depth, summed over tasks."""

    depth: int
    proposed_count: int
    reachable_count: int
    unreachable_count: int
    selected_count: int
    selected_reachable_count: int
    selected_unreachable_count: int
    unreachable_with_reachable_alternative_count: int
    tasks_with_reachable_candidate: int

    @property
    def reachable_fraction(self) -> float:
        if self.proposed_count == 0:
            return 0.0
        return self.reachable_count / self.proposed_count

    @property
    def selected_precision(self) -> float:
        if self.selected_count == 0:
            return 1.0
        return self.selected_reachable_count / self.selected_count


@dataclass(frozen=True, slots=True)
class TaskAttribution:
    """Attribution for a single task."""

    task_id: str
    task_input: str
    solved: bool
    total_steps: int
    first_loss_depth: int | None
    first_loss_label: str
    reachable_candidates_total: int
    selected_reachable_total: int
    selected_total: int
    depth_metrics: tuple[DepthAttribution, ...]


@dataclass(frozen=True, slots=True)
class AttributionReport:
    """Aggregate attribution across a run."""

    task_count: int
    solved_count: int
    unsolved_count: int
    first_loss_counts: dict[str, int]
    depth_metrics: tuple[DepthAttribution, ...]
    tasks: tuple[TaskAttribution, ...]

    @property
    def unsolved_first_loss_counts(self) -> dict[str, int]:
        """First-loss counts restricted to the loss labels (drops ``none``)."""
        return {
            k: v for k, v in self.first_loss_counts.items() if k != LOSS_NONE
        }

    def first_loss_percentages(self) -> dict[str, float]:
        """Percentage of *unsolved* tasks attributable to each loss class."""
        denom = self.unsolved_count
        if denom == 0:
            return {k: 0.0 for k in FIRST_LOSS_LABELS if k != LOSS_NONE}
        return {
            k: 100.0 * v / denom
            for k, v in self.first_loss_counts.items()
            if k != LOSS_NONE
        }


def _is_selected(event: dict[str, Any]) -> bool:
    """A candidate counts as selected via the boolean flag or a selected stage.

    Both routes are honoured because the recorder sets ``is_selected`` while
    never writing a selected ``stage`` value (the ``total_selected == 0``
    defect), and an auditor should not silently inherit that bug.
    """
    if bool(event.get("is_selected")):
        return True
    stage = event.get("stage")
    return stage in ("selected", "expanded", "terminal_solved")


def candidate_reachability(canonical_text: Any) -> bool | None:
    """Return whether a candidate's remaining numbers can still reach 24.

    ``None`` means the candidate carried no parsable move text, so reachability
    is undefined rather than false -- the caller decides how to count it.
    """
    if not isinstance(canonical_text, str) or not canonical_text.strip():
        return None
    try:
        numbers: tuple[Fraction, ...] = _parse_canonical_numbers(canonical_text)
    except Exception:
        return None
    if not numbers:
        return None
    try:
        return bool(_can_reach_target_numbers(numbers))
    except Exception:
        return None


def _depth_attribution(
    depth: int, events: list[dict[str, Any]]
) -> DepthAttribution:
    reachable = 0
    unreachable = 0
    selected = 0
    selected_reachable = 0
    selected_unreachable = 0
    unreachable_with_alt = 0
    any_reachable = False

    for ev in events:
        ok = candidate_reachability(ev.get("canonical_text"))
        is_sel = _is_selected(ev)
        if ok is True:
            reachable += 1
            any_reachable = True
        elif ok is False:
            unreachable += 1
        if is_sel:
            selected += 1
            if ok is True:
                selected_reachable += 1
            elif ok is False:
                selected_unreachable += 1

    if any_reachable:
        for ev in events:
            if _is_selected(ev) and candidate_reachability(
                ev.get("canonical_text")
            ) is False:
                unreachable_with_alt += 1

    return DepthAttribution(
        depth=depth,
        proposed_count=len(events),
        reachable_count=reachable,
        unreachable_count=unreachable,
        selected_count=selected,
        selected_reachable_count=selected_reachable,
        selected_unreachable_count=selected_unreachable,
        unreachable_with_reachable_alternative_count=unreachable_with_alt,
        tasks_with_reachable_candidate=1 if any_reachable else 0,
    )


def reaches_terminal(canonical_text: Any) -> bool:
    """Return whether a candidate's own chain ends at the target value.

    The terminal state is a single remaining number equal to 24.  This is
    distinct from *reachability*: a candidate can be reachable (its remaining
    numbers could still be combined to 24) without itself having reached 24.
    """
    if not isinstance(canonical_text, str) or not canonical_text.strip():
        return False
    try:
        numbers = _parse_canonical_numbers(canonical_text)
    except Exception:
        return False
    return len(numbers) == 1 and numbers[0] == _TARGET


def attribute_task(task_metrics: dict[str, Any]) -> TaskAttribution:
    """Compute attribution for one task's ``task_metrics`` mapping."""
    step_details: Iterable[dict[str, Any]] = task_metrics.get("step_details", []) or []
    solved = bool(task_metrics.get("solved"))

    depth_metrics: list[DepthAttribution] = []
    reachable_total = 0
    selected_reachable_total = 0
    selected_total = 0
    first_loss_depth: int | None = None
    first_loss_label = LOSS_NONE
    selected_ever_reached_terminal = False
    any_loss_label: str | None = None

    for step in step_details:
        depth = int(step.get("depth", len(depth_metrics)))
        events = list(step.get("candidates", []) or [])
        dm = _depth_attribution(depth, events)
        depth_metrics.append(dm)
        reachable_total += dm.reachable_count
        selected_reachable_total += dm.selected_reachable_count
        selected_total += dm.selected_count

        for ev in events:
            if _is_selected(ev) and reaches_terminal(ev.get("canonical_text")):
                selected_ever_reached_terminal = True
                break

        if first_loss_depth is None and dm.selected_reachable_count == 0:
            first_loss_depth = depth
            if dm.proposed_count == 0:
                first_loss_label = LOSS_NO_CANDIDATE_GENERATED
            elif dm.reachable_count == 0:
                first_loss_label = LOSS_GENERATION_RECALL
            else:
                first_loss_label = LOSS_SELECTION_REGRET
            any_loss_label = first_loss_label

    if solved or selected_ever_reached_terminal:
        # A task that reached the target has no loss, regardless of whether
        # intermediate steps happened to select an unreachable candidate.
        first_loss_depth = None
        first_loss_label = LOSS_NONE
    elif any_loss_label is None:
        # No step lacked a reachable selection, yet the target was never
        # reached: the chain stayed individually solvable and still did not
        # terminate on target.  Reported as its own class rather than folded
        # into a per-step label that does not describe it.
        first_loss_label = LOSS_TERMINAL_UNREACHED

    return TaskAttribution(
        task_id=str(task_metrics.get("task_id", "")),
        task_input=str(task_metrics.get("task_input", "")),
        solved=solved,
        total_steps=len(depth_metrics),
        first_loss_depth=first_loss_depth,
        first_loss_label=first_loss_label,
        reachable_candidates_total=reachable_total,
        selected_reachable_total=selected_reachable_total,
        selected_total=selected_total,
        depth_metrics=tuple(depth_metrics),
    )


def _merge_depth_metrics(
    all_metrics: list[DepthAttribution],
) -> tuple[DepthAttribution, ...]:
    by_depth: dict[int, dict[str, int]] = {}
    for dm in all_metrics:
        acc = by_depth.setdefault(
            dm.depth,
            {
                "proposed_count": 0,
                "reachable_count": 0,
                "unreachable_count": 0,
                "selected_count": 0,
                "selected_reachable_count": 0,
                "selected_unreachable_count": 0,
                "unreachable_with_reachable_alternative_count": 0,
                "tasks_with_reachable_candidate": 0,
            },
        )
        for field in acc:
            acc[field] += getattr(dm, field)
    return tuple(
        DepthAttribution(depth=d, **acc) for d, acc in sorted(by_depth.items())
    )


def attribute_run(task_metrics_list: Iterable[dict[str, Any]]) -> AttributionReport:
    """Compute attribution across a run's per-task ``task_metrics`` mappings."""
    tasks: list[TaskAttribution] = []
    for tm in task_metrics_list:
        tasks.append(attribute_task(tm))

    counts = {label: 0 for label in FIRST_LOSS_LABELS}
    for t in tasks:
        counts[t.first_loss_label] = counts.get(t.first_loss_label, 0) + 1

    solved = sum(1 for t in tasks if t.solved)
    merged = _merge_depth_metrics(
        [dm for t in tasks for dm in t.depth_metrics]
    )

    return AttributionReport(
        task_count=len(tasks),
        solved_count=solved,
        unsolved_count=len(tasks) - solved,
        first_loss_counts=counts,
        depth_metrics=merged,
        tasks=tuple(tasks),
    )


def load_task_metrics_from_journal(journal_path: str) -> list[dict[str, Any]]:
    """Read every task's ``task_metrics_json`` from a journal file.

    Skips the ``_meta`` entry.  Raises ``ValueError`` if a task entry lacks
    metrics, rather than silently shrinking the denominator.
    """
    with open(journal_path) as f:
        journal = json.load(f)
    out: list[dict[str, Any]] = []
    for key, entry in journal.items():
        if key == "_meta":
            continue
        raw = entry.get("task_metrics_json")
        if not raw:
            raise ValueError(f"journal entry {key!r} has no task_metrics_json")
        out.append(json.loads(raw))
    return out


def render_markdown(report: AttributionReport) -> str:
    """Render a human-readable attribution summary."""
    lines: list[str] = []
    lines.append("# Bottleneck Attribution (post-hoc oracle recomputation)")
    lines.append("")
    lines.append(
        f"Tasks: {report.task_count} | solved: {report.solved_count} | "
        f"unsolved: {report.unsolved_count}"
    )
    lines.append("")
    lines.append("## First-loss classification (unsolved tasks)")
    lines.append("")
    pct = report.first_loss_percentages()
    lines.append("| class | tasks | % of unsolved |")
    lines.append("|---|---:|---:|")
    for label in REPORTED_LOSS_LABELS:
        lines.append(
            f"| `{label}` | {report.first_loss_counts.get(label, 0)} | "
            f"{pct.get(label, 0.0):.1f}% |"
        )
    classified = sum(
        report.first_loss_counts.get(label, 0) for label in REPORTED_LOSS_LABELS
    )
    lines.append(
        f"| **classified** | **{classified}** | "
        f"**{100.0 * classified / report.unsolved_count if report.unsolved_count else 0.0:.1f}%** |"
    )
    lines.append("")
    lines.append("## By depth")
    lines.append("")
    lines.append(
        "| depth | proposed | reachable | reachable% | selected | "
        "sel reachable | sel precision | unreachable-with-alt |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for dm in report.depth_metrics:
        lines.append(
            f"| {dm.depth} | {dm.proposed_count} | {dm.reachable_count} | "
            f"{dm.reachable_fraction * 100:.1f}% | {dm.selected_count} | "
            f"{dm.selected_reachable_count} | "
            f"{dm.selected_precision * 100:.1f}% | "
            f"{dm.unreachable_with_reachable_alternative_count} |"
        )
    return "\n".join(lines) + "\n"
