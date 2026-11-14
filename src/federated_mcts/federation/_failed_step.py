"""Dead-end expansion telemetry: record a failed-generation step and signal the
caller to break out of the search loop.

This helper exists solely to keep ``standard_solvers.py`` under the 250 pure-LOC
plan cap.  It encapsulates the exact ``new_ys`` empty-check + ``record_failed_step``
call that was previously inlined at the pruning site.
"""

from __future__ import annotations

import time
from typing import Protocol


class _TaskRecorder(Protocol):
    """Structural interface for task-level step recorders.

    Any recorder passed to ``record_failed_step_if_empty`` must satisfy this
    protocol — no concrete import needed, keeping the module self-contained.
    """

    def record_failed_step(
        self,
        *,
        depth: int,
        parent_count: int,
        gen_rec: object,
        elapsed_ms: float,
        task: object,
        x: str,
    ) -> object: ...


def record_failed_step_if_empty(
    new_ys: list[str],
    task_recorder: _TaskRecorder | None,
    step: int,
    ys: list[str],
    gen_rec: object | None,
    t0: float,
    task: object,
    x: str,
) -> bool:
    """If *new_ys* is empty, record a dead-end step and return ``True``.

    The caller should ``break`` when this returns ``True``.

    *task_recorder* may be ``None`` (no-op / no-recorder path); when present it
    must expose ``.record_failed_step(depth=..., parent_count=..., gen_rec=...,
    elapsed_ms=..., task=..., x=...)``.
    """
    if not new_ys:
        if task_recorder is not None:
            task_recorder.record_failed_step(
                depth=step,
                parent_count=len(ys),
                gen_rec=gen_rec,
                elapsed_ms=(time.time() - t0) * 1000,
                task=task,
                x=x,
            )
        return True
    return False
