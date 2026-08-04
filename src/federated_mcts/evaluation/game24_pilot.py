"""Validation pilot fixture with exactly 20 deterministic task IDs.

The pilot IDs are the first 20 ordered validation IDs from the canonical
``data/splits/game24.json`` split file (sha256:
7eaf44120e5317e1b78d9468fec0547572f3aeb84c63a17382d5ecd2a93b42c0).

These must never intersect the frozen test split.  The module provides the
canonical list plus membership and intersection checks used by the
CLI/safety layer.
"""

from __future__ import annotations

# ── Canonical validation pilot (first 20 ordered validation IDs) ──
VALIDATION_PILOT_IDS: list[int] = [
    3, 9, 16, 19, 24, 25, 28, 36, 40, 42,
    45, 71, 77, 81, 102, 104, 105, 116, 117, 119,
]
"""Ordered validation pilot tasks — the exact 20 IDs from the approved plan."""

_VALIDATION_PILOT_SET: frozenset[int] = frozenset(VALIDATION_PILOT_IDS)


def is_pilot_id(task_id: int) -> bool:
    """Return True if *task_id* is in the validation pilot set."""
    return task_id in _VALIDATION_PILOT_SET


def pilot_ids_intersect_test(test_ids: frozenset[int]) -> bool:
    """Return True if any validation pilot ID intersects the frozen test set."""
    return not _VALIDATION_PILOT_SET.isdisjoint(test_ids)


def validate_pilot_membership(task_ids: list[int]) -> tuple[bool, str]:
    """Check that every ID in *task_ids* is a validation pilot ID.

    Returns ``(True, "")`` on success or ``(False, reason)`` listing
    non-pilot IDs.
    """
    non_pilot = [t for t in task_ids if t not in _VALIDATION_PILOT_SET]
    if non_pilot:
        return False, (
            f"IDs not in validation pilot set: {non_pilot}. "
            f"Pilot IDs are: {VALIDATION_PILOT_IDS}"
        )
    return True, ""


def pilot_ids() -> list[int]:
    """Return a copy of the canonical validation pilot ID list."""
    return list(VALIDATION_PILOT_IDS)
