"""Task resolution safety gates for the Game24 experiment runner.

Enforces BEFORE manifest, client, and authorization construction:

1. Explicit ``--pilot_ids`` cannot combine with ``--task_split`` or
   ``--task_start_index/--task_end_index`` range overrides.
2. Pilot IDs must belong to the validation split and never intersect the
   frozen test split.
3. ``--task_split=test`` MUST be accompanied by ``--selected_test_auth``.
4. ``--selected_test_auth`` is ONLY valid for the full test split (200 IDs).
5. ``--game24_decision_policy`` must be valid.

All checks return ``(True, "")`` on success or ``(False, reason)`` on
failure.  The caller is responsible for printing the reason and exiting
before any side effects occur.
"""

from __future__ import annotations

from federated_mcts.evaluation.game24_pilot import (
    VALIDATION_PILOT_IDS,
    pilot_ids_intersect_test,
)


def validate_task_resolution(
    *,
    pilot_ids: list[int] | None,
    task_split: str | None,
    task_start_index: int,
    task_end_index: int,
    selected_test_auth: str | None,
    decision_policy: str,
) -> tuple[bool, str]:
    """Validate task ID resolution against safety rules.

    Args:
        pilot_ids: Explicit ``--pilot_ids`` list or ``None``.
        task_split: ``--task_split`` value (``"train"``, ``"val"``, ``"test"``).
        task_start_index: ``--task_start_index`` value.
        task_end_index: ``--task_end_index`` value.
        selected_test_auth: ``--selected_test_auth`` path or ``None``.
        decision_policy: ``--game24_decision_policy`` value.

    Returns:
        ``(True, "")`` if safe, ``(False, reason)`` otherwise.
    """
    # ── Pilot exclusivity ──────────────────────────────────────────
    if pilot_ids is not None:
        if task_split is not None:
            return False, (
                "--pilot_ids cannot be combined with --task_split. "
                "Pilot IDs must run independently."
            )
        if task_start_index != 900 or task_end_index != 1000:
            return False, (
                "--pilot_ids cannot be combined with --task_start_index "
                "or --task_end_index.  Pilot IDs are explicit."
            )
        # Confirm pilot IDs are exactly the canonical set
        if sorted(pilot_ids) != sorted(VALIDATION_PILOT_IDS):
            return False, (
                f"--pilot_ids must match the canonical validation pilot set: "
                f"{VALIDATION_PILOT_IDS}"
            )

    # ── Decision policy validity ────────────────────────────────────
    valid_policies = {"local_validity", "heuristic_rules", "oracle_exact_pruned"}
    if decision_policy not in valid_policies:
        return False, (
            f"Unknown decision_policy {decision_policy!r}. "
            f"Must be one of: {sorted(valid_policies)}"
        )

    # ── Test split auth enforcement ─────────────────────────────────
    if task_split == "test":
        if selected_test_auth is None:
            return False, (
                "--task_split=test requires --selected_test_auth "
                "(path to auth registry JSONL)."
            )
    if selected_test_auth is not None:
        if task_split != "test":
            return False, (
                "--selected_test_auth may only be used with "
                "--task_split=test (full frozen test split)."
            )

    return True, ""


def validate_pilot_split_identity(
    split_file_path: str,
    pilot_ids: list[int],
    task: str = "game24",
) -> tuple[bool, str]:
    """Verify pilot IDs belong to the validation split and never intersect test.

    Reads ``data/splits/{task}.json``, checks that every *pilot_ids*
    element is in the ``"val"`` list and none are in the ``"test"`` list.

    Returns ``(True, "")`` on success or ``(False, reason)``.
    """
    import json

    try:
        with open(split_file_path) as f:
            splits = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        return False, f"Cannot read split file {split_file_path!r}: {exc}"

    val_set: frozenset[int] = frozenset(splits.get("val", []))
    test_set: frozenset[int] = frozenset(splits.get("test", []))

    not_in_val = [t for t in pilot_ids if t not in val_set]
    if not_in_val:
        return False, (
            f"Pilot IDs not in validation split: {not_in_val}"
        )

    if pilot_ids_intersect_test(test_set):
        in_test = [t for t in pilot_ids if t in test_set]
        return False, (
            f"Pilot IDs intersect frozen test split: {in_test}"
        )

    return True, ""
