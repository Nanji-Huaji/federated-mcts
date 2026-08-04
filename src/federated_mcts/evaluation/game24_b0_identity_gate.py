"""B0 target-identity gate: fail-closed decision-field comparison.

Compares the target validation manifest against the validated B0
source decision identity.  All decision-affecting fields must match
except task_ids, ordered_task_ids_sha256, task_split_name, and
split_file_sha256 (differences from 20-pilot → 200-val are expected).

Extracted from ``game24_b0_gate.py`` to keep each module ≤250 pure LOC.
"""

from __future__ import annotations

import sys

from federated_mcts.evaluation.game24_b0_evidence import B0EvidenceRecord
from federated_mcts.evaluation.game24_manifest import Game24Manifest


# Fields that legitimately differ between 20-pilot and 200-val runs
_ALLOWED_DIFF_FIELDS: frozenset[str] = frozenset({
    "task_ids",
    "ordered_task_ids_sha256",
    "task_split_name",
    "split_file_sha256",
})


def enforce_target_identity(
    target: Game24Manifest,
    evidence: B0EvidenceRecord,
) -> None:
    """Fail-closed: compare target manifest decision fields against B0 source.

    All decision-affecting fields must match the validated B0 source
    identity.  The fields in ``_ALLOWED_DIFF_FIELDS`` are exempt because
    they legitimately change from 20-pilot to 200-val.

    The target task must be 'game24' with exactly 200 val IDs.

    Raises ``SystemExit(1)`` on any mismatch — zero side effects.
    """
    src = evidence.source_decision
    mismatches: list[str] = []

    # ── Task requirement ──────────────────────────────────────────
    if target.task != "game24":
        mismatches.append(
            f"task: expected 'game24', got {target.task!r}"
        )
    if len(target.task_ids) != 200:
        mismatches.append(
            f"task_ids count: expected 200, got {len(target.task_ids)}"
        )

    # ── Decision policy ───────────────────────────────────────────
    if target.decision_policy != src.decision_policy:
        mismatches.append(
            f"decision_policy: expected {src.decision_policy!r}, "
            f"got {target.decision_policy!r}"
        )

    # ── Reproducibility ───────────────────────────────────────────
    if target.seed != src.seed:
        mismatches.append(
            f"seed: expected {src.seed}, got {target.seed}"
        )
    if target.temperature != src.temperature:
        mismatches.append(
            f"temperature: expected {src.temperature}, "
            f"got {target.temperature}"
        )

    # ── Solver configuration ──────────────────────────────────────
    for field, src_val, tgt_val in [
        ("method_generate", src.method_generate, target.method_generate),
        ("method_evaluate", src.method_evaluate, target.method_evaluate),
        ("method_select", src.method_select, target.method_select),
        ("n_generate_sample", src.n_generate_sample, target.n_generate_sample),
        ("n_evaluate_sample", src.n_evaluate_sample, target.n_evaluate_sample),
        ("n_select_sample", src.n_select_sample, target.n_select_sample),
    ]:
        if tgt_val != src_val:
            mismatches.append(
                f"{field}: expected {src_val!r}, got {tgt_val!r}"
            )

    # ── game24_exact_prune ────────────────────────────────────────
    if target.game24_exact_prune != src.game24_exact_prune:
        mismatches.append(
            f"game24_exact_prune: expected {src.game24_exact_prune!r}, "
            f"got {target.game24_exact_prune!r}"
        )

    # ── Model identity ────────────────────────────────────────────
    if target.model_config_sha256 != src.model_config_sha256:
        mismatches.append(
            f"model_config_sha256: expected {src.model_config_sha256}, "
            f"got {target.model_config_sha256}"
        )
    if sorted(target.model_names) != sorted(src.model_names):
        mismatches.append(
            f"model_names: expected {src.model_names}, "
            f"got {target.model_names}"
        )

    # ── Solver extras ─────────────────────────────────────────────
    if target.assign_strategy != src.assign_strategy:
        mismatches.append(
            f"assign_strategy: expected {src.assign_strategy!r}, "
            f"got {target.assign_strategy!r}"
        )
    if target.runner_path != src.runner_path:
        mismatches.append(
            f"runner_path: expected {src.runner_path!r}, "
            f"got {target.runner_path!r}"
        )

    # ── Budget caps ───────────────────────────────────────────────
    if target.max_call_limit != src.max_call_limit:
        mismatches.append(
            f"max_call_limit: expected {src.max_call_limit!r}, "
            f"got {target.max_call_limit!r}"
        )
    if target.max_token_limit != src.max_token_limit:
        mismatches.append(
            f"max_token_limit: expected {src.max_token_limit!r}, "
            f"got {target.max_token_limit!r}"
        )
    if target.max_cost_usd != src.max_cost_usd:
        mismatches.append(
            f"max_cost_usd: expected {src.max_cost_usd!r}, "
            f"got {target.max_cost_usd!r}"
        )
    if target.max_raw_unit_limit != src.max_raw_unit_limit:
        mismatches.append(
            f"max_raw_unit_limit: expected {src.max_raw_unit_limit}, "
            f"got {target.max_raw_unit_limit}"
        )

    if mismatches:
        print(
            "FATAL: Target-identity gate FAILED — "
            "target decision fields do not match B0 source evidence.",
            file=sys.stderr,
        )
        for m in mismatches:
            print(f"  {m}", file=sys.stderr)
        print(
            "\nAllowed differences (20-pilot → 200-val): "
            f"{', '.join(sorted(_ALLOWED_DIFF_FIELDS))}.\n"
            "All other decision fields MUST match the validated B0 source.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        "Target-identity gate PASSED — decision fields match B0 source evidence. "
        f"(model_config_sha256={src.model_config_sha256[:12]}..., "
        f"model_names={src.model_names}, "
        f"decision_fp={evidence.decision_policy_fingerprint[:12]}...)"
    )
