"""B0 measured-evidence gate enforcement for the Game24 runner.

Integrates with the runner orchestrator to validate B0 pilot evidence
and gate the full 200-ID validation run BEFORE output directory creation,
budget ledger setup, auth registry consumption, solver construction, or
any model/client I/O.

Provides:
- CLI argument registration (``add_b0_cli_args``)
- Gate enforcement (``enforce_b0_gate``)
- Dry-run rendering (``render_b0_dry_run``)

All three return without side effects when evidence is valid and
projection passes; they raise ``SystemExit(1)`` or return ``False``
when the gate fails or no B0 evidence was requested.
"""

from __future__ import annotations

import sys
from decimal import Decimal

from federated_mcts.evaluation.game24_b0_evidence import (
    B0EvidenceRecord,
    B0ProjectionRecord,
    FULL_VALIDATION_GATE,
    FULL_TASK_COUNT,
    SCALE_FACTOR,
    compute_b0_projection,
)
from federated_mcts.evaluation.game24_b0_validate import validate_b0_evidence
from federated_mcts.evaluation.game24_manifest import Game24Manifest
from federated_mcts.evaluation.game24_b0_identity_gate import enforce_target_identity


def add_b0_cli_args(parser) -> None:
    """Register B0 measured-evidence CLI flags on an argparse parser.

    These are only active for 200-ID validation runs (``--task_split val``
    with full validation) and are enforced before any side effects.
    """
    parser.add_argument(
        "--b0_manifest", type=str, default=None,
        help="Path to B0 pilot manifest (_manifest.json) for measured projection",
    )
    parser.add_argument(
        "--b0_performance", type=str, default=None,
        help="Path to B0 pilot performance (_performance.json) for measured projection",
    )
    parser.add_argument(
        "--b0_journal", type=str, default=None,
        help="Path to B0 pilot journal (_journal.json) for measured projection",
    )
    parser.add_argument(
        "--b0_budget", type=str, default=None,
        help="Path to B0 pilot budget ledger (_budget.jsonl) for measured projection",
    )
    parser.add_argument(
        "--b0_dry_run", action="store_true", default=False,
        help="Validate B0 evidence and render projection without side effects",
    )


def _has_b0_evidence(args) -> bool:
    """Return True if any B0 evidence flag was provided as a string path."""
    for flag in ("b0_manifest", "b0_performance", "b0_journal", "b0_budget"):
        val = getattr(args, flag, None)
        if isinstance(val, str) and val:
            return True
    return False


def _all_b0_paths_provided(args) -> bool:
    """Return True only if ALL four B0 evidence paths are provided as strings."""
    for flag in ("b0_manifest", "b0_performance", "b0_journal", "b0_budget"):
        val = getattr(args, flag, None)
        if not isinstance(val, str) or not val:
            return False
    return True


def _is_full_validation(args) -> bool:
    """Return True if this is a 200-ID validation (not pilot, not test)."""
    task_split = getattr(args, "task_split", None)
    pilot_ids = getattr(args, "pilot_ids", None)
    # Full validation: task_split="val" AND no pilot_ids override
    return task_split == "val" and not pilot_ids


def enforce_b0_gate(args, *, manifest: Game24Manifest | None = None) -> bool:
    """Validate B0 evidence and gate full 200-ID validation.

    Called BEFORE output directory creation, budget ledger setup,
    auth registry, solver, or client construction.

    When `manifest` is provided and B0 evidence is active, the gate
    also performs a fail-closed target-identity comparison:
    all decision-affecting fields (model, seed, temperature, methods,
    n settings, exact_prune, assign_strategy, budget caps, runner_path)
    MUST match the validated B0 source manifest — only task_ids,
    ordered_task_ids_sha256, task_split_name, and split_file_sha256
    are allowed to differ (20 pilot → 200 val).

    Returns:
        ``True`` if the caller should proceed (gate passed, no B0 flags).
        ``False`` if dry-run was requested (caller must return).
        Raises ``SystemExit(1)`` on any gate failure.

    Does NOT modify any files, construct solvers, or make API calls.
    """
    # If no B0 evidence flags, skip gate (proceed normally)
    if not _has_b0_evidence(args):
        return True

    b0_dry = bool(getattr(args, "b0_dry_run", False))

    # Require full 200-ID validation context
    if not _is_full_validation(args):
        print(
            "FATAL: --b0_* evidence flags require --task_split val "
            "(200-ID validation) without --pilot_ids.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Require all four evidence paths
    if not _all_b0_paths_provided(args):
        missing = [
            flag for flag in
            ("b0_manifest", "b0_performance", "b0_journal", "b0_budget")
            if not getattr(args, flag, None)
        ]
        print(
            f"FATAL: All four B0 evidence paths are required. "
            f"Missing: {missing}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Validate evidence (fail-closed)
    try:
        evidence = validate_b0_evidence(
            manifest_path=args.b0_manifest,
            performance_path=args.b0_performance,
            journal_path=args.b0_journal,
            budget_path=args.b0_budget,
        )
    except ValueError as e:
        print(f"FATAL: B0 evidence validation failed: {e}", file=sys.stderr)
        sys.exit(1)

    # ── Target-identity gate (fail-closed) ──────────────────────────
    if manifest is not None:
        enforce_target_identity(manifest, evidence)

    # Compute projection
    try:
        projection, startup_proof, retry_proof = compute_b0_projection(evidence)
    except ValueError as e:
        print(f"FATAL: B0 projection computation failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Render detailed projection
    _render_projection(evidence, projection, startup_proof, retry_proof)

    # Gate check
    if not projection.gate_passed:
        print(
            f"\nFATAL: B0 measured projection {projection.projected_raw_units} "
            f"exceeds gate {FULL_VALIDATION_GATE}. "
            f"Full 200-ID validation blocked before client construction.",
            file=sys.stderr,
        )
        sys.exit(1)

    if b0_dry:
        print("\nB0 DRY RUN COMPLETE. No output directories, files, journals, "
              "ledgers, auth registries, solvers, or API calls were created.")
        return False

    print("\nB0 measured projection gate PASSED. Proceeding with full validation.")
    return True


def _render_projection(
    evidence: B0EvidenceRecord,
    projection: B0ProjectionRecord,
    startup_proof: str,
    retry_proof: str,
) -> None:
    """Print a detailed projection summary (zero side effects)."""
    print("=" * 72)
    print("B0 MEASURED PROJECTION — DETAILED RENDERING")
    print("=" * 72)
    print(f"  Schema version:       {projection.schema_version}")
    print(f"  Pilot tasks:          {evidence.pilot_id_count} (completed: {evidence.tasks_completed})")
    print(f"  Target tasks:         {FULL_TASK_COUNT}")
    print(f"  Scale factor:         {SCALE_FACTOR}")
    print()
    print("  --- Evidence summary ---")
    print(f"  Decision policy:      {evidence.decision_policy}")
    print(f"  Oracle audit:         {evidence.oracle_audit}")
    print(f"  Seed:                 {evidence.seed}")
    print(f"  Generate/Evaluate/Select: {evidence.method_generate}/{evidence.method_evaluate}/{evidence.method_select}")
    print(f"  n_generate/evaluate/select: {evidence.n_generate_sample}/{evidence.n_evaluate_sample}/{evidence.n_select_sample}")
    print(f"  Calls:                {evidence.calls}")
    print(f"  Prompt tokens:        {evidence.prompt_tokens}")
    print(f"  Completion tokens:    {evidence.completion_tokens}")
    print(f"  Total tokens:         {evidence.total_tokens}")
    print()
    print("  --- Formula terms ---")
    print(f"  Pilot usage:          {projection.pilot_usage}")
    print(f"  × scale factor:       {projection.scale_factor}")
    print(f"  = Scaled usage:       {projection.scaled_usage}")
    print(f"  + Startup overhead:   {projection.startup_overhead}")
    print(f"    Proof: {startup_proof}")
    print(f"  + Retry allowance:    {projection.retry_allowance}")
    print(f"    Proof: {retry_proof}")
    print(f"  = Projected total:    {projection.projected_raw_units}")
    print()
    print(f"  Gate limit:           {projection.gate_limit}")
    print(f"  Gate passed:          {projection.gate_passed}")
    print()
    print("  --- Source artifact hashes ---")
    print(f"  Manifest SHA-256:     {evidence.manifest_sha256}")
    print(f"  Performance SHA-256:  {evidence.performance_sha256}")
    print(f"  Journal SHA-256:      {evidence.journal_sha256}")
    print(f"  Budget SHA-256:       {evidence.budget_sha256}")
    print(f"  Decision fingerprint: {evidence.decision_policy_fingerprint}")
    print(f"  Run fingerprint:      {evidence.run_fingerprint}")
    print("=" * 72)
