"""Runner orchestration helpers: preflight gates, dry-run, mock solver setup.

Extracted from ``game24_runner.py`` to keep each module ≤250 pure LOC.
"""

from __future__ import annotations

from federated_mcts.evaluation.game24_manifest import Game24Manifest
from federated_mcts.evaluation.game24_preflight import PreflightProjection
from federated_mcts.evaluation.game24_runner_setup import ExperimentSetup


def run_preflight_and_dry_run(
    setup: ExperimentSetup,
    manifest: Game24Manifest,
    projection: PreflightProjection | None,
    *,
    skip_full_val_gate: bool = False,
) -> bool:
    """Run preflight gate check and/or dry-run resolution.

    For dry runs: prints resolution and returns ``False`` (caller exits).
    For non-dry: gates full validation, returns ``True`` to proceed.

    When *skip_full_val_gate* is ``True`` (B0 measured evidence active),
    the static projection is still computed and printed but the full
    200-ID gate is deferred to the B0 measured-evidence gate.

    Returns ``True`` if the caller should continue with the real run,
    ``False`` if the caller should return (dry run).
    """
    import sys as _sys

    if projection is not None:
        print(
            f"Preflight projection: {projection.worst_case_raw_unit_estimate:.6f} "
            f"raw units (gate: {projection.gate_limit}, "
            f"passed: {projection.gate_passed})"
        )
        if not projection.gate_passed and len(setup.task_ids) > 20:
            if skip_full_val_gate:
                print(
                    "Static preflight projection exceeds gate — "
                    "deferring to B0 measured-evidence projection gate."
                )
            else:
                print(
                    "FATAL: Preflight projection exceeds gate. "
                    f"{projection.worst_case_raw_unit_estimate:.6f} > "
                    f"{projection.gate_limit}. "
                    "Full 200-ID validation blocked before client construction.",
                    file=_sys.stderr,
                )
                _sys.exit(1)

    if setup.dry_run:
        print("DRY RUN: flags/IDs/config/manifest/projection resolved.")
        print(f"  task IDs ({len(setup.task_ids)}): {setup.task_ids[:5]}...")
        dry_proj = projection.to_json() if projection else "no-projection"
        print(f"  projection: {dry_proj}")
        print("  NO output dirs/files, journals, ledgers, "
              "auth registries, solver, or API calls were created.")
        return False
    return True


def setup_mock_solver(args, setup: ExperimentSetup, solver_cls=None):
    """Build adapters after authorization, then use the normal injected solver.

    Offline and real runs exercise exactly the same constructor and search code.
    This boundary is the only runner code that resolves model credentials.
    """
    clients: dict
    if setup.mock_client:
        from federated_mcts.evaluation.game24_mock_client import build_mock_gpts
        clients = build_mock_gpts(setup.model_configs)
    else:
        from federated_mcts.models.clients import create_clients
        clients = create_clients(setup.model_configs, default_temperature=args.temperature)
    if solver_cls is None:
        from federated_mcts.federation import FederatedSolver
        solver_cls = FederatedSolver
    return solver_cls(args, clients=clients)
