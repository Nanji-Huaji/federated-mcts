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
    """Create FederatedSolver — when mock_client, bypasses real __init__.

    When ``--mock_client`` is set, the real ``FederatedSolver.__init__``
    is NEVER called.  Instead, the object is allocated via
    ``object.__new__`` and only the essential attributes (gpts, args,
    time_tracker) are set.  This guarantees no HTTP client, API key
    reading, model_config loading, or network I/O occurs.

    A test that patches ``FederatedSolver.__init__`` to unconditionally
    raise can prove mock mode never reaches the real constructor.

    When *solver_cls* is provided (e.g. a mock.patch target), it is
    used in non-mock mode so existing tests that replace
    ``FederatedSolver`` continue to work.
    """
    import json as _json

    if setup.mock_client:
        from federated_mcts.evaluation.game24_mock_client import build_mock_gpts
        from federated_mcts.federation.time_tracker import TimeTracker

        with open(setup.model_config_path) as f:
            model_config = _json.load(f)
        mock_gpts = build_mock_gpts(model_config)

        # Allocate solver without calling __init__ at all
        # Use FederatedSolver only for the mixin methods, not construction
        if solver_cls is not None:
            solver = object.__new__(solver_cls)
        else:
            from federated_mcts.federation import FederatedSolver as _FS
            solver = object.__new__(_FS)

        solver.args = args
        solver.gpts = mock_gpts
        solver.time_tracker = TimeTracker(list(mock_gpts.keys()))
        # latency_dict is a property backed by time_tracker — no need to set
        solver.dqn_session = None
        if hasattr(solver, "_strategy_cache"):
            solver._strategy_cache = {}
        return solver

    # Non-mock path: use real constructor
    if solver_cls is None:
        from federated_mcts.federation import FederatedSolver as _FS
        solver_cls = _FS
    return solver_cls(args)
