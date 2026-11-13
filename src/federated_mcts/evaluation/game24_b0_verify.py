"""B0 evidence sub-verifiers: journal completeness and budget hash-chain.

Extracted from ``game24_b0_validate.py`` to keep each module ≤250 pure LOC.
Verification is genuinely read-only: it never constructs a writable
``BudgetLedger`` (which would append an ``init`` entry to an empty or corrupt
file), so a failed check leaves the evidence bytes untouched.
All checks raise ``ValueError`` on failure.
"""

from __future__ import annotations

import os
from decimal import ROUND_HALF_UP, Decimal

from federated_mcts.evaluation.game24_b0_evidence import (
    CANONICAL_PILOT_IDS,
    RAW_UNIT_PRECISION,
)


def verify_journal(journal_path: str, run_fingerprint: str) -> set[str]:
    """Validate B0 journal: readable, non-empty, metadata match, 20 completed IDs.

    Returns the set of completed task ID strings."""
    from federated_mcts.evaluation.game24_persistence import (
        AtomicTaskJournal,
        _CorruptJournalError,
    )

    try:
        journal = AtomicTaskJournal(journal_path)
        records = journal.read_all()
    except _CorruptJournalError as e:
        raise ValueError(f"B0 journal corrupt: {e}")

    if not records:
        raise ValueError("B0 journal is empty")

    meta = records.get("_meta")
    if not isinstance(meta, dict):
        raise ValueError("B0 journal _meta missing or corrupt")
    stored_fp = str(meta.get("run_fingerprint", ""))
    if stored_fp != run_fingerprint:
        raise ValueError(
            f"B0 journal run_fingerprint {stored_fp[:20]}... "
            f"!= manifest {run_fingerprint[:20]}..."
        )

    completed = journal.completed_ids()
    expected = {str(tid) for tid in CANONICAL_PILOT_IDS}
    if completed != expected:
        missing = expected - completed
        extra = completed - expected
        parts: list[str] = []
        if missing:
            parts.append(f"missing: {sorted(missing, key=int)}")
        if extra:
            parts.append(f"extra: {sorted(extra, key=int)}")
        raise ValueError(f"B0 journal completed IDs mismatch: {'; '.join(parts)}")
    return completed


def verify_budget_ledger(budget_path: str) -> Decimal:
    """Validate B0 budget hash chain and spent reconciliation.

    The spent total is read from the hash-chained ledger and checked for
    finiteness and non-negativity.  It is deliberately NOT compared against a
    pinned constant: spend is a sampling outcome, so pinning it rejected every
    legitimate re-run, and the budget ceiling is enforced separately at runtime
    by ``--max_cost_usd``.  The hash chain — not a numeric pin — is what makes
    this ledger tamper-evident.

    Returns the verified spent amount as Decimal."""
    from federated_mcts.evaluation.game24_budget import (
        LedgerFormatError,
        read_verified_ledger,
    )

    if not os.path.exists(budget_path):
        raise ValueError(f"B0 budget file not found: {budget_path}")

    try:
        verified = read_verified_ledger(budget_path)
    except LedgerFormatError as exc:
        raise ValueError(f"B0 budget hash chain invalid: {exc}") from exc

    spent = verified.spent
    if not spent.is_finite():
        raise ValueError(f"B0 budget spent must be finite, got {spent!r}")
    if spent < 0:
        raise ValueError(f"B0 budget spent must be nonnegative, got {spent!r}")
    # Normalize at raw-unit precision using half-up so a binary-float
    # representation tail is absorbed rather than read as a real shortfall, and
    # so the value cross-reconciles exactly against the performance artifact.
    return spent.quantize(RAW_UNIT_PRECISION, rounding=ROUND_HALF_UP)
