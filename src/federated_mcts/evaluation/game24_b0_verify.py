"""B0 evidence sub-verifiers: journal completeness and budget hash-chain.

Extracted from ``game24_b0_validate.py`` to keep each module ≤250 pure LOC.
Zero side effects; all checks raise ``ValueError`` on failure.
"""

from __future__ import annotations

import os
from decimal import ROUND_DOWN, Decimal

from federated_mcts.evaluation.game24_b0_evidence import (
    B0_EXPECTED_SPENT,
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

    Returns the verified spent amount as Decimal."""
    from federated_mcts.evaluation.game24_budget import BudgetLedger

    if not os.path.exists(budget_path):
        raise ValueError(f"B0 budget file not found: {budget_path}")

    ledger = BudgetLedger(budget_path, total_limit=float(Decimal("19.99")))
    ok, reason = ledger.verify_chain()
    if not ok:
        raise ValueError(f"B0 budget hash chain invalid: {reason}")

    spent = Decimal(str(ledger.spent))
    if not spent.is_finite():
        raise ValueError(f"B0 budget spent must be finite, got {spent!r}")
    if spent < 0:
        raise ValueError(f"B0 budget spent must be nonnegative, got {spent!r}")
    spent_norm = spent.quantize(RAW_UNIT_PRECISION, rounding=ROUND_DOWN)
    expected_norm = B0_EXPECTED_SPENT.quantize(RAW_UNIT_PRECISION, rounding=ROUND_DOWN)
    if spent_norm != expected_norm:
        raise ValueError(
            f"B0 budget spent: expected {B0_EXPECTED_SPENT} (normalized {expected_norm}), "
            f"got {spent} (normalized {spent_norm})"
        )
    return spent_norm
