"""Fail-closed atomic task persistence for Game24 experiments.

Provides ``AtomicTaskJournal`` with temp+fsync+replace+parent-fsync,
and ``ResumeValidator`` that REJECTS corrupt/partial state (never silently
returns ``{}``).  Orphan temp files are cleaned; malformed JSON, duplicate
logical IDs, non-dict records, inconsistent completion, and missing
meta/fingerprint are all fatal.

The ``TaskCommitter``/``RunJournal`` seam turns one task's outcome, usage and
``Decimal`` cost into a single idempotent commit ordered so that a crash is
recoverable in both directions:

1. the task record is written with ``commit_state="pending"`` (the outcome is
   durable before any money is recorded);
2. the keyed spend is appended to the hash-chained ledger;
3. the record is marked ``commit_state="settled"``.

A crash after (1) is recovered by charging the recorded cost; a crash after
(2) is recovered without charging twice because the ledger entry carries the
same idempotency key.  A task is never treated as completed before it is
charged, and recorded money is never silently discarded.
"""
from __future__ import annotations

import fcntl
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from federated_mcts.evaluation.game24_budget import (
    BudgetLedger,
    SpendStatus,
    read_verified_ledger,
)


class _CorruptJournalError(ValueError):
    """Raised when the journal file is corrupt or partial."""


class CommitError(ValueError):
    """A task commit/recovery violated the journal+ledger protocol."""


class CommitConflictError(CommitError):
    """An idempotency key or recorded cost disagrees with existing evidence."""


class BudgetExhaustedError(CommitError):
    """A recovery charge would exceed the ledger cap; nothing is dropped."""

# ── Atomic journal ─────────────────────────────────────────────────

class AtomicTaskJournal:
    """Fail-closed temp+fsync+rename atomic journal."""

    def __init__(self, journal_path: str) -> None:
        self._path = journal_path
        self._lockfile = journal_path + ".lock"

    @property
    def path(self) -> str:
        return self._path

    # ── Read (fail-closed — never returns {} silently) ──────────

    def read_all(self) -> dict[str, dict]:
        """Read journal.  Raises ``_CorruptJournalError`` on corruption."""
        if not os.path.exists(self._path):
            return {}
        raw: str
        try:
            with open(self._path, encoding="utf-8") as f:
                raw = f.read()
        except OSError as e:
            raise _CorruptJournalError(f"Cannot read journal {self._path}: {e}") from e

        if raw.strip() == "":
            raise _CorruptJournalError("Journal file exists but is empty")

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise _CorruptJournalError(f"Journal is not valid JSON: {e}") from e

        if not isinstance(data, dict):
            raise _CorruptJournalError(
                f"Journal root must be a JSON object, got {type(data).__name__}"
            )

        records: dict[str, dict] = {}
        for k, v in data.items():
            kid = str(k)
            if not isinstance(v, dict):
                raise _CorruptJournalError(
                    f"Journal record {kid!r} must be a dict, got {type(v).__name__}"
                )
            if kid in records:
                raise _CorruptJournalError(f"Duplicate task ID {kid!r} in journal")
            state = v.get("commit_state")
            if state is not None and state not in ("pending", "settled"):
                raise _CorruptJournalError(
                    f"Journal record {kid!r} has unknown commit_state {state!r}"
                )
            records[kid] = v
        return records

    def completed_ids(self) -> set[str]:
        records = self.read_all()
        completed: set[str] = set()
        for k, rec in records.items():
            if k.startswith("_"):
                continue
            if rec.get("_completed", False):
                completed.add(k)
        for k, rec in records.items():
            if k.startswith("_"):
                continue
            if rec.get("_completed") is True and k not in completed:
                raise _CorruptJournalError(
                    f"Inconsistent _completed flag for task {k}"
                )
        return completed

    def get_manifest_fingerprint(self) -> str:
        records = self.read_all()
        meta = records.get("_meta", {})
        if not isinstance(meta, dict):
            raise _CorruptJournalError("_meta entry is not a dict")
        return str(meta.get("run_fingerprint", ""))

    # ── Write (atomic) ──────────────────────────────────────────

    def write_task(self, task_id: str, task_record: dict) -> None:
        records = self.read_all()
        tid = str(task_id)
        if tid in records:
            raise _CorruptJournalError(
                f"Task {tid} already exists in journal"
            )
        if any(v for k, v in records.items()
               if not k.startswith("_") and k != tid and
               str(v.get("task_id", "")) == tid):
            raise _CorruptJournalError(
                f"Duplicate logical task ID {tid!r} detected"
            )
        records[tid] = task_record
        self._atomic_replace(records)

    def put_task(self, task_id: str, task_record: dict) -> None:
        """Insert or replace one task record atomically.

        Unlike :meth:`write_task` this is an upsert, used by the commit
        protocol to create a ``pending`` record and to refresh one.  Replacing
        a record whose ``task_id`` collides with a *different* logical ID is
        still rejected.
        """
        records = self.read_all()
        tid = str(task_id)
        if any(v for k, v in records.items()
               if not k.startswith("_") and k != tid and
               str(v.get("task_id", "")) == tid):
            raise _CorruptJournalError(
                f"Duplicate logical task ID {tid!r} detected"
            )
        records[tid] = task_record
        self._atomic_replace(records)

    def mark_settled(self, task_id: str) -> None:
        """Mark a pending task record settled (completed and charged).

        Idempotent.  A legacy record with no ``commit_state`` is upgraded to
        ``settled``; an unknown state is corruption.
        """
        records = self.read_all()
        tid = str(task_id)
        if tid not in records:
            raise _CorruptJournalError(
                f"Cannot mark task {tid} settled: not in journal"
            )
        record = records[tid]
        state = record.get("commit_state")
        if state == "settled":
            return
        if state not in (None, "pending"):
            raise _CorruptJournalError(
                f"Cannot mark task {tid} settled from state {state!r}"
            )
        record["_completed"] = True
        record["commit_state"] = "settled"
        self._atomic_replace(records)

    def write_meta(self, *, run_fingerprint: str, **extra: object) -> None:
        records = self.read_all()
        meta_obj = records.get("_meta", {})
        if not isinstance(meta_obj, dict):
            raise _CorruptJournalError("_meta is corrupt")
        meta: dict[str, object] = meta_obj
        meta["run_fingerprint"] = run_fingerprint
        meta.update(extra)
        records["_meta"] = meta
        self._atomic_replace(records)

    # ── Internal ────────────────────────────────────────────────

    def _atomic_replace(self, records: dict) -> None:
        dirname = os.path.dirname(self._path) or "."
        os.makedirs(dirname, exist_ok=True)

        fd = os.open(self._lockfile, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            tmp = tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", dir=dirname, delete=False,
                encoding="utf-8",
            )
            tmp_name = tmp.name
            try:
                json.dump(records, tmp, indent=2, sort_keys=True,
                          ensure_ascii=False)
                tmp.flush()
                os.fsync(tmp.fileno())
                tmp.close()
                os.replace(tmp_name, self._path)
                # fsync parent directory so rename is durable
                dir_fd = os.open(dirname, os.O_RDONLY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
            except Exception:
                tmp.close()
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass
                raise
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)


# ── Resume validator ───────────────────────────────────────────────

class ResumeValidator:
    """Fail-closed resume: reject corrupt/mismatched journals."""

    def __init__(
        self,
        journal: AtomicTaskJournal,
        expected_run_fingerprint: str,
    ) -> None:
        self._journal = journal
        self._expected = expected_run_fingerprint

    def validate_resume(
        self, task_ids: list[int],
    ) -> tuple[bool, set[str], str]:
        """Return ``(can_continue, completed_ids, reject_reason)``."""
        # Fresh run
        if not self._expected:
            return True, set(), ""

        # Read (raises _CorruptJournalError on corruption)
        try:
            records = self._journal.read_all()
        except _CorruptJournalError as e:
            return False, set(), f"Corrupt journal: {e}"

        if not records:
            return True, set(), ""

        # Validate _meta
        meta = records.get("_meta")
        if not isinstance(meta, dict):
            return False, set(), "_meta entry missing or corrupt"
        stored_fp = str(meta.get("run_fingerprint", ""))
        if not stored_fp:
            return False, set(), "_meta.run_fingerprint missing"

        # Fingerprint match
        if stored_fp != self._expected:
            return False, set(), (
                f"Fingerprint mismatch: stored={stored_fp[:12]}... "
                f"expected={self._expected[:12]}..."
            )

        # Validate all records
        seen_ids: set[str] = set()
        for k, rec in records.items():
            if k.startswith("_"):
                continue
            if not isinstance(rec, dict):
                return False, set(), f"Task {k} record is not a dict"
            task_id_val = str(rec.get("task_id", k))
            if task_id_val in seen_ids:
                return False, set(), f"Duplicate task_id {task_id_val!r}"
            seen_ids.add(task_id_val)

        completed = self._journal.completed_ids()

        # Skip completed
        skip_ids: set[str] = set()
        for tidx in task_ids:
            sid = str(tidx)
            if sid in completed:
                skip_ids.add(sid)

        return True, skip_ids, ""


# ── Idempotent task commit (journal + ledger) ──────────────────────

@dataclass(frozen=True)
class RecoveryReport:
    """Result of :meth:`TaskCommitter.recover`.

    ``settled_ids`` is the safe skip-set for a resumed run.  A run is
    ``fully_reconciled`` only when every settled record is backed by a keyed
    ledger entry and the ledger holds no unmatched spend for this run.
    """

    settled_ids: frozenset[str]
    recovered_ids: frozenset[str]
    legacy_unmetered_ids: frozenset[str]
    orphan_spend_keys: frozenset[str]
    spent: Decimal

    @property
    def fully_reconciled(self) -> bool:
        """True when no legacy unmetered record and no orphan spend remains."""
        return not self.legacy_unmetered_ids and not self.orphan_spend_keys


class TaskCommitter:
    """Idempotent commit seam over one journal and one budget ledger.

    Construct once per run with the run's durable identity (normally the
    manifest ``run_fingerprint``).  ``commit`` records the outcome *before*
    charging and then settles; ``recover`` replays any interrupted commit and
    returns the set of tasks that may be skipped.

    Legacy records written by the old runner (``_completed: True`` and no
    ``commit_state``) carry no cost.  Recovery accepts them as completed but
    reports them in :attr:`RecoveryReport.legacy_unmetered_ids`; it never
    fabricates a cost for them and never silently treats the run as fully
    metered.
    """

    def __init__(
        self,
        journal: AtomicTaskJournal,
        ledger: BudgetLedger,
        *,
        run_id: str,
    ) -> None:
        if not isinstance(run_id, str) or not run_id:
            raise CommitError(f"run_id must be a non-empty string, got {run_id!r}")
        self._journal = journal
        self._ledger = ledger
        self._run_id = run_id

    @staticmethod
    def spend_key(run_id: str, task_id: str | int) -> str:
        """Return the stable ``(run, task)`` idempotency key.

        Raises :class:`CommitError` for a non-integer task id so the key can
        never silently alias a different task.
        """
        try:
            normalized = int(task_id)
        except (TypeError, ValueError) as exc:
            raise CommitError(
                f"task_id must be integer-like, got {task_id!r}"
            ) from exc
        return f"{run_id}:{normalized}"

    def commit(
        self,
        task_id: int | str,
        *,
        ys: list[str],
        task_info: dict,
        task_metrics_json: str = "",
        elapsed_ms: float = 0.0,
        usage: Mapping[str, int] | None = None,
        cost: Decimal | int | float | str = Decimal("0"),
    ) -> bool:
        """Commit one finished task: outcome, usage, and exact cost.

        Returns ``True`` when this call settled the task and ``False`` when it
        was already settled with the same cost (idempotent replay).  Raises
        :class:`CommitConflictError` for a conflicting replay,
        :class:`BudgetExhaustedError` when the ledger cap refuses the charge
        (the pending record stays for a later recovery), and ``ValueError``
        for malformed arguments.
        """
        tid = str(task_id)
        if not tid:
            raise CommitError("task_id must be non-empty")
        if not isinstance(task_info, dict):
            raise CommitError(f"task_info must be a dict, got {type(task_info).__name__}")
        if not isinstance(task_metrics_json, str):
            raise CommitError("task_metrics_json must be a string")
        elapsed = self._normalize_elapsed(elapsed_ms)
        normalized_usage = self._normalize_usage(usage)
        cost_d = self._normalize_cost(cost)
        key = self.spend_key(self._run_id, tid)

        existing = self._journal.read_all().get(tid)
        if existing is not None:
            state = existing.get("commit_state")
            if state == "settled" or (state is None and existing.get("_completed") is True):
                recorded = self._recorded_cost(existing, tid)
                if recorded is None:
                    raise CommitConflictError(
                        f"task {tid} is already completed by a legacy record "
                        "with no cost; refusing to invent a charge"
                    )
                spend = self._ledger.find_spend(key)
                if spend is None:
                    raise CommitConflictError(
                        f"task {tid} is marked settled but ledger has no spend {key!r}"
                    )
                if recorded != cost_d or spend.amount != cost_d:
                    raise CommitConflictError(
                        f"task {tid} was already committed at {recorded}, "
                        f"refusing {cost_d}"
                    )
                return False
            if state == "pending":
                pending_cost = self._recorded_cost(existing, tid)
                if pending_cost is not None and pending_cost != cost_d:
                    raise CommitConflictError(
                        f"task {tid} already has a pending commit at "
                        f"{pending_cost}, refusing {cost_d}"
                    )

        self._journal.put_task(tid, {
            "task_id": task_id,
            "ys": ys[-1] if ys else "",
            "task_info": task_info,
            "task_metrics_json": task_metrics_json,
            "elapsed_ms": elapsed,
            "usage_delta": normalized_usage,
            "cost_raw": str(cost_d),
            "spend_key": key,
            "commit_state": "pending",
            "_completed": False,
        })

        status = self._charge(key, cost_d, tid)
        if status is SpendStatus.OVER_CAP:
            raise BudgetExhaustedError(
                f"budget cap would be exceeded recording task {tid} "
                f"({cost_d} raw units); the pending record is retained"
            )
        self._journal.mark_settled(tid)
        return True

    def recover(self) -> RecoveryReport:
        """Finish interrupted commits and return what may be skipped.

        Fail-closed: a corrupt or chain-broken ledger, a corrupt journal, or a
        settled record whose keyed spend is missing/mismatched raises before
        any journal state is changed.
        """
        verified = read_verified_ledger(self._ledger.path)
        records = self._journal.read_all()

        settled: set[str] = set()
        recovered: set[str] = set()
        legacy: set[str] = set()
        unmatched = {
            key for key, spend in verified.spend_keys.items()
            if spend.run_id == self._run_id
        }

        for tid, record in records.items():
            if tid.startswith("_"):
                continue
            if record.get("execution_state") == "running":
                raise CommitError(
                    f"task {tid} was interrupted in-flight before metered commit; "
                    "usage is unknown, refusing automatic replay"
                )
            state = record.get("commit_state")
            if state == "pending":
                if record.get("_completed") is True:
                    raise CommitError(
                        f"task {tid} is both pending and completed"
                    )
                cost, key = self._pending_commit(record, tid)
                status = self._charge(key, cost, tid)
                if status is SpendStatus.OVER_CAP:
                    raise BudgetExhaustedError(
                        f"cannot recover task {tid}: cap would be exceeded "
                        f"recording {cost} raw units; evidence left intact"
                    )
                self._journal.mark_settled(tid)
                settled.add(tid)
                recovered.add(tid)
                unmatched.discard(key)
            elif state == "settled" or (
                state is None and record.get("_completed") is True
            ):
                settled_key = record.get("spend_key")
                if settled_key:
                    spend = verified.find_spend(str(settled_key))
                    settled_cost = self._recorded_cost(record, tid)
                    if spend is None or settled_cost is None:
                        raise CommitConflictError(
                            f"task {tid} is settled but has no keyed ledger spend "
                            f"{settled_key!r}"
                        )
                    if spend.run_id != self._run_id or spend.task_id != tid:
                        raise CommitConflictError(
                            f"spend {settled_key!r} does not belong to run/task "
                            f"{self._run_id}/{tid}"
                        )
                    if spend.amount != settled_cost:
                        raise CommitConflictError(
                            f"task {tid} recorded cost {settled_cost} but ledger "
                            f"holds {spend.amount}"
                        )
                    unmatched.discard(str(settled_key))
                else:
                    legacy.add(tid)
                settled.add(tid)
            elif state is None:
                raise CommitError(
                    f"task {tid} is an incomplete legacy record "
                    "(_completed is not true and no commit_state)"
                )
            else:
                raise CommitError(
                    f"task {tid} has unknown commit_state {state!r}"
                )

        return RecoveryReport(
            settled_ids=frozenset(settled),
            recovered_ids=frozenset(recovered),
            legacy_unmetered_ids=frozenset(legacy),
            orphan_spend_keys=frozenset(unmatched),
            spent=self._ledger.spent_exact,
        )

    # ── Internal seam (tests inject crashes here) ───────────────

    def _charge(
        self, key: str, cost: Decimal, task_id: str,
    ) -> SpendStatus:
        return self._ledger.spend_idempotent(
            key=key, amount=cost, run_id=self._run_id, task_id=task_id,
        )

    @staticmethod
    def _normalize_elapsed(elapsed_ms: object) -> float:
        if isinstance(elapsed_ms, bool) or not isinstance(
            elapsed_ms, (int, float)
        ):
            raise CommitError(
                f"elapsed_ms must be a number, got {type(elapsed_ms).__name__}"
            )
        value = float(elapsed_ms)
        if value != value or value in (float("inf"), float("-inf")):
            raise CommitError(f"elapsed_ms must be finite, got {elapsed_ms!r}")
        if value < 0:
            raise CommitError(f"elapsed_ms must be non-negative, got {value}")
        return value

    @staticmethod
    def _normalize_usage(usage: Mapping[str, int] | None) -> dict[str, int]:
        if usage is None:
            return {}
        if not isinstance(usage, Mapping):
            raise CommitError(
                f"usage must be a mapping, got {type(usage).__name__}"
            )
        normalized: dict[str, int] = {}
        for name, value in usage.items():
            if not isinstance(name, str) or not name:
                raise CommitError(f"usage key must be a non-empty string: {name!r}")
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise CommitError(
                    f"usage value for {name!r} must be a non-negative int, "
                    f"got {value!r}"
                )
            normalized[name] = value
        return normalized

    @staticmethod
    def _normalize_cost(cost: object) -> Decimal:
        if isinstance(cost, bool) or not isinstance(
            cost, (Decimal, int, float, str)
        ):
            raise CommitError(
                f"cost must be a Decimal-compatible number, got {type(cost).__name__}"
            )
        try:
            value = Decimal(str(cost))
        except (InvalidOperation, ValueError) as exc:
            raise CommitError(f"cost must be a decimal number, got {cost!r}") from exc
        if not value.is_finite():
            raise CommitError(f"cost must be finite, got {cost!r}")
        if value < 0:
            raise CommitError(f"cost must be non-negative, got {value}")
        return value

    @classmethod
    def _recorded_cost(cls, record: Mapping[str, object], task_id: str) -> Decimal | None:
        raw = record.get("cost_raw")
        if raw is None:
            return None
        return cls._normalize_cost(raw)

    def _pending_commit(self, record: Mapping[str, object], tid: str) -> tuple[Decimal, str]:
        key = record.get("spend_key")
        if not isinstance(key, str) or not key:
            raise CommitError(f"pending task {tid} has no spend_key")
        if key != self.spend_key(self._run_id, tid):
            raise CommitError(
                f"pending task {tid} has spend_key {key!r}, expected "
                f"{self.spend_key(self._run_id, tid)!r}"
            )
        cost = self._recorded_cost(record, tid)
        if cost is None:
            raise CommitError(f"pending task {tid} has no recorded cost")
        return cost, key


RunJournal = TaskCommitter
"""Alias: the journal+ledger commit seam for one run."""
