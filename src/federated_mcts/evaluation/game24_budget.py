"""Shared raw-unit aggregate budget ledger.

Provides a thread-safe, fsync-backed, lock-protected, hash-chained
append-only budget ledger.  All planned experiments within one scope
share a single raw-unit budget (default 19.99).  Separate caps for API
calls and tokens are also supported and reported as incomplete stop
states when exceeded.

No currency conversion is performed — raw units are the canonical
budget dimension.  Cost caps (`--max_cost_usd`) are enforced at the
runner level alongside this ledger.

Follows the contract from Todo 3 of the game24-model-only-improvements plan.
"""

from __future__ import annotations

import enum
import fcntl
import hashlib
import json
import math
import os
import threading
import time
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from types import MappingProxyType
from typing import Mapping


DEFAULT_RAW_UNIT_LIMIT: float = 19.99
"""Default total budget in raw units."""


class LedgerFormatError(ValueError):
    """A ledger could not be parsed, is truncated, or fails hash verification."""


class LedgerKeyConflictError(LedgerFormatError):
    """One idempotency key was reused with a different amount."""


class SpendStatus(enum.Enum):
    """Outcome of an idempotent keyed spend."""

    RECORDED = "recorded"
    DUPLICATE = "duplicate"
    OVER_CAP = "over_cap"


@dataclass(frozen=True)
class LedgerSpend:
    """One ``spend`` entry, read-only.

    *key* is ``None`` for legacy entries written before idempotency keys
    existed; those entries still count toward the total but can never be
    matched back to a task.
    """

    key: str | None
    run_id: str | None
    task_id: str | None
    amount: Decimal
    entry_hash: str


@dataclass(frozen=True)
class VerifiedLedger:
    """Immutable, side-effect-free view of a verified ledger file."""

    path: str
    entries: tuple[dict, ...]
    head_hash: str
    spent: Decimal
    total_limit: float
    calls_seen: int
    tokens_seen: int
    spends: tuple[LedgerSpend, ...]
    spend_keys: Mapping[str, LedgerSpend] = field(repr=False, compare=False)

    def find_spend(self, key: str) -> LedgerSpend | None:
        """Return the keyed spend for *key*, or ``None``."""
        return self.spend_keys.get(key)


@dataclass(frozen=True)
class _ParsedLedger:
    entries: tuple[dict, ...]
    spent: Decimal
    total_limit: float
    calls_seen: int
    tokens_seen: int
    spends: tuple[LedgerSpend, ...]
    spend_keys: Mapping[str, LedgerSpend]


def _decimal_amount(name: str, value: object) -> Decimal:
    """Return *value* as a finite, non-negative Decimal, rejecting bools."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str, Decimal)):
        raise ValueError(f"{name} must be a number, got {type(value).__name__}")
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{name} must be a number, got {value!r}") from exc
    if not parsed.is_finite():
        raise ValueError(f"{name} must be finite, got {value!r}")
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative, got {parsed}")
    return parsed


def _parse_ledger_text(text: str, path: str) -> _ParsedLedger:
    """Parse and hash-verify ledger text using the single shared rule set.

    Fail-closed by construction: an empty file, a missing trailing newline
    (torn append), a blank line, malformed JSON, a broken ``prev_hash`` link,
    a recomputed ``entry_hash`` mismatch, an invalid amount, or an unknown
    record shape all raise :class:`LedgerFormatError`.  Reading never writes.
    """
    if text.strip() == "":
        raise LedgerFormatError(f"ledger {path} is empty")
    if not text.endswith("\n"):
        raise LedgerFormatError(
            f"ledger {path} is truncated (last entry lacks a trailing newline)"
        )

    entries: list[dict] = []
    prev_hash = ""
    for index, line in enumerate(text.split("\n")[:-1]):
        if line.strip() == "":
            raise LedgerFormatError(f"ledger {path} has a blank line at entry {index}")
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            raise LedgerFormatError(
                f"cannot parse ledger entry {index}: {exc}"
            ) from exc
        if not isinstance(entry, dict):
            raise LedgerFormatError(
                f"ledger entry {index} must be a JSON object, "
                f"got {type(entry).__name__}"
            )
        if entry.get("prev_hash", "") != prev_hash:
            raise LedgerFormatError(
                f"chain broken at entry {index}: prev_hash "
                f"{entry.get('prev_hash')!r} != expected {prev_hash!r}"
            )
        recomputed = _sha256_json(
            {k: v for k, v in entry.items() if k != "entry_hash"}
        )
        if recomputed != entry.get("entry_hash"):
            raise LedgerFormatError(
                f"hash mismatch at entry {index}: "
                f"{recomputed!r} != {entry.get('entry_hash')!r}"
            )
        kind = entry.get("kind")
        if not isinstance(kind, str) or kind not in {"init", "spend", "cap_exceeded"}:
            raise LedgerFormatError(f"ledger entry {index} has unknown kind {kind!r}")
        if not isinstance(entry.get("timestamp_utc"), str):
            raise LedgerFormatError(f"ledger entry {index} has no timestamp_utc")
        entries.append(entry)
        prev_hash = entry["entry_hash"]

    if not entries:
        raise LedgerFormatError(f"ledger {path} is empty")

    total_limit = _parse_total_limit(entries)
    spent = Decimal("0")
    calls_seen = 0
    tokens_seen = 0
    spends: list[LedgerSpend] = []
    spend_keys: dict[str, LedgerSpend] = {}

    for index, entry in enumerate(entries):
        kind = entry["kind"]
        if kind == "spend":
            try:
                amount = _decimal_amount(
                    f"ledger entry {index} amount", entry.get("amount")
                )
                if not math.isfinite(float(amount)):
                    raise ValueError(f"ledger entry {index} amount overflows numeric storage")
                exact = entry.get("amount_exact")
                if exact is not None:
                    exact_amount = _decimal_amount(
                        f"ledger entry {index} amount_exact", exact
                    )
                    if float(exact_amount) != float(amount):
                        raise ValueError(f"ledger entry {index} amount disagrees with amount_exact")
                    amount = exact_amount
            except ValueError as exc:
                raise LedgerFormatError(str(exc)) from exc
            key, run_id, task_id = _parse_spend_key(entry, index)
            spend = LedgerSpend(
                key=key, run_id=run_id, task_id=task_id,
                amount=amount, entry_hash=entry["entry_hash"],
            )
            if key is not None:
                if key in spend_keys:
                    raise LedgerFormatError(
                        f"duplicate spend key {key!r} at entry {index}"
                    )
                spend_keys[key] = spend
            spends.append(spend)
            spent += amount
        elif kind == "cap_exceeded":
            used = entry.get("used")
            if isinstance(used, bool) or not isinstance(used, int) or used < 0:
                raise LedgerFormatError(
                    f"ledger entry {index} has invalid cap usage {used!r}"
                )
            cap_type = entry.get("cap_type")
            if cap_type == "max_total_calls":
                calls_seen = max(calls_seen, used)
            elif cap_type == "max_total_tokens":
                tokens_seen = max(tokens_seen, used)
            else:
                raise LedgerFormatError(
                    f"ledger entry {index} has unknown cap_type {cap_type!r}"
                )

    return _ParsedLedger(
        entries=tuple(entries),
        spent=spent,
        total_limit=total_limit,
        calls_seen=calls_seen,
        tokens_seen=tokens_seen,
        spends=tuple(spends),
        spend_keys=MappingProxyType(spend_keys),
    )


def _parse_total_limit(entries: list[dict]) -> float:
    inits = [e for e in entries if e.get("kind") == "init"]
    if len(inits) != 1:
        raise LedgerFormatError(
            f"ledger must contain exactly one init entry, found {len(inits)}"
        )
    if entries[0].get("kind") != "init":
        raise LedgerFormatError("ledger must start with an init entry")
    try:
        limit = _decimal_amount("ledger total_limit", inits[0].get("total_limit"))
    except ValueError as exc:
        raise LedgerFormatError(str(exc)) from exc
    if limit <= 0 or not math.isfinite(float(limit)):
        raise LedgerFormatError(f"ledger total_limit must be finite and positive, got {limit}")
    return float(limit)


def _parse_spend_key(
    entry: dict, index: int,
) -> tuple[str | None, str | None, str | None]:
    key = entry.get("spend_key")
    run_id = entry.get("run_id")
    task_id = entry.get("task_id")
    if key is None and run_id is None and task_id is None:
        return None, None, None
    if not isinstance(key, str) or not key:
        raise LedgerFormatError(f"ledger entry {index} has invalid spend_key")
    if not isinstance(run_id, str) or not run_id:
        raise LedgerFormatError(f"ledger entry {index} has invalid run_id")
    if not isinstance(task_id, str) or not task_id:
        raise LedgerFormatError(f"ledger entry {index} has invalid task_id")
    return key, run_id, task_id


def read_verified_ledger(path: str) -> VerifiedLedger:
    """Read and hash-verify *path* without modifying a single byte.

    This is the read-only counterpart of :class:`BudgetLedger`: it shares the
    exact same parsing/hash rules, but never appends an ``init`` entry, never
    repairs a torn tail, and never creates the file.  Callers that must
    validate evidence (for example the B0 verifier) use this instead of
    constructing a writable ledger.

    Raises :class:`LedgerFormatError` on a missing, empty, truncated, corrupt,
    or chain-broken ledger.
    """
    try:
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
    except FileNotFoundError as exc:
        raise LedgerFormatError(f"ledger not found: {path}") from exc
    except OSError as exc:
        raise LedgerFormatError(f"cannot read ledger {path}: {exc}") from exc

    parsed = _parse_ledger_text(text, path)
    return VerifiedLedger(
        path=path,
        entries=parsed.entries,
        head_hash=parsed.entries[-1]["entry_hash"],
        spent=parsed.spent,
        total_limit=parsed.total_limit,
        calls_seen=parsed.calls_seen,
        tokens_seen=parsed.tokens_seen,
        spends=parsed.spends,
        spend_keys=parsed.spend_keys,
    )


def _validate_finite_nonnegative(name: str, value: float) -> float:
    """Return *value* if finite, non-NaN, nonnegative, and not bool.

    Raises ``ValueError`` otherwise.
    """
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number, got bool ({value!r})")
    v = float(value)
    if math.isnan(v) or math.isinf(v):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if v < 0.0:
        raise ValueError(f"{name} must be non-negative, got {v}")
    return v


class BudgetLedger:
    """Shared budget tracker for a set of experiments.

    Thread-safe.  Survives process restarts.  Supports:
    - Raw-unit spending with cap enforcement
    - API call limits
    - Token limits
    - Incomplete stop states (``stop_reason`` string)
    """

    def __init__(
        self,
        ledger_path: str,
        *,
        total_limit: float = DEFAULT_RAW_UNIT_LIMIT,
        max_calls: int | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self._path = ledger_path
        self._lock = threading.Lock()

        # Proposed caps — may be overridden by persistent state
        self._total_limit = _validate_finite_nonnegative("total_limit", total_limit)
        self._max_calls = max_calls
        self._max_tokens = max_tokens

        # Load existing state from disk
        self._spent: float = 0.0
        # Authoritative exact sum of the per-spend ``amount`` fields.  Raw units
        # are a decimal dimension, so the ledger must not accumulate them through
        # binary floats: 20 recorded spends whose true sum is 0.1664672 drift to
        # 0.16646719999999998, which ROUND_DOWN truncated to 0.16646719 and made
        # exact reconciliation against the canonical pinned value impossible.
        self._spent_exact: Decimal = Decimal("0")
        self._chain: list[dict] = []
        self._chain_head: str = ""
        self._spend_keys: dict[str, LedgerSpend] = {}
        self._calls_seen: int = 0
        self._tokens_seen: int = 0
        self._started_tasks: int = 0
        self._load_existing()

        # If ledger is empty, write the init entry with total_limit
        if not self._chain:
            self._record_init_entry()

    # ── Public properties ─────────────────────────────────────────

    @property
    def remaining(self) -> float:
        """Raw units remaining."""
        return float(max(Decimal("0"), Decimal(str(self._total_limit)) - self._spent_exact))

    @property
    def spent(self) -> float:
        """Raw units spent so far."""
        return self._spent

    @property
    def spent_exact(self) -> Decimal:
        """Exact decimal sum of recorded spend amounts.

        Authoritative counterpart to :attr:`spent`, which is a binary float and
        therefore carries representation drift.  Reconciliation against pinned
        expected values must use this property.
        """
        return self._spent_exact

    @property
    def started_tasks(self) -> int:
        """Number of tasks started (for metadata)."""
        return self._started_tasks

    @property
    def chain_length(self) -> int:
        return len(self._chain)

    @property
    def path(self) -> str:
        """Filesystem path of the ledger (read-only inspection)."""
        return self._path

    # ── Spend operations ──────────────────────────────────────────

    def spend(self, amount: float) -> bool:
        """Attempt to spend *amount* raw units from the budget.

        Returns ``True`` if the spend was recorded, ``False`` if it
        would exceed the limit.  Raises ``ValueError`` for NaN, inf,
        bool, or negative *amount*.
        """
        amount = _validate_finite_nonnegative("spend amount", amount)
        if amount == 0.0:
            return True

        with self._lock:
            if self._spent + amount > self._total_limit:
                return False
            self._spent += amount
            self._spent_exact += Decimal(str(amount))
            self._record_entry(
                "spend", {"amount": amount, "remaining": self.remaining}
            )
            return True

    def find_spend(self, key: str) -> LedgerSpend | None:
        """Return the keyed spend for *key*, or ``None``.

        Legacy unkeyed entries are invisible here by design: they count toward
        the total but cannot be attributed to one run/task.
        """
        if not isinstance(key, str) or not key:
            raise ValueError(f"spend key must be a non-empty string, got {key!r}")
        with self._lock:
            return self._spend_keys.get(key)

    def spend_idempotent(
        self,
        *,
        key: str,
        amount: Decimal | int | float | str,
        run_id: str,
        task_id: str,
    ) -> SpendStatus:
        """Record *amount* against ``(run_id, task_id)`` at most once.

        The idempotency identity is the explicit ``spend_key`` *key*; the
        ``run_id``/``task_id`` fields are persisted alongside it so one shared
        ledger stays attributable across runs and tasks.  Replaying the same
        key with the same amount returns :attr:`SpendStatus.DUPLICATE` and
        writes nothing; replaying it with a different amount raises
        :class:`LedgerKeyConflictError`.

        Returns :attr:`SpendStatus.OVER_CAP` without writing when recording
        would exceed ``total_limit``.  Zero-amount spends are recorded (unlike
        legacy :meth:`spend`) because the keyed entry is the durable marker
        that recovery needs.
        """
        if not isinstance(key, str) or not key:
            raise ValueError(f"spend key must be a non-empty string, got {key!r}")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError(f"run_id must be a non-empty string, got {run_id!r}")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError(f"task_id must be a non-empty string, got {task_id!r}")
        amount_d = _decimal_amount("spend amount", amount)

        with self._lock:
            self._reload_from_disk()
            existing = self._spend_keys.get(key)
            if existing is not None:
                if existing.amount != amount_d:
                    raise LedgerKeyConflictError(
                        f"spend key {key!r} already recorded {existing.amount}, "
                        f"refusing different amount {amount_d}"
                    )
                return SpendStatus.DUPLICATE
            if self._spent_exact + amount_d > Decimal(str(self._total_limit)):
                return SpendStatus.OVER_CAP
            self._spent_exact += amount_d
            self._spent = float(self._spent_exact)
            self._record_entry(
                "spend",
                {
                    "amount": float(amount_d),
                    "amount_exact": str(amount_d),
                    "remaining": self.remaining,
                    "spend_key": key,
                    "run_id": run_id,
                    "task_id": task_id,
                },
            )
            self._spend_keys[key] = LedgerSpend(
                key=key,
                run_id=run_id,
                task_id=task_id,
                amount=amount_d,
                entry_hash=self._chain_head,
            )
            return SpendStatus.RECORDED

    def check_and_mark_calls(
        self, used_calls: int,
    ) -> tuple[bool, str | None]:
        """Check if *used_calls* exceeds the call cap.

        Returns ``(True, None)`` if under cap, or
        ``(False, "max_total_calls")`` if exceeded.
        """
        if self._max_calls is None:
            return True, None

        with self._lock:
            self._calls_seen = max(self._calls_seen, used_calls)
            if used_calls > self._max_calls:
                self._record_entry(
                    "cap_exceeded",
                    {"cap_type": "max_total_calls", "used": used_calls},
                )
                return False, "max_total_calls"
            return True, None

    def check_and_mark_tokens(
        self, used_tokens: int,
    ) -> tuple[bool, str | None]:
        """Check if *used_tokens* exceeds the token cap.

        Returns ``(True, None)`` if under cap, or
        ``(False, "max_total_tokens")`` if exceeded.
        """
        if self._max_tokens is None:
            return True, None

        with self._lock:
            self._tokens_seen = max(self._tokens_seen, used_tokens)
            if used_tokens > self._max_tokens:
                self._record_entry(
                    "cap_exceeded",
                    {"cap_type": "max_total_tokens", "used": used_tokens},
                )
                return False, "max_total_tokens"
            return True, None

    def mark_task_started(self) -> None:
        """Record that a new task was started (for metadata)."""
        with self._lock:
            self._started_tasks += 1

    # ── Verification ──────────────────────────────────────────────

    def verify_chain(self) -> tuple[bool, str]:
        """Verify the hash chain of the on-disk ledger.

        Returns ``(True, "")`` if valid, or ``(False, reason)``.
        Delegates to the same fail-closed parser used by
        :func:`read_verified_ledger`; never rewrites the file.
        """
        try:
            with open(self._path, encoding="utf-8") as handle:
                text = handle.read()
        except OSError as exc:
            return False, f"Cannot parse ledger: {exc}"
        try:
            _parse_ledger_text(text, self._path)
        except LedgerFormatError as exc:
            return False, f"Cannot parse ledger: {exc}"
        return True, ""

    # ── Internal helpers ──────────────────────────────────────────

    def _record_entry(self, kind: str, data: dict) -> None:
        """Append a new entry to the on-disk ledger."""
        entry = {
            "kind": kind,
            "timestamp_utc": _utc_now_iso(),
            "prev_hash": self._chain_head,
        }
        entry.update(data)
        entry["entry_hash"] = _sha256_json(entry)
        line = _canonical_json_str(entry) + "\n"

        self._append_line(line)
        self._chain.append(entry)
        self._chain_head = entry["entry_hash"]

    def _append_line(self, line: str) -> None:
        fd = os.open(self._path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            os.write(fd, line.encode("utf-8"))
            os.fsync(fd)
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def _load_existing(self) -> None:
        """Replay the ledger from disk using the shared fail-closed parser.

        A missing file is a fresh ledger.  An existing file that is empty,
        truncated, corrupt, or chain-broken raises :class:`LedgerFormatError`
        instead of being silently replaced with a new ``init`` entry.
        """
        if not os.path.exists(self._path):
            return
        self._reload_from_disk()

    def _reload_from_disk(self) -> None:
        try:
            with open(self._path, encoding="utf-8") as handle:
                text = handle.read()
        except OSError as exc:
            raise LedgerFormatError(f"cannot read ledger {self._path}: {exc}") from exc
        parsed = _parse_ledger_text(text, self._path)
        self._total_limit = parsed.total_limit
        self._spent = float(parsed.spent)
        self._spent_exact = parsed.spent
        self._calls_seen = parsed.calls_seen
        self._tokens_seen = parsed.tokens_seen
        self._chain = list(parsed.entries)
        self._chain_head = parsed.entries[-1]["entry_hash"]
        self._spend_keys = dict(parsed.spend_keys)

    def _record_init_entry(self) -> None:
        """Write the first entry recording total_limit."""
        self._record_entry("init", {"total_limit": self._total_limit})


# ═══════════════════════════════════════════════════════════════════
#  Internal helpers
# ═══════════════════════════════════════════════════════════════════

def _sha256_json(obj: object) -> str:
    data = _canonical_json_str(obj).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _canonical_json_str(obj: object) -> str:
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    )


def _utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
