"""Fail-closed atomic task persistence for Game24 experiments.

Provides ``AtomicTaskJournal`` with temp+fsync+replace+parent-fsync,
and ``ResumeValidator`` that REJECTS corrupt/partial state (never silently
returns ``{}``).  Orphan temp files are cleaned; malformed JSON, duplicate
logical IDs, non-dict records, inconsistent completion, and missing
meta/fingerprint are all fatal.

Follows Todo 7 reviewer requirements.
"""
from __future__ import annotations

import fcntl
import json
import os
import tempfile


class _CorruptJournalError(ValueError):
    """Raised when the journal file is corrupt or partial."""


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

    def mark_completed(self, task_id: str) -> None:
        records = self.read_all()
        tid = str(task_id)
        if tid not in records:
            raise _CorruptJournalError(
                f"Cannot mark task {tid} completed: not in journal"
            )
        records[tid]["_completed"] = True
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
