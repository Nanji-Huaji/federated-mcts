"""Post-decision Game24 oracle audit: canonical bytes → frozen disjoint annotations."""
from __future__ import annotations
from collections.abc import Callable
import hashlib
import json
import math
import re
from dataclasses import dataclass
from fractions import Fraction
from federated_mcts.tasks.game24_exact_prune import _can_reach_target_numbers

SNAPSHOT_SCHEMA_VERSION: int = 1
ORACLE_ASSISTED_DECISION_POLICIES: frozenset[str] = frozenset({"oracle_exact_pruned"})
VALID_EVALUATOR_MODES: frozenset[str] = frozenset({"vote", "value"})
@dataclass(frozen=True, slots=True)
class PostDecisionCandidate:
    idx: int
    raw_text: str
    canonical_numbers: tuple[Fraction, ...]
    evaluator_score: float | None
    evaluator_mode: str | None
    is_selected: bool
    is_terminal: bool


@dataclass(frozen=True, slots=True)
class PostDecisionSnapshot:
    schema_version: int
    task_id: str
    task_input: str
    depth: int
    candidates: tuple[PostDecisionCandidate, ...]
    selected_indices: tuple[int, ...]
    prompt_call_identity: str
    is_stopping: bool
    decision_policy: str


@dataclass(frozen=True, slots=True)
class DepthAuditMetrics:
    depth: int
    proposed_count: int
    reachable_count: int
    unreachable_count: int
    selected_count: int
    selected_reachable_count: int
    best_reachable_idx: int | None
    best_reachable_score: float | None
    best_selected_idx: int | None
    best_selected_score: float | None
    selection_regret: float | None
    selected_precision: float

@dataclass(frozen=True, slots=True)
class OracleAuditAnnotations:
    task_id: str
    total_depths: int
    depth_metrics: tuple[DepthAuditMetrics, ...]
    per_candidate_reachability: tuple[tuple[int, int, bool], ...]
    aggregate_selection_regret: float | None
    snapshot_hash: str

# ═══ Canonical serialization ═══

def _candidate_to_dict(c: PostDecisionCandidate) -> dict:
    return {
        "idx": c.idx, "raw_text": c.raw_text,
        "canonical_numbers": [str(n) for n in c.canonical_numbers],
        "evaluator_score": c.evaluator_score,
        "evaluator_mode": c.evaluator_mode,
        "is_selected": c.is_selected, "is_terminal": c.is_terminal,
    }


def _canonical_json_bytes(obj: object) -> bytes:
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")


# ── Boundary validators (run BEFORE frozen construction) ──


def _check_finite_float(value: object, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a number, not bool")
    if not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be float or None; got {type(value).__name__}")
    f = float(value)
    if math.isnan(f) or math.isinf(f):
        raise ValueError(f"{label} must be finite; got {value!r}")
    return f


def _frac(s: str) -> Fraction:
    try:
        return Fraction(s)
    except (ValueError, ZeroDivisionError):
        raise ValueError(f"unparseable Fraction: {s!r}")


def _validate_boundary(candidates: list, selected_indices: list) -> frozenset[int]:
    """Full validation: exact types, no duplicates, consistency, mode values."""
    known: set[int] = set()
    for i, c in enumerate(candidates):
        p = f"candidates[{i}]"
        if type(c.get("idx")) is not int:
            raise ValueError(f"{p}.idx not int")
        idx = c["idx"]
        if idx in known:
            raise ValueError(f"{p}.idx={idx} duplicate")
        known.add(idx)
        if type(c.get("raw_text")) is not str:
            raise ValueError(f"{p}.raw_text must be str")
        cn = c.get("canonical_numbers")
        if not isinstance(cn, list):
            raise ValueError(f"{p}.canonical_numbers must be a list")
        for j, s in enumerate(cn):
            if not isinstance(s, str) or s.strip() == "":
                raise ValueError(f"{p}.canonical_numbers[{j}] must be non-empty str")
        _check_finite_float(c.get("evaluator_score"), f"{p}.evaluator_score")
        mode = c.get("evaluator_mode")
        if mode is not None and mode not in VALID_EVALUATOR_MODES:
            raise ValueError(f"{p}.evaluator_mode={mode!r} invalid (expect None/vote/value)")
        for bf in ("is_selected", "is_terminal"):
            if type(c.get(bf)) is not bool:
                raise ValueError(f"{p}.{bf} must be bool")
    sel_set: set[int] = set()
    for j, sid in enumerate(selected_indices):
        if type(sid) is not int:
            raise ValueError(f"selected_indices[{j}] must be int")
        if sid in sel_set:
            raise ValueError(f"selected_indices[{j}]={sid} duplicate")
        sel_set.add(sid)
        if sid not in known:
            raise ValueError(f"selected_indices[{j}]={sid} unknown")
    for c in candidates:
        cid = c["idx"]
        if c["is_selected"] is not (cid in sel_set):
            raise ValueError(
                f"candidate idx={cid} is_selected={c['is_selected']} "
                f"inconsistent with selected_indices")
    return frozenset(known)


# ── Deserialize (the audit boundary entrypoint) ──


def deserialize_audit_input(data: bytes) -> PostDecisionSnapshot:
    """Parse canonical bytes → PostDecisionSnapshot with full boundary validation."""
    if not isinstance(data, bytes):
        raise ValueError(f"audit input must be bytes; got {type(data).__name__}")
    if len(data) == 0:
        raise ValueError("audit input must not be empty bytes")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError("audit input is not valid UTF-8")

    # Reject JSON NaN / Infinity value tokens (Python json.loads accepts them)
    if re.search(r'(?:^|[^-."\w])NaN(?:$|[^"\w])', text):
        raise ValueError("JSON NaN token rejected")
    if re.search(r'(?:^|[^-."\w])(?:-)?Infinity(?:$|[^"\w])', text):
        raise ValueError("JSON Infinity token rejected")

    try:
        d = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"audit input is not valid JSON: {exc}")

    # Schema-level validation (inline to save a function boundary)
    v = d.get("schema_version")
    if type(v) is not int or v != SNAPSHOT_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SNAPSHOT_SCHEMA_VERSION}; got {v!r}")
    for key in ("task_id", "task_input", "prompt_call_identity", "decision_policy"):
        if not isinstance(d.get(key), str):
            raise ValueError(f"snapshot.{key} must be str")
    for key in ("depth",):
        if type(d.get(key)) is not int:
            raise ValueError(f"snapshot.{key} must be int")
    if type(d.get("is_stopping")) is not bool:
        raise ValueError("snapshot.is_stopping must be bool")
    for key in ("candidates", "selected_indices"):
        if not isinstance(d.get(key), list):
            raise ValueError(f"snapshot.{key} must be a list")

    _validate_boundary(d["candidates"], d["selected_indices"])

    return PostDecisionSnapshot(
        schema_version=d["schema_version"], task_id=d["task_id"],
        task_input=d["task_input"], depth=d["depth"],
        candidates=tuple(
            PostDecisionCandidate(
                idx=c["idx"], raw_text=c["raw_text"],
                canonical_numbers=tuple(sorted(
                    _frac(s) for s in c["canonical_numbers"])),
                evaluator_score=_check_finite_float(
                    c["evaluator_score"], f"candidates[{c['idx']}].evaluator_score"),
                evaluator_mode=c["evaluator_mode"],
                is_selected=c["is_selected"], is_terminal=c["is_terminal"],
            ) for c in d["candidates"]
        ),
        selected_indices=tuple(d["selected_indices"]),
        prompt_call_identity=d["prompt_call_identity"],
        is_stopping=d["is_stopping"], decision_policy=d["decision_policy"],
    )


def serialize_snapshot(snapshot: PostDecisionSnapshot) -> bytes:
    return _canonical_json_bytes({
        "schema_version": snapshot.schema_version,
        "task_id": snapshot.task_id, "task_input": snapshot.task_input,
        "depth": snapshot.depth,
        "candidates": [_candidate_to_dict(c) for c in snapshot.candidates],
        "selected_indices": list(snapshot.selected_indices),
        "prompt_call_identity": snapshot.prompt_call_identity,
        "is_stopping": snapshot.is_stopping,
        "decision_policy": snapshot.decision_policy,
    })


# ═══ Audit runner ═══


def _best_from_candidates(
    candidates: tuple[PostDecisionCandidate, ...],
    *,
    predicate: Callable[[PostDecisionCandidate], bool],
) -> tuple[int | None, float | None]:
    best_idx: int | None = None
    best_score: float | None = None
    for c in candidates:
        if not predicate(c) or c.evaluator_score is None:
            continue
        if best_score is None or c.evaluator_score > best_score:
            best_score = c.evaluator_score
            best_idx = c.idx
    return best_idx, best_score


def run_audit(snapshot: PostDecisionSnapshot) -> OracleAuditAnnotations:
    snap_bytes = serialize_snapshot(snapshot)
    snap_hash = hashlib.sha256(snap_bytes).hexdigest()

    per_cand: list[tuple[int, int, bool]] = []
    reachable: dict[int, bool] = {}
    for c in snapshot.candidates:
        ok = _can_reach_target_numbers(c.canonical_numbers)
        reachable[c.idx] = ok
        per_cand.append((snapshot.depth, c.idx, ok))

    sel_set = frozenset(snapshot.selected_indices)
    reachable_cnt = sum(1 for c in snapshot.candidates if reachable[c.idx])
    sel_reachable_cnt = sum(
        1 for c in snapshot.candidates if c.idx in sel_set and reachable[c.idx])

    best_r_idx, best_r_score = _best_from_candidates(
        snapshot.candidates, predicate=lambda c: reachable[c.idx])
    best_s_idx, best_s_score = _best_from_candidates(
        snapshot.candidates, predicate=lambda c: c.idx in sel_set)

    regret = (best_r_score - best_s_score) if (
        best_r_score is not None and best_s_score is not None) else None
    sel_cnt = len(sel_set)
    precision = sel_reachable_cnt / sel_cnt if sel_cnt > 0 else 1.0

    dm = DepthAuditMetrics(
        depth=snapshot.depth, proposed_count=len(snapshot.candidates),
        reachable_count=reachable_cnt,
        unreachable_count=len(snapshot.candidates) - reachable_cnt,
        selected_count=sel_cnt, selected_reachable_count=sel_reachable_cnt,
        best_reachable_idx=best_r_idx, best_reachable_score=best_r_score,
        best_selected_idx=best_s_idx, best_selected_score=best_s_score,
        selection_regret=regret, selected_precision=precision,
    )
    return OracleAuditAnnotations(
        task_id=snapshot.task_id, total_depths=1, depth_metrics=(dm,),
        per_candidate_reachability=tuple(per_cand),
        aggregate_selection_regret=regret, snapshot_hash=snap_hash,
    )


# ═══ Preflight / compatibility (Todo 7 will integrate) ═══


def is_oracle_assisted_policy(decision_policy: str) -> bool:
    return decision_policy in ORACLE_ASSISTED_DECISION_POLICIES


def preflight_audit_compatible(decision_policy: str) -> tuple[bool, str]:
    if not isinstance(decision_policy, str):
        return False, f"decision_policy must be str; got {type(decision_policy).__name__}"
    if decision_policy == "":
        return False, "decision_policy is empty"
    if is_oracle_assisted_policy(decision_policy):
        return False, (
            f"audit incompatible with oracle-assisted policy "
            f"{decision_policy!r} — oracle already used for selections")
    return True, ""
