"""Offline replay of the pinned historical greedy Game24 tree archive.

The public functions return only numbers, labels, digests and provenance. Nothing
imports the original experiment, runs a model, executes an expression, or writes
raw prompts, candidates or logs. Invalid trajectories remain observations.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from collections import Counter
from collections.abc import Sequence
from decimal import Decimal
from itertools import permutations
from pathlib import Path


# After JSON decoding, the *entire* payload is three lines in this grammar,
# with zero or one terminal LF. ASCII digits, binary operators and literal
# delimiters only: no expression/prose salvage or whitespace normalization.
_LINE = re.compile(r"([0-9]+) ([-+*/]) ([0-9]+) = ([0-9]+) \(left: ([0-9]+(?: [0-9]+)*)\)")


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _json_sha(value) -> str:
    """Canonical compact JSON for derived keys (not a hash of source file bytes)."""
    return _sha(json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8"))


def _require(condition, problem: str) -> None:
    # Pass only schema labels / indices; never echo material being audited.
    if not condition:
        raise ValueError(problem)


def _expression_set(numbers, ops):
    """Resolve only the reported ops, keeping every consistent operand attribution.

    Intermediate and leaf occurrences can have the same numeric value. Picking
    one arbitrarily could mark the same evidence as two different solutions.
    Fully parenthesized expressions canonicalize + and * by operand sorting,
    not associativity or general algebraic equivalence. No expression leaves
    this function: return just a digest of the sorted set and its cardinality.
    """
    states = {tuple(sorted((n, str(n)) for n in numbers))}
    for a, op, b, value in ops:
        following = set()
        for state in states:
            for i, j in permutations(range(len(state)), 2):
                (va, ea), (vb, eb) = state[i], state[j]
                if (va, vb) != (a, b):
                    continue
                if op in ("+", "*") and ea > eb:
                    ea, eb = eb, ea
                rest = [p for k, p in enumerate(state) if k not in (i, j)]
                rest.append((value, f"({ea}{op}{eb})"))
                following.add(tuple(sorted(rest)))
        states = following
    expressions = sorted(state[0][1] for state in states)
    _require(expressions, "no operand provenance for validated trajectory")
    return _json_sha(expressions), len(expressions)


def replay_trajectory(numbers: Sequence[int], text: str) -> dict:
    """Independently judge one retained slot from its actual four inputs.

    raw_sha256 hashes the decoded, otherwise untouched string's UTF-8 bytes.
    For valid solutions, solution_sha256 identifies [sorted input multiset,
    the three ordered [a, op, b, result] steps]. Left-list formatting
    and ordering do not define a new solution. No algebraic-equivalence claim.
    fail_step is one-based, null when the whole-payload check fails or succeeds.
    """
    state = list(numbers)
    _require(len(state) == 4 and all(type(n) is int and n >= 0 for n in state),
             "expected four nonnegative integer inputs")
    _require(isinstance(text, str), "expected a decoded trajectory string")
    lines = text.removesuffix("\n").split("\n")
    # Preserve lone surrogate escapes in malformed JSON strings for digest-only diagnostics.
    raw_sha = _sha(text.encode("utf-8", errors="surrogatepass"))
    ops = []

    def verdict(reason, step=None):
        expr_sha, n_expr = _expression_set(numbers, ops) if reason == "valid" else (None, 0)
        return {"expression_set_sha256": expr_sha, "n_possible_expressions": n_expr,
                "valid": reason == "valid", "reason": reason, "fail_step": step,
                "line_count": len(lines), "raw_sha256": raw_sha,
                "solution_sha256": _json_sha([sorted(numbers), ops]) if reason == "valid" else None}

    if len(lines) != 3:
        return verdict("step_count")
    for depth, line in enumerate(lines, 1):
        m = _LINE.fullmatch(line)
        if not m:
            return verdict("format", depth)
        a, op, b, result, left = m.groups()
        try:
            a, b, result = map(int, (a, b, result))
            remaining = Counter(map(int, left.split(" ")))
        except ValueError:  # e.g. a decimal literal exceeds the runtime int limit
            return verdict("format", depth)
        for operand in (a, b):
            if operand not in state:
                return verdict("unavailable_operand", depth)
            state.remove(operand)  # consume twice even for equal-valued operands
        if op == "+":
            value = a + b
        elif op == "-":
            value = a - b
            if value < 0:
                return verdict("negative_result", depth)
        elif op == "*":
            value = a * b
        else:
            if b == 0:
                return verdict("division_by_zero", depth)
            if a % b:
                return verdict("inexact_division", depth)
            value = a // b
        if value != result:
            return verdict("wrong_result", depth)
        state.append(value)
        if Counter(state) != remaining:
            return verdict("wrong_left_multiset", depth)
        ops.append([a, op, b, value])
    return verdict("valid") if state == [24] else verdict("not_24", 3)


_GLOB = "*_10_start0_end148*check_True*warm_True*idx_0.json"
_STEM = ("logs/game24/tot/deepseek-v3/0.7_propose_n_generate_sample_1_value_n_evaluate_sample_1_"
         "method_select_greedy_n_select_sample_10_start0_end148_smg_False_sme_False_check_True_"
         "rule_False_warm_True_last_False_idx_0")
_PINS = {
    "archive": (_STEM + ".json", "c161b7c21e1774160011b9d61aa38c904e5f37cb51cddca211bfdd08920974a4"),
    "performance": (_STEM + "_performance.json", "167448a73cca65d8d2624c339b362d86a6791c1417b842a9c52aee5b72eef2a4"),
    "dataset": ("src/tot/data/24/24.csv", "b9f12b3e36d987a3c714c4cef17d89a137d7c59da26532fcfb93b4821d8111b5"),
    "prompt_context": ("src/tot/prompts/game24.py", "97333c4fa70c1af261adb998aba785e83c5085c50debe3c8336fc43d1adb44f7"),
    "task_context": ("src/tot/tasks/game24.py", "d1600b8d123dfdde6048d4d8c30fbdc71c26278319f45d113707b30791b5649a"),
    "solver_context": ("src/tot/methods/bfs.py", "0992ed817fa15f611378ab7e10bd85b012f04efe8c1576c1285bccf4df8569ed"),
    "format_context": ("src/tot/pattern_match.py", "f74f39c5d9d76b05e6afa8a2f62eca200e862f83f9595a7af2f6c65d78491bba"),
    "accounting_context": ("src/tot/models.py", "7829d6992c2ef4da90c6d212cd072565488d005966d84ea069f1d0c13afec49d"),
}


def _sources(root):
    paths = sorted(p.relative_to(root).as_posix() for p in (root / "logs").rglob(_GLOB) if p.is_file())
    _require(paths == [_PINS["archive"][0]], "missing or unexpected matching historical archive(s)")
    blobs, sources = {}, {}
    for key, (path, expected) in _PINS.items():
        blob = (root / path).read_bytes()
        sha = _sha(blob)
        _require(sha == expected, f"{key}: SHA-256 mismatch ({sha})")
        sources[key] = {"path": path, "sha256": sha, "bytes": len(blob)}
        if not key.endswith("_context"):
            blobs[key] = blob
    return blobs, sources


def _archive_rows(blob):
    """Parse real JSON elements and hash their exact UTF-8 source spans."""
    text = blob.decode("utf-8")
    whitespace = re.compile(r"[ \n\t\r]*")
    decoder = json.JSONDecoder(parse_float=Decimal)
    pos = whitespace.match(text).end()
    _require(text[pos:pos+1] == "[", "archive: expected array")
    pos = whitespace.match(text, pos + 1).end()
    while text[pos:pos+1] != "]":
        record, end = decoder.raw_decode(text, pos)
        yield record, _sha(text[pos:end].encode("utf-8"))
        pos = whitespace.match(text, end).end()
        if text[pos:pos+1] == "]":
            break
        _require(text[pos:pos+1] == ",", "archive: expected separator")
        pos = whitespace.match(text, pos + 1).end()
        _require(text[pos:pos+1] != "]", "archive: trailing comma")
    _require(whitespace.match(text, pos + 1).end() == len(text), "archive: trailing material")


def _run(record, raw_sha, idx, dataset):
    tag = f"archive idx {idx}: "
    _require(set(record) == {"idx", "steps", "ys", "infos", "usage_so_far", "time_consumption"},
             tag + "record fields")
    _require(type(record["idx"]) is int and record["idx"] == idx and 0 <= idx < 148, tag + "index")
    # idx is the ordinal *within this historical subset*. It is NOT the CSV row.
    # Exact string identity at every depth is checked below, not just filename/idx.
    rank = idx + (100 if idx < 100 else 400)
    row = dataset[rank-1]
    puzzle = record["steps"][0]["x"]
    _require(row["Rank"] == str(rank) and row["Puzzles"] == puzzle, tag + "dataset identity")
    _require(re.fullmatch(r"[0-9]+(?: [0-9]+){3}", puzzle), tag + "four integer inputs")
    numbers = list(map(int, puzzle.split()))
    parents, steps = [""], record["steps"]
    _require(len(steps) == 3, tag + "depth count")
    for depth, s in enumerate(steps):
        _require(set(s) == {"step", "x", "ys", "new_ys", "values", "select_new_ys"}, tag + "step fields")
        _require(type(s["step"]) is int and s["step"] == depth and s["x"] == puzzle, tag + "depth or input")
        _require(s["ys"] == parents, tag + "parent frontier")
        pool, values = s["new_ys"], s["values"]
        _require(len(pool) == len(values) > 0 and all(isinstance(y, str) for y in pool), tag + "proposal slots")
        _require(all(type(v) in (int, Decimal) and Decimal(v).is_finite() for v in values), tag + "scores")
        ranks = sorted(range(len(pool)), key=values.__getitem__, reverse=True)[:10]
        parents = [pool[k] for k in ranks]  # stable: equal values keep the logged proposal order
        _require(s["select_new_ys"] == parents, tag + "greedy selection")
    ys, old = record["ys"], record["infos"]
    _require(ys == parents and len(old) == len(ys), tag + "terminal alignment")
    candidates = []
    for rank0, (y, legacy) in enumerate(zip(ys, old)):
        _require(set(legacy) == {"r"} and type(legacy["r"]) is int and legacy["r"] in (0, 1),
                 tag + "legacy labels")
        candidates.append({"slot": rank0 + 1, "pool_slot": ranks[rank0] + 1,
                           **replay_trajectory(numbers, y), "legacy_r": legacy["r"]})
    return {"idx": idx, "input": numbers, "dataset_rank": rank, "run_sha256": raw_sha,
            "candidates": candidates}


def _profile_frontiers(record, run, frequencies, collisions):
    """Only counts, slot references and hashes escape the raw frontiers."""
    trace, by_depth = [], []
    for s in record["steps"]:
        depth = s["step"]
        parents, pool, selected = s["ys"], s["new_ys"], s["select_new_ys"]
        pool_freq = Counter(pool)
        frequencies["proposal_pools"].update(pool_freq)
        frequencies["all_selected_frontiers"].update(selected)
        if depth == 2:
            frequencies["final_pool"].update(pool_freq)
        for y, multiplicity in pool_freq.items():
            if multiplicity > 1:
                collisions.append({"idx": run["idx"], "depth": depth, "raw_sha256": _sha(y.encode()),
                                   "pool_slots": [i for i, v in enumerate(pool, 1) if v == y],
                                   "selected_slots": [i for i, v in enumerate(selected, 1) if v == y]})
        parent_set = set(parents)
        stems = {p.removesuffix("\n") for p in parents}
        carried = sum(y in parent_set for y in pool)
        extended = sum(y not in parent_set and y.removesuffix("\n").rpartition("\n")[0] in stems
                       for y in pool)
        by_depth.append({"depth": depth, "parent_slots": len(parents), "proposal_slots": len(pool),
                         "scored_slots": len(s["values"]), "selected_slots": len(selected),
                         "carry_slots": carried, "extended_slots": extended,
                         "unlinked_slots": len(pool) - carried - extended})
        # Ordered structural fingerprint; no idx, token counts, timers, or legacy
        # labels sneak in and make repeated search traces appear trivially unique.
        trace.append([depth, s["x"], parents, pool, [str(v) for v in s["values"]], selected])
    run.update(depths=by_depth, trace_sha256=_json_sha(trace),
               selected_vector_sha256=_json_sha(record["ys"]))


def _histogram(values):
    return {str(k): n for k, n in sorted(Counter(values).items())}


def _duplication(runs, frequencies, collisions):
    slots = [c for r in runs for c in r["candidates"]]
    valid = [c for c in slots if c["valid"]]
    frequencies.update({
        "input_indices": Counter(r["idx"] for r in runs),
        "input_multisets": Counter(tuple(sorted(r["input"])) for r in runs),
        "records_verbatim": Counter(r["run_sha256"] for r in runs),
        "ordered_search_traces": Counter(r["trace_sha256"] for r in runs),
        "ordered_selected_vectors": Counter(r["selected_vector_sha256"] for r in runs),
        "retained_trajectories": Counter(c["raw_sha256"] for c in slots),
        "valid_ordered_solutions": Counter(c["solution_sha256"] for c in valid),
        "valid_expression_sets": Counter(c["expression_set_sha256"] for c in valid),
    })
    result = {}
    for name, counts in frequencies.items():
        total = sum(counts.values())
        result[name] = {"occurrences": total, "distinct": len(counts), "duplicate_excess": total - len(counts),
                        "multiplicity_histogram": _histogram(counts.values())}
    result["within_pool_collisions"] = collisions
    result["ambiguous_expression_slots"] = sum(c["n_possible_expressions"] > 1 for c in valid)
    return result


_MODEL = "deepseek-v3"
_COUNTERS = ("total_calls", "prompt_tokens", "completion_tokens", "total_tokens")


def _budget_delta(record, previous):
    """Counts are cumulative per model, not additive per selected trajectory."""
    tag = f"archive idx {record['idx']}: "
    usage = record["usage_so_far"]
    _require(set(usage) == {_MODEL}, tag + "usage model")
    cur = usage[_MODEL]
    _require(set(cur) == {*_COUNTERS, "cost"}, tag + "usage fields")
    _require(all(type(cur[k]) is int and cur[k] >= 0 for k in _COUNTERS), tag + "usage counters")
    _require(cur["prompt_tokens"] + cur["completion_tokens"] == cur["total_tokens"], tag + "token sum")
    _require(type(cur["cost"]) in (int, Decimal) and Decimal(cur["cost"]).is_finite(), tag + "cost")
    estimated = Decimal(cur["prompt_tokens"] + 2 * cur["completion_tokens"]) / 10**6
    _require(abs(estimated - cur["cost"]) < Decimal("1e-12"), tag + "historical tariff")
    delta = {k: v - previous.get(k, 0) for k, v in cur.items()}
    _require(all(v >= 0 for v in delta.values()), tag + "nonmonotonic cumulative usage")
    delta["cost"] = format(delta["cost"], "f")  # no repricing or float rounding
    times = record["time_consumption"]
    _require(set(times) == {"generation", "evaluation"} and all(t == 0 for t in times.values()),
             tag + "expected uninformative zero time fields")
    return {**delta, "cost_unit": "CNY", "model": _MODEL}, cur


def _accounting_summary(performance, runs, final):
    _require(set(performance) == {"acc", "avg_sum", "sm", "llm", _MODEL}, "performance: fields")
    _require(performance["llm"] == _MODEL and performance["sm"] == "qwen2.5-32b-instruct",
             "performance: model labels")
    _require(performance[_MODEL] == final, "performance: final cumulative snapshot")
    sums = {k: sum(r["budget"][k] for r in runs) for k in _COUNTERS}
    _require(all(sums[k] == final[k] for k in sums), "performance: counter deltas")
    _require(sum(Decimal(r["budget"]["cost"]) for r in runs) == final["cost"], "performance: cost deltas")
    legacy = [sum(c["legacy_r"] for c in r["candidates"]) for r in runs]
    for key, n in (("acc", sum(v > 0 for v in legacy)), ("avg_sum", sum(legacy))):
        _require(abs(Decimal(n) / len(runs) - performance[key]) < Decimal("1e-12"), "performance: " + key)
    return {**sums, "model": _MODEL, "cost": format(final["cost"], "f"), "cost_unit": "CNY",
            "currency_field_in_json": None, "unit_basis": "historical token tariff; yuan in accounting_context",
            "edge_model_named_in_performance": performance["sm"], "edge_usage": None,
            "prefix_usage": None, "per_call_usage": None, "role_split_usage": None,
            "billed_cost": None, "wall_seconds": None, "logged_time_fields_all_zero": True}


def load_tree_archive(repo_root: Path = Path(__file__).resolve().parents[3]) -> dict:
    """Return sources, summary and all retained slots in original per-puzzle order.

    Paths and content hashes pin the real artifacts and code context. Structural
    mismatches raise ValueError rather than silently substituting another run.
    A candidate parse failure, in contrast, stays in its original slot.
    """
    blobs, sources = _sources(Path(repo_root))
    data = list(csv.DictReader(io.StringIO(blobs["dataset"].decode("utf-8"))))
    runs, previous = [], {}
    frequencies = {key: Counter() for key in ("proposal_pools", "final_pool", "all_selected_frontiers")}
    collisions = []
    for idx, (r, raw_sha) in enumerate(_archive_rows(blobs["archive"])):
        run = _run(r, raw_sha, idx, data)
        _profile_frontiers(r, run, frequencies, collisions)
        run["budget"], previous = _budget_delta(r, previous)
        runs.append(run)
    _require(len(runs) == 148, "archive: incomplete index coverage")
    cs = [r["candidates"] for r in runs]
    performance = json.loads(blobs["performance"], parse_float=Decimal)
    search_by_depth = [{"depth": i, **{field: sum(r["depths"][i][field] for r in runs)
                                     for field in runs[0]["depths"][i] if field != "depth"}} for i in range(3)]
    summary = {
        "archive_glob_in_logs": _GLOB, "n_archive_files": 1,
        "input_order_sha256": _json_sha([sorted(r["input"]) for r in runs]),
        "slot_width_histogram": _histogram(len(row) for row in cs),
        "selected_line_count_histogram": _histogram(c["line_count"] for row in cs for c in row),
        "legacy_label_disagreements": sum(c["valid"] != bool(c["legacy_r"]) for row in cs for c in row),
        "duplication": _duplication(runs, frequencies, collisions), "search_by_depth": search_by_depth,
        "budget": _accounting_summary(performance, runs, previous),
        "n_runs": len(runs), "n_retained_slots": sum(map(len, cs)),
        "selected_slot_verdicts": dict(sorted(Counter(c["reason"] for row in cs for c in row).items())),
        "solved_at": {k: sum(any(c["valid"] for c in row[:limit]) for row in cs)
                      for k, limit in (("1", 1), ("3", 3), ("full", None))},
    }
    return {"runs": runs, "summary": summary, "sources": sources}
