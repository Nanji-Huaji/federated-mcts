"""Stage 1: seal candidate indices (or a sampling law). No outcome/solver imports."""
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
STEM = ("0.9_propose_n_generate_sample_2_vote_n_evaluate_sample_3_method_select_greedy_"
        "n_select_sample_2_start900_end1000_smg_False_sme_False_check_True_rule_False_"
        "warm_False_last_False_idx_0_ntasks200")
RAW = ROOT / "logs/game24/tot/qwen3.7-plus" / (STEM + ".json")
MANIFEST = RAW.with_name(STEM + "_manifest.json")
RAW_SHA = "24b3754c03cd7978ebb008550d99a6681c55f6baeb86f36e7a18401771dac765"
MANIFEST_SHA = "222cfbb1d038a2b739985afb22dc794503e4d027291c368527d8796ff6aff7f5"
FROZEN = HERE / "frozen.jsonl"
METHODS = ("stable", "progress", "state_max", "state_sum", "progress_state_max", "zero_uniform")
WIDTH = 2
LEFT = re.compile(r"\(left: (-?\d+(?: -?\d+)*)\)\s*\Z")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def pinned(path, digest):
    data = path.read_bytes()
    if sha(data) != digest:
        raise ValueError(f"SHA-256 mismatch: {path}")
    return data


def load_b0():
    rows = json.loads(pinned(RAW, RAW_SHA))
    manifest = json.loads(pinned(MANIFEST, MANIFEST_SHA))
    ids = [r["idx"] for r in rows]
    if (manifest["task_split_name"] != "val" or len(ids) != len(set(ids)) or
            len(ids) != 200 or ids != manifest["task_ids"]):
        raise ValueError("Not the ordered 200-question B0 validation run")
    return rows


def state(x, y):
    """Observable, signed, sorted multiset; not a validity or feasibility test."""
    if y:
        match = LEFT.search(y)
        if not match:
            raise ValueError(f"Missing logged remaining numbers: {y!r}")
        x = match[1]
    return tuple(sorted(map(int, x.split())))


def group_top(ids, states, votes, reduce_votes):
    groups = defaultdict(list)
    for i in ids:
        groups[states[i]].append(i)
    ranking = []
    for group in groups.values():
        # Never let an earlier low-vote duplicate replace the best-voted member.
        rep = min(group, key=lambda i: (-votes[i], i))
        score = reduce_votes(votes[i] for i in group)
        ranking.append((-score, rep))
    return [rep for _, rep in sorted(ranking)[:WIDTH]]


def choose(x, parents, candidates, votes):
    if len(candidates) != len(votes) or any(type(v) is not int or v < 0 for v in votes):
        raise ValueError("Expected one nonnegative vote count per candidate")
    ids = range(len(votes))
    states = [state(x, y) for y in candidates]
    ranked = sorted(ids, key=lambda i: (-votes[i], i))
    stable = ranked[:WIDTH]
    eligible = [i for i in ranked if candidates[i] not in parents or len(states[i]) == 1]
    pos = [i for i in stable if votes[i] > 0]
    zeros = [i for i in ranked if not votes[i]]
    draw = len(stable) - len(pos)
    # Only randomize zeros competing for a seat, never a positive tie.
    lottery = {"fixed": stable, "draw": 0, "from": []}
    if 0 < draw < len(zeros):
        lottery = {"fixed": pos, "draw": draw, "from": zeros}
    return {"stable": stable, "progress": eligible[:WIDTH],
            "state_max": group_top(ids, states, votes, max),
            "state_sum": group_top(ids, states, votes, sum),
            "progress_state_max": group_top(eligible, states, votes, max),
            "zero_uniform": lottery}


def compact(obj):
    return json.dumps(obj, separators=(",", ":"), sort_keys=True)


def main():
    header = {"schema": 1, "source_sha256": RAW_SHA, "manifest_sha256": MANIFEST_SHA}
    lines = [compact(header)]
    for r in load_b0():
        steps = r["steps"]
        assert len(steps) == 3 and steps[0]["ys"] == [""]
        for d in (0, 1):
            s = steps[d]  # Always B0's own input parents, not another arm's picks.
            x, parents, ys, vs = (s[k] for k in ("x", "ys", "new_ys", "values"))
            assert s["step"] == d and x == steps[0]["x"] and sum(vs) == 3
            if d:
                assert parents == steps[d-1]["select_new_ys"]
            # Here every candidate is a verbatim parent or one appended reduction.
            for y in ys:
                p, = (p for p in parents if y.startswith(p))
                if y != p:
                    assert len(y[len(p):].splitlines()) == 1
                    assert len(state(x, y)) == len(state(x, p)) - 1
            picks = choose(x, parents, ys, vs)
            assert [ys[i] for i in picks["stable"]] == s["select_new_ys"]
            lines.append(compact({"idx": r["idx"], "depth": d, "picks": picks}))
    data = ("\n".join(lines) + "\n").encode()
    FROZEN.write_bytes(data)
    print(f"Sealed {len(lines)-1} B0 frontiers; stable order matches every log selection.")
    print(f"{sha(data)}  {FROZEN.name}")


if __name__ == "__main__":
    main()
