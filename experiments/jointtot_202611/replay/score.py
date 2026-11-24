"""Stage 2: evaluate the already pinned choices. Standard library only."""
import csv
import json
import math
import random
import re
from fractions import Fraction
from functools import lru_cache
from itertools import combinations

from freeze import (FROZEN, HERE, MANIFEST, MANIFEST_SHA, METHODS, RAW, RAW_SHA,
                    ROOT, compact, load_b0, pinned, sha)

FROZEN_SHA = "574fb1c6d03e39dc9cc8731308ebb9933852b44116376d9d5b9ba52e468f7772"
B, SEED = 10000, 20261124
LINE = re.compile(r"(-?\d+) ([+*/-]) (-?\d+) = (-?\d+) \(left: (-?\d+(?: -?\d+)*)\)")


def legal_end(x, y):
    """Replay the whole logged prefix, without repairing operands, arithmetic or left."""
    nums = sorted(map(int, x.split()))
    if not nums or min(nums) < 0:
        return None
    for line in y.splitlines():
        m = LINE.fullmatch(line)
        if not m:
            return None
        a, b, c = map(int, (m[1], m[3], m[4]))
        if min(a, b, c) < 0 or (m[2] == "/" and (b == 0 or a % b != 0)):
            return None
        actual = {"+": a+b, "-": a-b, "*": a*b, "/": a//b if b else None}[m[2]]
        if c != actual:
            return None
        try:
            nums.remove(a)
            nums.remove(b)  # two occurrences needed when a == b
        except ValueError:
            return None
        nums = sorted(nums + [c])
        if nums != sorted(map(int, m[5].split())):
            return None
    return tuple(nums)


@lru_cache(None)
def reachable(nums, steps):
    if not nums or min(nums) < 0 or len(nums)-1 > steps:
        return False
    if len(nums) == 1:
        return nums == (24,)
    for i, j in combinations(range(len(nums)), 2):
        a, b = nums[i], nums[j]
        rest = [v for k, v in enumerate(nums) if k not in (i, j)]
        outs = {a+b, a*b, abs(a-b)}
        if b and a % b == 0:
            outs.add(a//b)
        if a and b % a == 0:
            outs.add(b//a)
        if any(reachable(tuple(sorted(rest+[c])), steps-1) for c in outs):
            return True
    return False


def hit(live, pick):
    """Any retained live child. For the sealed zero lottery integrate all outcomes."""
    if isinstance(pick, list):
        return int(any(i in live for i in pick))
    if any(i in live for i in pick["fixed"]):
        return 1
    n, k = len(pick["from"]), pick["draw"]
    dead = sum(i not in live for i in pick["from"])
    return 1 - Fraction(math.comb(dead, k), math.comb(n, k))


def frontier(idx, d, step, picks):
    ys, parents = step["new_ys"], step["ys"]
    assert len(set(ys)) == len(ys)  # early depths: text identifies the exact index
    assert [ys[i] for i in picks["stable"]] == step["select_new_ys"]
    ends = [legal_end(step["x"], y) for y in ys]
    left = 2 - d  # slots AFTER choosing this child; a carried parent still spent a slot
    live = {i for i, nums in enumerate(ends) if reachable(nums, left)}
    row = dict(idx=idx, depth=d, steps_left=left, pool_n=len(ys),
               nonadvance_n=sum(y in parents for y in ys),
               stable_nonadvance_n=sum(ys[i] in parents for i in picks["stable"]),
               invalid_prefix_n=ends.count(None), pool_live_n=len(live),
               pool_live=int(bool(live)), live_indices=compact(sorted(live)),
               zero_contest=int(bool(picks["zero_uniform"]["draw"])),
               source_sha256=RAW_SHA, manifest_sha256=MANIFEST_SHA, frozen_sha256=FROZEN_SHA)
    for m in METHODS:
        row[m + ("_law" if m == "zero_uniform" else "_indices")] = compact(picks[m])
        row[m+"_hit"] = hit(live, picks[m])
    return row


def paired_ci(diffs, seed):
    """Paired percentile bootstrap: at a single depth each entry is one eligible task."""
    if not diffs:
        return None
    n = len(diffs)
    if len(set(diffs)) == 1:
        return [round(100*float(diffs[0]), 6)]*2
    rng = random.Random(seed)
    ds = list(map(float, diffs))
    rates = sorted(100 * sum(rng.choices(ds, k=n))/n for _ in range(B))
    return [round(rates[math.ceil(p*B)-1], 6) for p in (.025, .975)]


def number(x):
    return int(x) if x == int(x) else round(float(x), 6)


def summarize(rows, seed):
    assert len(rows) == len({r["idx"] for r in rows})
    eligible = [r for r in rows if r["pool_live"]]
    n = len(eligible)
    base = [r["stable_hit"] for r in eligible]
    metrics = {}
    for m in METHODS:
        ps = [r[m+"_hit"] for r in eligible]
        diffs = [p-b for p, b in zip(ps, base)]
        gained = sum(p for p, b in zip(ps, base) if not b)
        lost = sum(1-p for p, b in zip(ps, base) if b)
        assert sum(ps)-sum(base) == gained-lost
        v = dict(hits=number(sum(ps)), retention_pct=number(100*Fraction(sum(ps), n)) if n else None,
                 gained=number(gained), lost=number(lost),
                 delta_pp=number(100*Fraction(gained-lost, n)) if n else None,
                 paired_ci95_pp=paired_ci(diffs, seed))
        if m == "zero_uniform":
            v["exact_expected_counts"] = {k: str(x) for k, x in
                                          (("hits", sum(ps)), ("gained", gained), ("lost", lost))}
            v["fixed_sample_lottery_sd_hits"] = round(math.sqrt(sum(p*(1-p) for p in ps)), 6)
        metrics[m] = v
    return dict(questions=len(rows), live_pool_questions=n, dead_pool_questions=len(rows)-n,
                candidates=sum(r["pool_n"] for r in rows),
                invalid_prefix_candidates=sum(r["invalid_prefix_n"] for r in rows),
                nonadvancing_candidates=sum(r["nonadvance_n"] for r in rows),
                stable_nonadvancing_slots=sum(r["stable_nonadvance_n"] for r in rows), methods=metrics)


def audit_b0(rows):
    # No alternative selections at depth 2. Check the actual run, not a simulated rollout.
    matched = [0, 0, 0]
    solved = outputs = checked = mismatches = 0
    for r in rows:
        for d, s in enumerate(r["steps"]):
            ids = sorted(range(len(s["values"])), key=s["values"].__getitem__, reverse=True)[:2]
            matched[d] += [s["new_ys"][i] for i in ids] == s["select_new_ys"]
        ys, infos = r["ys"], r["infos"]
        assert len(infos) == len(ys) == 2 and ys == r["steps"][2]["select_new_ys"]
        rewards = [v["r"] for v in infos]
        assert all(v in (0, 1) for v in rewards)
        solved += any(rewards)
        outputs += sum(rewards)
        checked += len(ys)
        mismatches += sum((legal_end(r["steps"][0]["x"], y) == (24,)) != v for y, v in zip(ys, rewards))
    assert matched == [len(rows)]*3
    return dict(stable_order_matches_by_depth=matched, recorded_questions=len(rows),
                recorded_solved_questions=solved, recorded_outputs=checked, recorded_correct_outputs=outputs,
                strict_terminal_vs_recorded_output_disagreements=mismatches)


def main():
    header, *sealed = map(json.loads, pinned(FROZEN, FROZEN_SHA).splitlines())
    assert header == {"schema": 1, "source_sha256": RAW_SHA, "manifest_sha256": MANIFEST_SHA}
    src = load_b0()
    assert [(f["idx"], f["depth"]) for f in sealed] == [(r["idx"], d) for r in src for d in (0, 1)]
    records = [frontier(r["idx"], d, r["steps"][d], sealed[2*i+d]["picks"])
               for i, r in enumerate(src) for d in (0, 1)]
    by_depth = {}
    for d in (0, 1):
        rows = [r for r in records if r["depth"] == d]
        by_depth[d] = {name: summarize(rs, SEED+d) for name, rs in (
            ("all", rows), ("zero_contest", [r for r in rows if r["zero_contest"]]),
            ("no_zero_contest", [r for r in rows if not r["zero_contest"]]))}
    audit = audit_b0(src)
    csv_path = HERE / "per_frontier.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(records)
    summary = dict(source=str(RAW.relative_to(ROOT)), source_sha256=RAW_SHA,
                   manifest=str(MANIFEST.relative_to(ROOT)), manifest_sha256=MANIFEST_SHA,
                   frozen_sha256=FROZEN_SHA, per_frontier_sha256=sha(csv_path.read_bytes()),
                   depths=[0, 1], frontiers=len(records), width=2,
                   estimand="At fixed B0 parents: P(retain >=1 strict live child | original pool has >=1).",
                   uncertainty={"resamples": B, "unit": "paired live-pool validation task within stratum",
                                "base_seed": SEED, "seed_rule": "base_seed + depth",
                                "statistic": "percentile bootstrap of difference in percentage points vs stable; "
                                             "zero_uniform integrates the lottery before resampling tasks"},
                   baseline_audit=audit, by_depth=by_depth)
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    print(compact(audit))
    for d in by_depth:
        s = by_depth[d]["all"]
        print(f"depth {d}: {s['live_pool_questions']}/{s['questions']} pools live; hits (gained/lost)")
        for m, v in s["methods"].items():
            print(f"  {m:18} {v['hits']} (+{v['gained']}/-{v['lost']}), "
                  f"delta {v['delta_pp']} pp, 95% paired {v['paired_ci95_pp']}")


if __name__ == "__main__":
    main()
