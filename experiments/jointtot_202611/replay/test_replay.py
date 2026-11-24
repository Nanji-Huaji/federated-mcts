"""Small offline checks; no B0 data or outcome files needed."""
import unittest
from fractions import Fraction
from itertools import combinations

from freeze import METHODS, choose, group_top, state
from score import hit, legal_end, reachable, summarize


class ReplayTest(unittest.TestCase):
    def test_stable_and_only_parent_copy_filtered(self):
        x = "1 1 2 6"
        p = "1 + 1 = 2 (left: 2 2 6)\n"
        ys = [p, p+"2 - 6 = -4 (left: -4 2)\n", p+"2 + 2 = 4 (left: 4 6)\n"]
        out = choose(x, [p], ys, [2, 1, 0])
        self.assertEqual(out["stable"], [0, 1])
        self.assertEqual(out["progress"], [1, 2])  # not a legality filter
        self.assertEqual(out["zero_uniform"], {"fixed": [0, 1], "draw": 0, "from": []})
        self.assertEqual(choose(x, [p], [p], [3])["progress"], [])  # no backfill

    def test_state_and_votes(self):
        self.assertEqual(state("", "op (left: 4 2 2)\n"), (2, 2, 4))
        self.assertNotEqual(state("", "op (left: 2 4)"), (2, 2, 4))
        states = [(2,), (2,), (4,), (8,)]
        # An earlier duplicate has zero votes: preserve its later two-vote sibling.
        self.assertEqual(group_top(range(4), states, [0, 2, 1, 0], max), [1, 2])
        self.assertEqual(group_top(range(4), states, [2, 1, 0, 0], max), [0, 2])
        keys = [(4,), (2,), (2,)]
        self.assertEqual(group_top(range(3), keys, [1, 1, 1], max), [0, 1])
        self.assertEqual(group_top(range(3), keys, [1, 1, 1], sum), [1, 0])

    def test_zero_lottery(self):
        ys = [f"op (left: {i} 1)\n" for i in range(4)]
        law = choose("", [], ys, [0, 3, 0, 0])["zero_uniform"]
        self.assertEqual(law, {"fixed": [1], "from": [0, 2, 3], "draw": 1})
        self.assertEqual(hit({1}, law), 1)
        self.assertEqual(hit({2}, law), Fraction(1, 3))
        self.assertEqual(choose("", [], ys, [1, 1, 1, 0])["zero_uniform"]["fixed"], [0, 1])
        law = choose("", [], ys, [0]*4)["zero_uniform"]
        all_pairs = list(combinations(range(4), 2))
        for live in (set(), {0}, {0, 1}, set(range(4))):
            self.assertEqual(hit(live, law), Fraction(sum(bool(live.intersection(p)) for p in all_pairs), 6))

    def test_full_prefix_legality(self):
        self.assertEqual(legal_end("1 1 1 8", "1 + 1 = 2 (left: 8 1 2)\n"), (1, 2, 8))
        for line in ("1 + 1 = 2 (left: 1 1 2)", "1 + 1 = 1 (left: 1 1 8)",
                     "8 + 8 = 16 (left: 1 1 16)", "1 / 8 = 0 (left: 0 1 1)"):
            self.assertIsNone(legal_end("1 1 1 8", line))
        self.assertIsNone(legal_end("0 24", "24 / 0 = 0 (left: 0)"))
        # Final left alone looks live (4+20), but this prefix passed through -1.
        bad = "1 - 2 = -1 (left: -1 3 20)\n3 - -1 = 4 (left: 4 20)\n"
        self.assertIsNone(legal_end("1 2 3 20", bad))

    def test_slot_budget_and_integer_variant(self):
        parent = (1, 2, 8)  # (1+2)*8
        self.assertTrue(reachable(parent, 2))
        self.assertFalse(reachable(parent, 1))
        self.assertTrue(reachable((0, 24), 1))
        self.assertFalse(reachable((5, 24), 1))  # cannot just stop when 24 is present
        self.assertFalse(reachable((3, 3, 8, 8), 3))  # fraction-only Game24
        self.assertFalse(reachable((-1, 25), 1))
        self.assertTrue(reachable((24,), 0))

    def test_paired_common_denominator(self):
        rows = []
        for i, (pool_live, base, alt) in enumerate(((0, 0, 0), (1, 1, 0), (1, 0, 1))):
            r = dict(idx=i, pool_live=pool_live, pool_n=2, invalid_prefix_n=0,
                     nonadvance_n=0, stable_nonadvance_n=0)
            r.update({m+"_hit": base if m == "stable" else alt for m in METHODS})
            rows.append(r)
        s = summarize(rows, 1)
        self.assertEqual((s["questions"], s["live_pool_questions"]), (3, 2))
        v = s["methods"]["progress"]
        self.assertEqual((v["hits"], v["gained"], v["lost"], v["delta_pp"]), (1, 1, 1, 0))


if __name__ == "__main__":
    unittest.main()
