"""Focused arithmetic/validation tests and an end-to-end regression on fixed archives."""

import itertools
import json
import unittest
from collections import Counter
from decimal import Decimal

from archive import ARMS, N, Sources, load_sources, usage, validate_records
from paired_stats import paired, empirical_sum_weights
from rescore import calculate


class PairTests(unittest.TestCase):
    def test_enumeration_vs_resample_each_item(self):
        # Enumerate ordered question-level resamples independently, including repeated zeros.
        diffs = [-1, 0, 0, 1]
        brute = Counter(map(sum, itertools.product(diffs, repeat=len(diffs))))
        self.assertEqual(brute, empirical_sum_weights(1, 1, 2))

    def test_direction_and_exact_p(self):
        a, b = [1, 1, 1, 1, 0], [0, 0, 0, 1, 0]
        p, rev = paired(a, b), paired(b, a)
        self.assertEqual((p['tot_only'], p['speculative_solve_only'], p['spec_minus_tot_pp']), (3, 0, -60))
        self.assertEqual(p['exact_two_sided_mcnemar_p'], .25)
        self.assertEqual(p['paired_percentile_95_pp'], [-x for x in reversed(rev['paired_percentile_95_pp'])])
        self.assertEqual(paired([1], [1])['exact_two_sided_mcnemar_p'], 1)
        for pair in [([], []), ([0], [0, 0]), ([0, 2], [0, 0])]:
            with self.subTest(pair=pair), self.assertRaises(ValueError):
                paired(*pair)


class IntegrityTests(unittest.TestCase):
    def test_identity_fails_closed(self):
        data, _, arms, _, _ = load_sources(Sources())
        rows = arms['tot']
        with self.assertRaisesRegex(ValueError, 'index set/order'):
            validate_records(rows[:1] + rows[:-1], data)
        rows[0]['steps'][2]['x'] += ' '
        with self.assertRaisesRegex(ValueError, 'question mismatch'):
            validate_records(rows, data)

    def test_cumulative_not_per_item(self):
        def point(i, n):
            raw = {'prompt_tokens': n, 'completion_tokens': 2*n, 'total_tokens': 3*n,
                   'total_calls': n, 'cost': Decimal(5*n)/1_000_000}
            return {'idx': i, 'usage_so_far': {'M': raw}, 'time_consumption': {'generation': 1, 'evaluation': 1}}
        x, y = point(0, 2), point(1, 5)
        per_item, total = usage([x, y], y['usage_so_far'])
        self.assertEqual([r['all_models_delta']['total_tokens'] for r in per_item], [6, 9])
        self.assertEqual(total['all_models']['total_tokens'], 15)
        with self.assertRaisesRegex(ValueError, 'nonmonotone'):
            usage([y, x], x['usage_so_far'])
        with self.assertRaisesRegex(ValueError, 'sidecar'):
            usage([x, y], x['usage_so_far'])

    def test_archive_regression_including_all_failures(self):
        files, summary = calculate()
        items = [json.loads(l) for l in files['outcomes.jsonl'].splitlines()]
        slots = [json.loads(l) for l in files['candidates.jsonl'].splitlines()]
        review = {json.loads(l)['id'] for l in files['review.jsonl'].splitlines()}
        self.assertEqual([r['idx'] for r in items], list(range(N)))
        self.assertEqual(len(slots), 277)
        for c in slots:
            if c['frozen_parse']['value'] is None or len(set(c['r'].values())) > 1:
                self.assertIn(c['id'], review)
            self.assertEqual(c['r']['archived'], c['legacy_replay_r'])
        for panel, counts in [('archived', (40, 42)), ('frozen', (42, 40)), ('current_checkout', (36, 33))]:
            self.assertEqual(tuple(sum(i['methods'][a]['any'][panel] for i in items) for a in ARMS), counts)
        self.assertEqual(summary['paired_any']['frozen']['paired_percentile_95_pp'], [-12., 4.])


if __name__ == '__main__':
    unittest.main()
