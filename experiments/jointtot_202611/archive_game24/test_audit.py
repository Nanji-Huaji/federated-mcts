"""Offline tests; no project imports, clients, or fixture files written."""
import csv
import io
import itertools
import json
import runpy
import unittest
from pathlib import Path

A = runpy.run_path(str(Path(__file__).with_name('audit.py')))
check = A['verify']
X = '4 5 6 10'
Y = ('10 - 4 = 6 (left: 5 6 6)\n'
     '5 * 6 = 30 (left: 6 30)\n'
     '30 - 6 = 24 (left: 24)\n')


class AuditTests(unittest.TestCase):
    def test_actual_encoding_and_rank(self):
        blobs, _ = A['inputs']()
        row = json.loads(blobs['edge_gen_raw'])[0]
        self.assertEqual((row['idx'], row['steps'][0]['x']), (900, X))
        self.assertEqual(row['ys'], row['steps'][-1]['select_new_ys'])
        self.assertEqual(row['ys'][2], Y)  # real rank 3: computed 6 and original 6 are distinct copies
        self.assertNotEqual(check(X, row['ys'][0]), 'valid')
        self.assertEqual(check(X, Y), 'valid')
        self.assertEqual(check(X, Y.replace('5 6 6', '6 5 6')), 'valid')  # multiset, not display order
        self.assertEqual(check(X, Y.removesuffix('\n')), 'valid')

    def test_no_format_salvage_or_early_24(self):
        for bad in [Y.replace('\n', r'\n'), Y+'\n', 'Answer: '+Y, Y+'24', Y.replace(' * ', ' × '),
                    Y.replace(')\n', ') trailing text\n', 1), Y.replace('5 6 6', '5, 6, 6'),
                    Y.split('\n', 1)[1], '4 * 6 = 24 (left: 24)\n', '(left: 24)']:
            with self.subTest(bad=bad):
                self.assertNotEqual(check(X, bad), 'valid')
        self.assertEqual(check(X, Y.replace('30 - 6 = 24 (left: 24)', '6 + 30 = 36 (left: 36)')), 'not_24')

    def test_no_wrong_arithmetic_missing_operands_or_left(self):
        for bad, reason in [
            (Y.replace('10 - 4', '10 - 10'), 'unavailable_operand'),
            (Y.replace('10 - 4 = 6', '10 - 4 = 7'), 'wrong_result'),
            (Y.replace('left: 5 6 6', 'left: 5 5 6'), 'wrong_left_multiset'),
            (Y.replace('left: 24', 'left: 24 24'), 'wrong_left_multiset'),
        ]:
            with self.subTest(bad=bad):
                self.assertEqual(check(X, bad), reason)
        zero = '1 - 1 = 0 (left: 0 4 6)\n6 / 0 = 6 (left: 4 6)\n6 * 4 = 24 (left: 24)'
        self.assertEqual(check('1 1 4 6', zero), 'division_by_zero')

    def test_fraction_exact_no_rounding(self):
        y = ('8 / 3 = 8/3 (left: 8 3 8/3)\n3 - 8/3 = 1/3 (left: 8 1/3)\n'
             '8 / 1/3 = 24 (left: 24)')
        self.assertEqual(check('3 3 8 8', y), 'valid')
        self.assertEqual(check('3 3 8 8', y.replace('= 8/3', '= 2')), 'wrong_result')

    def test_empirical_paired_interval_against_brute_force(self):
        # Independent enumeration of all resampled task-ID sequences for n=4.
        d = [-1, -1, 0, 1]
        vals = sorted(sum(xs) * 25 for xs in itertools.product(d, repeat=4))
        expected = [vals[(q*len(vals)+39)//40-1] for q in (1, 39)]
        self.assertEqual(A['paired_bootstrap'](1, 2, 4), expected)
        self.assertEqual(A['paired_bootstrap'](0, 0, 4), [0, 0])
        self.assertEqual(A['paired_bootstrap'](4, 0, 4), [100, 100])

    def test_all_records_and_accounting(self):
        result, table = A['build']()
        rows = list(csv.DictReader(io.StringIO(table.decode('utf-8'))))
        self.assertEqual(len({(r['idx'], r['arm']) for r in rows}), 90)
        self.assertEqual(len(rows), 90)
        for arm, any_n, first_n, valid_slots, cloud_tokens, edge_calls in [
            ('cloud', 26, 14, 84, 1979327, 0),
            ('edge_gen', 23, 13, 73, 1287342, 2108),
            ('warm', 25, 13, 83, 1251614, 2104),
        ]:
            rs = [r for r in rows if r['arm'] == arm]
            counts = [sum(int(r[k]) for r in rs) for k in ('strict_any', 'strict_first', 'strict_hits',
                       'cloud_total_tokens_increment', 'edge_total_calls_increment')]
            self.assertEqual(counts, [any_n, first_n, valid_slots, cloud_tokens, edge_calls])
            self.assertTrue(all(r['strict_valid_ranks'] == r['legacy_valid_ranks'] for r in rs))
            self.assertEqual(result['arms'][arm]['sensitivity']['final_pool_covered_tasks'], any_n)
        w, = [p for p in result['paired']['strict_any'] if p['a'] == 'warm' and p['b'] == 'cloud']
        self.assertEqual((w['both'], w['a_only_idx'], w['b_only_idx'], w['neither']), (24, [917], [904, 920], 3))
        self.assertIsNone(result['arms']['cloud']['historical_usage']['edge']['old_cost_field'])


if __name__ == '__main__':
    unittest.main()
