"""End-to-end tests on the original frozen archives; no model substitutions."""
import csv
import io
from decimal import Decimal
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit import build, compare, paired_interval, require_pairing, sha


class RealArchiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary, cls.table = build()
        cls.rows = list(csv.DictReader(io.StringIO(cls.table.decode())))

    def test_all_bound_and_all_submissions_retained(self):
        s, rows = self.summary, self.rows
        self.assertEqual(s['pairing']['matched_slots'], 1480)
        self.assertEqual(len(rows), 148)
        self.assertEqual([int(r['idx']) for r in rows], list(range(148)))
        self.assertEqual([int(r['dataset_rank']) for r in rows], list(range(100,200))+list(range(500,548)))
        self.assertEqual(len({r['input_sha256'] for r in rows}), 148)
        for r in rows:
            self.assertEqual(len(r['complete_verdicts_batch_order'].split(';')), 10)
            self.assertEqual(len(r['tree_verdicts_rank_order'].split(';')), int(r['tree_slots']))
        self.assertEqual(sum(int(r['tree_slots']) for r in rows), 1458)
        self.assertEqual(sum(int(r['complete_valid_slots']) for r in rows), 760)
        self.assertEqual(sum(s['complete']['strict_verdicts'].values()), 1480)
        self.assertGreater(s['complete']['strict_verdicts']['format'], 0)
        self.assertEqual(s['per_task_csv_sha256'], sha(self.table))

    def test_order_and_coverage_with_paired_discordance(self):
        p = self.summary['primary_paired_full_tree_vs_complete_k']
        self.assertEqual([p[k]['complete_solved'] for k in ('1','3','10')], [77,106,130])
        self.assertEqual([p[k]['tree_solved'] for k in ('1','3','10')], [132]*3)
        self.assertEqual([p['10'][s] for s in ('both','tree_only','complete_only','neither')], [124,8,6,10])
        q = self.summary['sensitivity_paired_tree_topk_vs_complete_k']
        self.assertEqual([q[k]['tree_solved'] for k in ('1','3','10')], [87,124,132])
        self.assertEqual(self.summary['complete']['duplicates']['duplicate_slots_after_first'], 311)
        self.assertEqual(self.summary['complete']['duplicates']['distinct_task_valid_expression_shapes'], 249)
        m = self.summary['complete']['derived_files_check']
        self.assertEqual(m['distinct_full_response_streams'], 10)
        self.assertEqual(m['merged_slot_to_original_batch_zero_based'], [4,9,6,8,1,5,3,2,7,0])

    def test_counters_are_deltas_not_repeated_tree_prefix_prices(self):
        u = self.summary['usage']['totals_by_arm']
        self.assertEqual([u[k]['total_calls'] for k in ('complete_k1','complete_k3','complete_k10','tree_full')],
                         [148,444,1480,16048])
        self.assertEqual(u['complete_k10']['total_tokens'], 976116)
        self.assertEqual(u['tree_full']['total_tokens'], 10182831)
        self.assertLess(abs(Decimal(u['complete_k10']['cost'])-Decimal('1.311392')), Decimal('1e-12'))
        self.assertLess(abs(Decimal(u['tree_full']['cost'])-Decimal('10.932474')), Decimal('1e-12'))
        self.assertNotIn('tree_top1_cost', self.rows[0])


class ProtocolTests(unittest.TestCase):
    def test_no_match_by_idx_alone(self):
        runs = [{'idx': i, 'input': (2,3,4,5)} for i in range(148)]
        bound = {(b, i): (2,3,4,5) for b in range(10) for i in range(148)}
        require_pairing(bound, runs)
        bound[0,0] = (2,4,4,4)
        with self.assertRaises(ValueError):
            require_pairing(bound, runs)
        del bound[0,0]
        with self.assertRaises(ValueError):
            require_pairing(bound, runs)

    def test_paired_interval_draws_tasks_not_candidate_slots(self):
        self.assertEqual(paired_interval(0,0,12), [0.0,0.0])
        self.assertEqual(paired_interval(1,1,2), [-100.0,100.0])
        pairs = [{'x': 0, 'y': 1}, {'x': 0, 'y': 1}, {'x': 1, 'y': 0}]
        x = compare(pairs, 'x', 'y')
        self.assertEqual((x['tree_only'], x['complete_only'], x['tasks']), (2,1,3))


if __name__ == '__main__':
    unittest.main()
