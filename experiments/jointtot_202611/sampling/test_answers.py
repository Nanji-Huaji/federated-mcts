import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from answers import grade_answer, score_answer


class SubmittedExpressionTests(unittest.TestCase):
    def test_decodes_final_expression_not_earlier_work(self):
        self.assertEqual(grade_answer([1, 2, 3, 4], 'scratch work\nAnswer: (1 + 2 + 3) * 4 = 24\n'), 'valid')
        self.assertEqual(grade_answer([1, 2, 3, 4], '(1 + 2 + 3) * 4 = 24\nno solution'), 'format')
        self.assertEqual(grade_answer([1, 2, 3, 4], 'Final Answer: 4*(3+2+1)'), 'valid')
        self.assertEqual(grade_answer([1, 2, 3, 4], ''), 'empty')

    def test_multiplicity_claims_and_exact_rules(self):
        self.assertEqual(grade_answer([1, 2, 3, 4], '3*4*(1+1)'), 'wrong_leaves')
        self.assertEqual(grade_answer([1, 2, 3, 4], '24'), 'leaf_count')
        self.assertEqual(grade_answer([3, 3, 8, 8], '8/(3-8/3)'), 'inexact_division')
        self.assertEqual(grade_answer([1, 3, 4, 6], '(1-3)*(4*6)'), 'negative_intermediate')
        self.assertEqual(grade_answer([3, 3, 2, 6], '6*2/(3-3)'), 'division_by_zero')
        self.assertEqual(grade_answer([1, 2, 3, 4], '1+2+3+4=24'), 'not_24')
        self.assertEqual(grade_answer([1, 2, 3, 4], '(1+2+3)*4=23'), 'wrong_or_ambiguous_rhs')
        self.assertEqual(grade_answer([1, 2, 3, 4], '(1+2+3)*4=24=24'), 'wrong_or_ambiguous_rhs')
        self.assertEqual(grade_answer([4, 4, 6, 8], '(8+4)*(6-4)'), 'valid')
        self.assertEqual(grade_answer([2, 6, 2, 6], '6*(6-2)/2'), 'not_24')
        self.assertEqual(grade_answer([2, 6, 2, 6], '6*6-6*2'), 'wrong_leaves')
        self.assertEqual(grade_answer([8, 2, 3, 2], '(8/2)*3*2'), 'valid')

    def test_valid_structure_ignores_commutation_not_association(self):
        p = (1, 2, 3, 4)
        a = score_answer(p, '(1+2+3)*4')
        self.assertEqual(a, score_answer(p, '4 * (3 + (2+1))'))
        self.assertNotEqual(a, score_answer(p, '(1+(2+3))*4'))
        self.assertIsNone(score_answer(p, '2+2=24')[1])

    def test_only_whitelisted_syntax(self):
        for s in ('__import__("os")', 'pow(2,3)', '4**1+3+2', '(1+2+3)*4//1',
                  '1_2+3+4', '2*3*4*(+1)', '-1*2*3*4', '1.0*2*3*4',
                  '(1,2,3,4)[0]', '(1+2+3)*4 # text', '(True+2+3)*4', '٠١*2*3*4',
                  '1*(2*3)*4; 24', '0x1*2*3*4', '(1+2+3)*4' + '0'*260):
            self.assertEqual(grade_answer([1,2,3,4], s), 'format')


if __name__ == '__main__':
    unittest.main()
