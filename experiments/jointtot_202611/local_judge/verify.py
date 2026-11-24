"""Verify the archived local forwards, ranks and controlled questions; never loads a model."""
from collections import Counter
from fractions import Fraction
import argparse
import itertools
import json
import math
import unittest
from pathlib import Path
import run

INPUT_SHA = '54a574ef35cfe3f42dec8f8719ac1265c548b5d31063f30a1f83863519fa51be'


class ProtocolTests(unittest.TestCase):
    def test_move_consumes_occurrences(self):
        self.assertEqual(run.first_state([1,1,3,8], '1 + 1 = 2 (left: 2 3 8)\n'), (2,3,8))
        self.assertIsNone(run.first_state([1,2,3,8], '1 + 1 = 2 (left: 2 3 8)'))
        self.assertIsNone(run.first_state([1,1,3,8], '1 + 1 = 2 (left: 2 1 8)'))
    def test_move_checks_arithmetic(self):
        self.assertIsNone(run.first_state([1,1,3,8], '8 / 3 = 2 (left: 1 1 2)'))
        self.assertIsNone(run.first_state([1,1,3,8], '3 - 8 = -5 (left: 1 1 -5)'))
    def test_verifier_examples(self):
        self.assertTrue(run.reachable((1,3,8)))
        self.assertTrue(run.reachable((2,4,4)))
        self.assertFalse(run.reachable((1,1,1)))
        self.assertFalse(run.reachable((3,3,8,8)))  # rational-only solution not an integer path
    def test_order_reverses_identities(self):
        row={'options':[(1,2,3),(2,3,4),(3,4,5),(4,5,6)]}
        self.assertEqual(run.display(row,True)[3],run.display(row,False)[0])


def independent_live(nums):
    # Independent subset-DP, not the recursive search used by run.summary.
    cache={1<<i:{n} for i,n in enumerate(nums)}
    for mask in range(1,1<<len(nums)):
        if mask in cache: continue
        ans=set(); sub=(mask-1)&mask
        while sub:
            other=mask^sub
            if sub<other:
                for a,b in itertools.product(cache[sub],cache[other]):
                    ans |= {a+b,a*b,abs(a-b)}
                    if b and a%b==0: ans.add(a//b)
                    if a and b%a==0: ans.add(b//a)
            sub=(sub-1)&mask
        cache[mask]=ans
    return 24 in cache[(1<<len(nums))-1]


def validate(root):
    b=(root/'inputs.json').read_bytes(); assert run.sha(b)==INPUT_SHA
    items=json.loads(b)['items']; assert [r['task_id'] for r in items]==list(run.PILOT)
    chosen={}; bits={}; summ=json.loads((root/'summary.json').read_text()); lookup={r['task_id']:r for r in summ['per_task']}
    for item in items:
        assert len(set(tuple(s) for s in item['options']))==4 and len(item['input'])==4
        bits[item['task_id']]=[int(independent_live(s)) for s in item['options']]
        assert bits[item['task_id']]==lookup[item['task_id']]['reachable']
    score_hashes={}
    for size, snapshot in run.SIZES.items():
        raw=(root/f'scores_{size}.json').read_bytes(); score_hashes[size]=run.sha(raw); scores=json.loads(raw)
        assert scores['input_sha256']==INPUT_SHA and scores['snapshot']==snapshot
        assert scores['probe_sha256']==run.sha(Path(run.__file__).read_bytes())
        assert scores['model']==f'Qwen/Qwen2.5-{size}-Instruct' and scores['device']=='cpu'
        assert scores['dtype']=='bfloat16' and scores['letter_token_ids']==[32,33,34,35]
        records=scores['records']; assert len(records)==2*len(items)
        chosen[size]={}
        for r in records:
            i, rev, logits=r['task_id'],r['reverse'],r['letter_logits']; assert len(logits)==4 and all(math.isfinite(x) for x in logits)
            ranks=sorted(range(4),key=lambda k:(-logits[k],k)); indexes=[3-k if rev else k for k in ranks]
            own=lookup[i][f'{size}_{"reverse" if rev else "primary"}']; assert own==dict(top_state_index=indexes[0],top1=bits[i][indexes[0]],top2=int(any(bits[i][j] for j in indexes[:2])))
            chosen[size][i,rev]=(r['prompt_sha256'],r['input_tokens'],own['top1'])
    assert set(chosen['7B'])==set(chosen['32B'])==set(itertools.product(run.PILOT,[False,True]))
    assert all(chosen['7B'][k][:2]==chosen['32B'][k][:2] for k in chosen['7B'])
    diffs=[chosen['32B'][i,False][2]-chosen['7B'][i,False][2] for i in run.PILOT]; n=len(diffs); s,t=diffs.count(1),diffs.count(-1)
    # Invert the empirical three-outcome distribution exactly (only 20 puzzle pairs).
    hist=Counter(); plus,minus,zero=Fraction(s,n),Fraction(t,n),Fraction(n-s-t,n)
    for a in range(n+1):
        for c in range(n+1-a):
            remain=n-a-c; hist[a-c]+=math.comb(n,a)*math.comb(n-a,c)*plus**a*minus**c*zero**remain
    def quant(q):
        mass=Fraction()
        for d,prob in sorted(hist.items()):
            mass+=prob
            if mass>=q: return 100*d/n
        raise AssertionError
    paired_p=min(1,2*sum(math.comb(s+t,k) for k in range(min(s,t)+1))/2**(s+t)) if s+t else 1.
    checks=dict(n_puzzles=n,n_unique_presentations_per_model=2*n,input_sha256=INPUT_SHA,score_sha256=score_hashes,
        conditional_ceiling=sum(any(v) for v in bits.values()),new_forward_prompts=4*n,prompt_identity_checks=2*n,
        oracle_crosschecks=sum(len(m) for m in bits.values()),primary_larger_only=s,primary_smaller_only=t,
        primary_larger_minus_smaller_pp=100*sum(diffs)/n,paired_percentile95_pp=[quant(Fraction(1,40)),quant(Fraction(39,40))],
        exact_two_sided_mcnemar=paired_p)
    return checks


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--out',type=Path,default=run.HERE/'artifacts'); p.add_argument('--tests-only',action='store_true'); p.add_argument('--check',action='store_true'); args=p.parse_args()
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(ProtocolTests)
    if not unittest.TextTestRunner().run(suite).wasSuccessful(): raise SystemExit(1)
    if not args.tests_only:
        report=validate(args.out); path=args.out/'checks.json'
        if args.check:
            if json.loads(path.read_text())!=report: raise SystemExit('changed results: checks.json')
        else: run.save(path,report)
        print(json.dumps(report,indent=2))
