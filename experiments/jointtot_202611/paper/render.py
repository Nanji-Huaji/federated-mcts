"""Render the checked English section from the five separate experiment summaries."""
import argparse
import hashlib
import json
import re
from decimal import Decimal
from pathlib import Path

HERE=Path(__file__).resolve().parent
BASE=HERE.parent

def load(src): return json.loads((BASE/src).read_text())
def texrows(rows): return '\n'.join(' & '.join(map(str,row))+r' \\' for row in rows)
def signed(n): return f'{n:+.2f}' if n else '0.00'
def interval(c): return f'[{signed(c[0])}, {signed(c[1])}]'
def num(n): return f'{n:,}'

def make():
    files=['archive_game24/summary.json','replay/summary.json','gsm8k/summary.json','local_judge/artifacts/summary.json','local_judge/artifacts/checks.json','sampling/summary.json']
    a,r,g,j,v,s=map(load,files); vals={}
    rows=[]
    for arm,name in [('cloud','All cloud'),('edge_gen','Edge proposes'),('warm','Edge + warm start')]:
        w=a['arms'][arm]; u=w['historical_usage']['cloud']; assert w['n_tasks']==30 and w['selected_slots']==300
        rows.append([name,w['strict_first_solved'],w['strict_any_solved'],num(u['total_calls']),f"{Decimal(u['old_cost_field']):.3f}"])
    w=a['arms']['warm']; pair=next(q for q in a['paired']['strict_any'] if (q['a'],q['b'])==('warm','cloud'))
    assert (pair['a_only'],pair['b_only'])==(1,2)
    vals.update(cloud_rows=texrows(rows),saving=f"{a['cloud_only_savings_vs_cloud']['warm']['cloud_cost_field_reduction_pct']:.2f}",warmdiff=signed(pair['difference_a_minus_b_pp']),warmci=interval(pair['paired_bootstrap95_pp']),edgetokens=num(w['historical_usage']['edge']['total_tokens']),breakeven=f"{Decimal(a['cloud_only_savings_vs_cloud']['warm']['cloud_cost_field_reduction_6dp']):.4f}")
    step=[r['by_depth'][str(i)]['all'] for i in (0,1)]; m0,m1=[x['methods'] for x in step]
    vals.update(live0=step[0]['live_pool_questions'],live1=step[1]['live_pool_questions'], b0solved=r['baseline_audit']['recorded_solved_questions'])
    assert [r['baseline_audit']['stable_order_matches_by_depth'][i] for i in (0,1,2)]==[200,200,200]
    rr=[]
    for k,name in [('stable','Recorded votes'),('progress','Remove unchanged parent'),('state_max','Merge equal states'),('progress_state_max','Remove + merge'),('zero_uniform','Lottery on zero votes')]:
        counts=[m[k]['hits'] for m in (m0,m1)];rr.append([name]+[f'{c:.2f}' if k=='zero_uniform' else c for c in counts])
    diff=m1['progress_state_max']; assert m1['progress']['gained']==5 and m1['state_max']['gained']==1
    vals.update(replay_rows=texrows(rr), gained=diff['gained'],lost=diff['lost'],replaydiff=signed(diff['delta_pp']),replayci=interval(diff['paired_ci95_pp']))
    assert j['n']==20 and v['new_forward_prompts']==80 and v['oracle_crosschecks']==80
    jr=[]
    for k in ['7B','32B']:
        z=j['models'][k]; jr.append([k,z['primary']['top1'],z['primary']['top2'],z['reverse']['top1'],z['same_winner_after_reversal']])
    lo,hi=v['paired_percentile95_pp']; interpret=('The interval includes zero; this pilot does not resolve a model-size advantage.' if lo<=0<=hi else 'This difference is specific to the fixed pilot and scoring format.')
    rev=[j['models'][k]['reverse']['top1'] for k in ('7B','32B')]
    if rev[0]==rev[1]: interpret=f'Reversing the options erased the top-one gap: both models hit {rev[0]} of 20. The size effect depends on presentation.'
    vals.update(judge_rows=texrows(jr),prompts=v['new_forward_prompts'],eligible=j['pool_has_solution'],chance=f"{j['uniform_random_expected_hits']['top1']:g}",largeonly=v['primary_larger_only'],smallonly=v['primary_smaller_only'],scalediff=signed(v['primary_larger_minus_smaller_pp']),scaleci=interval([lo,hi]),judge_interpretation=interpret)
    old,cur=[g['paired_any'][k] for k in ('archived','frozen')]; totals=g['resources_historical']['totals']
    vals.update(g_old0=old['tot_correct'],g_old1=old['speculative_solve_correct'],g_new0=cur['tot_correct'],g_new1=cur['speculative_solve_correct'],gdiff=signed(cur['spec_minus_tot_pp']),gci=interval(cur['paired_percentile_95_pp']),gt0=num(totals['tot']['all_models']['total_tokens']),gt1=num(totals['speculative_solve']['all_models']['total_tokens']))
    vals['sampling']=sampling_text(s)
    template=(HERE/'experiments.template').read_text(); assert set(re.findall(r'\[\[(\w+)\]\]',template))==set(vals)
    text=re.sub(r'\[\[(\w+)\]\]',lambda m:str(vals[m[1]]),template)
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    evidence={'inputs':{f:digest(BASE/f) for f in files},'template_sha256':digest(HERE/'experiments.template'),'rendered_sha256':hashlib.sha256(text.encode()).hexdigest()}
    return {'jointtot-experiments.tex':text,'evidence-index.json':json.dumps(evidence,indent=2)+'\n'}


def sampling_text(s):
    # The sampling audit has its own cohort/provenance; never pool it with IDs 900--929.
    from sampling_paragraph import paragraph
    return paragraph(s)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');args=p.parse_args()
    for name,text in make().items():
        dest=HERE/name
        if args.check:
            if dest.read_text()!=text: raise SystemExit(f'{name}: stale; rerender with render.py')
        else: dest.write_text(text)
        print(name)
