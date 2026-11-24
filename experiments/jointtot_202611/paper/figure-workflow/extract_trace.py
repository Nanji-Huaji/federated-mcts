"""Reconstruct the illustrated winning trace and its discarded sibling from the pinned warm-start archive."""
import argparse
import hashlib
import json
import re
from pathlib import Path

HERE=Path(__file__).resolve().parent
META=HERE.parents[1]/'archive_game24/summary.json'
REPO=HERE.parents[3]
LINE=re.compile(r'(\d+) ([+*/-]) (\d+) = (\d+) \(left: ([0-9 ]+)\)')

def operation(state,line):
    m=LINE.fullmatch(line); assert m, line
    a,op,b,c,rest=m.groups(); a,b,c=map(int,(a,b,c)); rem=state.copy()
    for arg in [a,b]: rem.remove(arg)
    if op=='/': assert b and a%b==0
    result={'+':lambda:a+b,'-':lambda:a-b,'*':lambda:a*b,'/':lambda:a//b}[op]()
    expected=sorted(map(int,rest.split())); assert result==c and c>=0 and sorted(rem+[c])==expected
    return {'operands':[a,op,b],'result':c,'state':expected}

def build():
    meta=json.loads(META.read_bytes()); source=meta['inputs']['warm_raw']; raw=(REPO/source['path']).read_bytes()
    digest=hashlib.sha256(raw).hexdigest(); assert digest==source['sha256']=='ef0e9e0dad15be81f3f6ba194e58b7b7cb60f26341ac8c54a709a089afb9d1f5'
    row=next(q for q in json.loads(raw) if q['idx']==900); rs=row['steps']; initial=list(map(int,rs[0]['x'].split())); assert initial==[4,5,6,10]
    winner=row['ys'][3]; lines=winner.splitlines();state=initial; prefix='';nodes=[]; branch=None
    for depth,(step,line) in enumerate(zip(rs,lines,strict=True)):
        opts=step['new_ys']; selected=step['select_new_ys']; vals=step['values']; ranks=sorted(range(len(opts)),key=lambda i:-vals[i]); assert [opts[i] for i in ranks[:10]]==selected and step['x']==rs[0]['x']
        if depth==1:
            sibling=prefix+'6 + 10 = 16 (left: 16 20)\n'; index=opts.index(sibling); branch=operation(state,sibling.splitlines()[-1]); assert sibling not in selected
            branch.update(parent_layer=1,candidate_layer=2,pool_index=index+1,stable_rank=ranks.index(index)+1,value=vals[index]); assert branch['stable_rank']==11
        prefix+=line+'\n'; o=operation(state,line);state=o['state']; index=opts.index(prefix); pos=selected.index(prefix)
        o.update(layer=depth+1,pool=len(opts),kept=len(selected),pool_index=index+1,stable_rank=pos+1,value=vals[index]);nodes.append(o)
    assert state==[24] and len(lines)==3 and winner==rs[-1]['select_new_ys'][3] and len(set(rs[-1]['values']))==1
    assert [n['pool'] for n in nodes]==[8,45,19] and [n['kept'] for n in nodes]==[8,10,10]
    assert nodes[0]['operands']==[4,'*',5] and nodes[1]['operands']==[10,'-',6] and nodes[2]['operands']==[4,'+',20]
    return {'source':source['path'],'source_sha256':digest,'arm':'warm','task_id':900,'initial':initial,'beam_cap':10,'stages':nodes,'pruned_sibling':branch,'all_final_values_tied':True,'verified_expression':'4 × 5 + (10 − 6) = 24','scope':'Archival selected trace and stable ranks; entire layer counts; terminal arithmetic verification is retrospective.'}

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--check',action='store_true');args=ap.parse_args(); target=HERE/'trace.json'; out=json.dumps(build(),ensure_ascii=False,indent=2)+'\n'
    if args.check: assert target.read_text()==out
    else: target.write_text(out)
    print('Source SHA, 3 retained prefixes, sibling rank, 3 whole-layer counts and final arithmetic verified.')
