"""Offline paired first-step judge probe. Prepare is label-blind; infer never calls the oracle."""
import argparse
import hashlib
import itertools
import json
import math
import os
import re
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PILOT = (3, 9, 16, 19, 24, 25, 28, 36, 40, 42, 45, 71, 77, 81, 102, 104, 105, 116, 117, 119)
ARCHIVE_SHA = '24b3754c03cd7978ebb008550d99a6681c55f6baeb86f36e7a18401771dac765'
SEED = 1729
SIZES = {'7B': 'a09a35458c702b33eeacc393d103063234e8bc28', '32B': '5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd'}
LABELS = 'ABCD'
INSTRUCTION = ('Game of 24. Pick the remaining triple that can make exactly 24. Use each number once '\
               'and combine two numbers per step with +, -, *, or /. Intermediate values must be '\
               'nonnegative integers; division must be exact. Reply with one letter.\n')


def sha(b):
    return hashlib.sha256(b).hexdigest()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def first_state(task, y):
    m = re.fullmatch(r'(\d+)\s*([*/+\-])\s*(\d+)\s*=\s*(\d+)\s*\(left:\s*([\d ]+)\)', y.strip())
    if not m:
        return None
    a, b, ans = map(int, (m[1], m[3], m[4])); op = m[2]
    expected = {'+': a+b, '-': a-b, '*': a*b, '/': a//b if b and a%b == 0 else None}[op]
    rest = Counter(task)
    if expected != ans or not (Counter([a,b]) <= rest):
        return None
    rest.subtract([a,b]); rest[ans] += 1
    left = tuple(sorted(map(int, m[5].split())))
    return left if len(left) == 3 and Counter(left) == +rest else None


def prepare(out):
    source, = (ROOT/'logs/game24/tot/qwen3.7-plus').glob('*_ntasks200.json')
    b = source.read_bytes(); assert sha(b) == ARCHIVE_SHA
    rows = {r['idx']: r['steps'][0] for r in json.loads(b) if r['idx'] in PILOT}
    assert set(rows) == set(PILOT)
    items = []
    for i in PILOT:
        s = rows[i]; assert s['step'] == 0
        nums = tuple(map(int, s['x'].split()))
        states = {x for y in s['new_ys'] if (x := first_state(nums, y)) is not None}
        def order(state):
            return sha(f'{SEED}|{i}|{",".join(map(str,state))}'.encode()), state
        four = sorted(states, key=order)[:4]
        if len(four) != 4: raise ValueError(f'Fewer than four valid distinct states at {i}')
        items.append(dict(task_id=i, input=list(nums), valid_pool_size=len(states), options=four))
    pack = dict(archive_sha256=ARCHIVE_SHA, source=str(source.relative_to(ROOT)), seed=SEED,
                protocol='root4-v1: fixed pilot, SHA order, original + reversed, no votes or oracle', items=items)
    save(out/'inputs.json',pack)
    print('frozen', len(items), 'puzzles;', sha((out/'inputs.json').read_bytes()), flush=True)


def display(item, reverse):
    return list(reversed(item['options'])) if reverse else item['options']


def infer(size, out, hub, threads, batch_size):
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM
    torch.set_num_threads(threads); torch.set_num_interop_threads(1)
    torch.manual_seed(SEED)
    name = f'Qwen2.5-{size}-Instruct'
    snap = hub/f'models--Qwen--{name}'/'snapshots'/SIZES[size]
    index = json.loads((snap/'model.safetensors.index.json').read_text())
    shards = sorted(set(index['weight_map'].values()))
    files = {w:dict(blob=(snap/w).resolve().name, bytes=(snap/w).stat().st_size) for w in shards}
    tok = AutoTokenizer.from_pretrained(snap, local_files_only=True, trust_remote_code=False)
    tok.padding_side = 'left'
    ids = [tok.encode(label, add_special_tokens=False) for label in LABELS]
    assert all(len(x)==1 for x in ids) and len({x[0] for x in ids})==4
    input_bytes = (out/'inputs.json').read_bytes(); items = json.loads(input_bytes)['items']
    asks = []
    for item in items:
        for reverse in (False, True):
            options = display(item, reverse)
            text = INSTRUCTION + '\n'.join(f'{label}: {" ".join(map(str, state))}' for label,state in zip(LABELS, options))
            messages = [{'role':'system','content':'You solve arithmetic puzzles.'},{'role':'user','content':text}]
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            asks.append((item['task_id'], reverse, prompt))
    print('loading', name, 'CPU/bfloat16', flush=True)
    start = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(snap, torch_dtype=torch.bfloat16,
        attn_implementation='eager', local_files_only=True, trust_remote_code=False).eval()
    load_s = time.monotonic()-start; start=time.monotonic()
    results=[]
    with torch.inference_mode():
        for offset in range(0,len(asks),batch_size):
            group = asks[offset:offset+batch_size]
            inputs = tok([q[2] for q in group],padding=True,return_tensors='pt',add_special_tokens=False)
            positions=(inputs.attention_mask.cumsum(-1)-1).clamp(min=0)
            scores = model(**inputs,position_ids=positions,use_cache=False,logits_to_keep=1).logits[:, -1, [x[0] for x in ids]].float().tolist()
            for q, score, tokens in zip(group,scores,inputs.attention_mask.sum(dim=1).tolist()):
                assert all(math.isfinite(s) for s in score)
                results.append(dict(task_id=q[0], reverse=q[1], prompt_sha256=sha(q[2].encode()),
                                    input_tokens=tokens, letter_logits=score))
            print(name,len(results),'/',len(asks),round(time.monotonic()-start,1),'seconds',flush=True)
    save(out/f'scores_{size}.json',dict(model=f'Qwen/{name}', snapshot=SIZES[size], config_sha256=sha((snap/'config.json').read_bytes()),
        shards=files, input_sha256=sha(input_bytes), probe_sha256=sha(Path(__file__).read_bytes()),
        torch=str(torch.__version__),transformers=transformers.__version__,device='cpu',dtype='bfloat16',
        letter_token_ids=[x[0] for x in ids],threads=threads,batch=batch_size,load_seconds=load_s,
        forward_seconds=time.monotonic()-start,records=results))


@lru_cache(None)
def reachable(state):
    if len(state)==1: return state==(24,)
    for i,j in itertools.combinations(range(len(state)),2):
        a,b=state[i],state[j]; rest=[x for k,x in enumerate(state) if k not in (i,j)]
        vals={a+b,a*b,abs(a-b)}
        if b and a%b==0: vals.add(a//b)
        if a and b%a==0: vals.add(b//a)
        if any(reachable(tuple(sorted(rest+[v]))) for v in vals): return True
    return False


def summary(out):
    b=(out/'inputs.json').read_bytes(); items={q['task_id']:q for q in json.loads(b)['items']}
    assert set(items)==set(PILOT)
    rows={i:{'task_id':i,'reachable': [int(reachable(tuple(s))) for s in items[i]['options']]} for i in PILOT}
    counts={}
    for size in SIZES:
        source=json.loads((out/f'scores_{size}.json').read_text()); assert source['input_sha256']==sha(b) and source['snapshot']==SIZES[size]
        records=source['records']; assert len(records)==2*len(PILOT) and {(r['task_id'], r['reverse']) for r in records}==set(itertools.product(PILOT,[False,True]))
        for r in records:
            logits=r['letter_logits']; assert len(logits)==4 and all(math.isfinite(x) for x in logits)
            rank=sorted(range(4), key=lambda k:(-logits[k], k)); rank=[3-k if r['reverse'] else k for k in rank]
            row=rows[r['task_id']]; mask=row['reachable']
            row[f'{size}_{"reverse" if r["reverse"] else "primary"}']={'top_state_index':rank[0],'top1':mask[rank[0]],'top2':int(any(mask[k] for k in rank[:2]))}
        def total(order, key): return sum(row[f'{size}_{order}'][key] for row in rows.values())
        counts[size] = {order:{'top1':total(order,'top1'),'top2':total(order,'top2')} for order in ['primary','reverse']}
        counts[size]['same_winner_after_reversal']=sum(row[f'{size}_primary']['top_state_index']==row[f'{size}_reverse']['top_state_index'] for row in rows.values())
    eligible=sum(any(r['reachable']) for r in rows.values())
    random1=sum(sum(r['reachable'])/4 for r in rows.values()); random2=sum(1-math.comb(4-sum(r['reachable']),2)/6 for r in rows.values())
    pairs=Counter((r['7B_primary']['top1'],r['32B_primary']['top1']) for r in rows.values())
    save(out/'summary.json',dict(input_sha256=sha(b),n=len(rows),pool_has_solution=eligible,
        uniform_random_expected_hits=dict(top1=random1,top2=random2),models=counts,
        primary_top1_pairs_7B_32B={f'{a}{b}':pairs[a,b] for a,b in itertools.product(range(2),repeat=2)},per_task=list(rows.values())))
    print('models',counts,'pairs',pairs,'eligible',eligible,'random',random1,random2)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('stage',choices=['prepare','infer','summary']); p.add_argument('--size',choices=SIZES)
    p.add_argument('--out',type=Path,default=HERE/'artifacts'); p.add_argument('--hub',type=Path,default=Path.home()/'.cache/huggingface/hub')
    p.add_argument('--threads',type=int,default=12); p.add_argument('--batch',type=int,default=4); args=p.parse_args()
    if args.stage=='prepare': prepare(args.out)
    elif args.stage=='infer':
        if not args.size: p.error('--size required for infer')
        os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'; os.environ['TOKENIZERS_PARALLELISM']='false'
        infer(args.size,args.out,args.hub,args.threads,args.batch)
    else: summary(args.out)
