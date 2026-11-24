#!/usr/bin/env python3
"""Offline comparison of archived complete responses and a width-10 tree on recovered tasks."""
import argparse
from collections import Counter
import csv
from decimal import Decimal
import hashlib
import io
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))  # also works with python -ISB
from answers import score_answer
from identity import load_identity
from tree_archive import load_tree_archive

N, REPEATS, KS = 148, 10, (1, 3, 10)
NAIVE = Path('logs/game24/naive/deepseek-v3')
MODEL = 'deepseek-v3'
COUNTERS = ('total_calls', 'prompt_tokens', 'completion_tokens', 'total_tokens')


def require(ok, what):
    if not ok:
        raise ValueError(what)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def hash_json(value):
    return sha(json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode())


def pinned_json(path, expected):
    blob = path.read_bytes()
    require(sha(blob) == expected, 'archive bytes differ: ' + path.name)
    return json.loads(blob, parse_float=Decimal)


def delta(cur, previous):
    require(set(cur) == {*COUNTERS, 'cost'}, 'unexpected usage fields')
    require(all(type(cur[k]) is int and cur[k] >= 0 for k in COUNTERS), 'invalid counter')
    require(cur['total_tokens'] == cur['prompt_tokens'] + cur['completion_tokens'], 'token sum mismatch')
    expected = Decimal(cur['prompt_tokens'] + 2 * cur['completion_tokens']) / 10**6
    require(abs(cur['cost'] - expected) < Decimal('1e-12'), 'historic tariff mismatch')
    x = {k: cur[k] - previous.get(k, 0) for k in (*COUNTERS, 'cost')}
    require(x['total_calls'] == 1 and min(x.values()) >= 0, 'expected one call increment per response')
    return x


def sum_usage(items):
    items = list(items)
    return {**{k: sum(u[k] for u in items) for k in COUNTERS},
            'cost': format(sum((Decimal(u['cost']) for u in items), Decimal(0)), 'f')}


def load_answers(root, identity):
    streams, finals = [], []
    for b in range(REPEATS):
        records = pinned_json(root / NAIVE / f'{b}.json', identity['batch_sha256'][str(b)])
        require([r['idx'] for r in records] == list(range(N)), 'wrong batch idx sequence')
        prev, slots = {}, []
        for r in records:
            require(len(r['ys']) == len(r['infos']) == 1 and isinstance(r['ys'][0], str), 'not one response')
            require(set(r['usage_so_far']) == {MODEL}, 'wrong model accounting keys')
            require(r['time_consumption'] == {'generation': 0, 'evaluation': 0}, 'unexpected timing evidence')
            u = r['usage_so_far'][MODEL]
            slots.append({'answer': r['ys'][0], 'old_r': r['infos'][0]['r'], 'budget': delta(u, prev)})
            prev = u
        streams.append(slots)
        finals.append(prev)
    return streams, finals


def check_merged(root, streams, finals):
    # These are derived files, not eleven extra runs. Confirm their precise relation to the sources.
    hashes = {'merged_results.json': '300fdd1d5feafbe06890e782a42c256a06dab352557a56f81dc2b266b63b96a7',
              'merged.json': 'f85e87629d6b48ea19dd79f83b5e35a0b909416df9ea6c286c7a6dde4807530a'}
    merged, perf = [pinned_json(root / NAIVE / name, h) for name, h in hashes.items()]
    require([r['idx'] for r in merged] == list(range(N)) and all(len(r['ys']) == REPEATS for r in merged),
            'inconsistent merge shape')
    original = [tuple(x['answer'] for x in s) for s in streams]
    perm = []
    for slot in range(REPEATS):
        col = tuple(r['ys'][slot] for r in merged)
        matches = [i for i, s in enumerate(original) if s == col]
        require(len(matches) == 1, 'ambiguous merged response streams')
        perm.append(matches[0])
    require(set(perm) == set(range(REPEATS)), 'not a permutation of ten batches')
    agg, mu = sum_usage(finals), merged[-1]['usage_so_far'][MODEL]
    require(all(agg[k] == mu[k] for k in COUNTERS) and abs(Decimal(agg['cost']) - mu['cost']) < Decimal('1e-12'),
            'merged terminal accounting mismatch')
    require(perf[MODEL] == finals[-1], 'merged summary is not last run')
    return {'source_sha256': hashes, 'merged_slot_to_original_batch_zero_based': perm,
            'merged_results_terminal_counters_match_sum': True, 'merged_json_counters_match_last_batch_only': True,
            'ordered_answer_stream_sha256_by_digit': [hash_json(s) for s in original],
            'stream_hash_encoding': 'SHA-256 of UTF-8 compact JSON string array in idx order; no LF',
            'distinct_full_response_streams': len(set(original))}


def require_pairing(mapping, runs):
    require([r['idx'] for r in runs] == list(range(N)), 'unexpected tree indices')
    require(set(mapping) == {(b, i) for b in range(REPEATS) for i in range(N)}, 'incomplete historical bindings')
    for r in runs:
        require(all(tuple(mapping[b, r['idx']]) == tuple(r['input']) for b in range(REPEATS)),
                'historical four numbers disagree with the tree; refusing comparison')


def paired_interval(gain, loss, n):
    """Exact percentiles of the empirical paired task bootstrap (no Monte Carlo)."""
    require(n > 0 and min(gain, loss, n-gain-loss) >= 0, 'inconsistent pairs')
    draws = [(s, count) for s, count in [(1,gain), (-1,loss), (0,n-gain-loss)] if count]
    mass = {0: 1}
    for _ in range(n):
        nxt = Counter()
        for diff, ways in mass.items():
            for d, count in draws:
                nxt[diff+d] += ways * count
        mass = nxt
    total = n ** n
    require(sum(mass.values()) == total, 'bootstrap mass mismatch')
    acc, bounds = 0, []
    for d, count in sorted(mass.items()):
        acc += count
        while len(bounds) < 2 and 40 * acc >= (1,39)[len(bounds)] * total:
            bounds.append(round(100 * d/n, 6))
    return bounds


def compare(rows, naive_col, tree_col):
    cells = Counter((r[naive_col], r[tree_col]) for r in rows)
    both, tree_only, naive_only, neither = (cells[k] for k in [(1,1), (0,1), (1,0), (0,0)])
    n = len(rows)
    return {'tasks': n, 'both': both, 'tree_only': tree_only, 'complete_only': naive_only, 'neither': neither,
            'complete_solved': both + naive_only, 'tree_solved': both + tree_only,
            'tree_minus_complete_pp': round(100 * (tree_only-naive_only)/n, 6),
            'paired_task_bootstrap95_pp': paired_interval(tree_only, naive_only, n)}


def build(root=ROOT):
    mapping, identity = load_identity(root)
    require(identity['run_to_batch'] == list(range(REPEATS)), 'ordinal and digit order differ')
    tree = load_tree_archive(root)
    require_pairing(mapping, tree['runs'])
    streams, finals = load_answers(root, identity)
    merged = check_merged(root, streams, finals)
    rows, reasons, equal_task_pairs, by_batch = [], Counter(), Counter(), [0] * REPEATS
    identical_counts, raw_diff_from_old = Counter(), 0
    for t in tree['runs']:
        idx, puzzle = t['idx'], t['input']
        sample = [stream[idx] for stream in streams]
        texts = [s['answer'] for s in sample]
        # Inputs are historically bound *before* reading/crediting candidate expressions.
        graded = [score_answer(mapping[b, idx], text) for b, text in enumerate(texts)]
        v = [reason for reason, _ in graded]
        distinct_valid_shapes = len({key for reason, key in graded if reason == 'valid'})
        reasons.update(v)
        h = [int(s == 'valid') for s in v]
        raw_diff_from_old += sum(new != s['old_r'] for new, s in zip(h, sample))
        for i in range(REPEATS):
            by_batch[i] += h[i]
            for j in range(i+1, REPEATS):
                equal_task_pairs[f'{i},{j}'] += int(texts[i] == texts[j])
        distinct = len(set(texts))
        identical_counts[distinct] += 1
        cand = t['candidates']
        require(1 <= len(cand) <= REPEATS, 'unexpected retained width')
        th = [int(c['valid']) for c in cand]
        row = {'idx': idx, 'dataset_rank': t['dataset_rank'],
               'input_sha256': sha(' '.join(map(str, puzzle)).encode('ascii')),
               'complete_slots': len(sample), 'complete_distinct_texts': distinct,
               'complete_verdicts_batch_order': ';'.join(v), 'complete_valid_slots': sum(h),
               'complete_distinct_valid_shapes': distinct_valid_shapes,
               'tree_slots': len(cand), 'tree_distinct_texts': len({c['raw_sha256'] for c in cand}),
               'tree_verdicts_rank_order': ';'.join(c['reason'] for c in cand), 'tree_valid_slots': sum(th),
               'tree_full': int(any(th))}
        for k in KS:
            row[f'complete_k{k}'] = int(any(h[:k]))
            row[f'tree_top{k}'] = int(any(th[:k]))
            u = sum_usage(s['budget'] for s in sample[:k])
            row.update({f'complete_k{k}_{key}': z for key, z in u.items()})
        row.update({f'tree_full_{key}': t['budget'][key] for key in (*COUNTERS, 'cost')})
        rows.append(row)
    n = len(rows)
    budgets = {f'complete_k{k}': sum_usage(finals[:k]) for k in KS}
    budgets['tree_full'] = sum_usage(r['budget'] for r in tree['runs'])
    for arm, expected in budgets.items():
        u = sum_usage({f: r[arm+'_'+f] for f in (*COUNTERS, 'cost')} for r in rows)
        require(all(u[f] == expected[f] for f in COUNTERS) and Decimal(u['cost']) == Decimal(expected['cost']),
                'per-task increments do not reconcile')
    buf = io.StringIO(newline='')
    writer = csv.DictWriter(buf, list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    data = buf.getvalue().encode()
    summary = {
        'schema': 'game24.sampling.paired.v1', 'evidence': 'Offline rescoring of real archived responses; no inference.',
        'tasks': n, 'per_task_csv_sha256': sha(data),
        'identity': identity,
        'pairing': {'rule': 'For every (batch, idx), the four ordered integers bound without consulting answer arithmetic match the archived tree input at that idx.',
                    'meaning_of_bound': 'Reconstructed from historical recipe, snapshot and exact response/collector evidence; not prompt capture.',
                    'matched_slots': len(mapping), 'primary_cohort': 'All 148, including no solution and every unparseable output.',
                    'input_sha256_encoding': 'SHA-256 of four ordered ASCII decimal integers separated by single spaces, no LF',
                    'current_csv_one_based_ranks_in_order': [[100,199], [500,547]]},
        'protocol': {'task_hit': 'At least one strictly verified submission in the fixed prefix; no repair, search or averaging over permutations.',
                     'complete': 'Digit files in chronological order 0..k-1. Final nonblank line (<=256 chars), optional Answer: / Final Answer:, optional = 24. ASCII decimal integer leaves; + - * /; parentheses and whitespace. Allowlisted AST. Exactly four leaves with the actual multiset; all intermediate results nonnegative integers; no division by zero or remainder; final 24. Earlier CoT is not an alternate submission.',
                     'tree': 'Primary: any of the final retained trajectories, not all generated nodes. Exactly three legal binary steps from the four inputs, same integer rules, verify results and every declared left multiset. Sensitivity: first min(k, retained).',
                     'independence': 'Repeated full-answer requests; no evidence of RNG seeds, cache settings, server revision or statistical independence. Tree slots share frontiers.',
                     'interval': 'Exact 2.5%/97.5% percentiles resampling matched tasks with replacement from this fixed cohort (148 draws); descriptive, not proof of significance/equivalence or representative held-out accuracy.'},
        'complete': {'slots': n*REPEATS, 'strict_verdicts': dict(sorted(reasons.items())),
                     'strict_hits_by_batch': by_batch,
                     'legacy_positive_slots': sum(s['old_r'] for stream in streams for s in stream),
                     'strict_vs_legacy_disagreements': raw_diff_from_old,
                     'solved_in_first_k': {str(k): sum(r[f'complete_k{k}'] for r in rows) for k in KS},
                     'duplicates': {'equality': 'Exact decoded response strings, including whitespace, for the same historically bound task.',
                                    'tasks_by_distinct_texts_out_of_ten': dict(sorted(identical_counts.items())),
                                    'unique_task_text_pairs': sum(d*c for d,c in identical_counts.items()),
                                    'duplicate_slots_after_first': sum((REPEATS-d)*c for d,c in identical_counts.items()),
                                    'distinct_task_valid_expression_shapes': sum(r['complete_distinct_valid_shapes'] for r in rows),
                                    'valid_shape_definition': 'AST of a verified answer, constant integer leaves; ignore whitespace/parentheses redundant for grouping and commute +/* only. No associativity or algebraic simplification.',
                                    'same_output_tasks_by_pair_of_digit_files': dict(equal_task_pairs)},
                     'derived_files_check': merged},
        'tree': tree['summary'], 'tree_sources': tree['sources'],
        'primary_paired_full_tree_vs_complete_k': {str(k): compare(rows, f'complete_k{k}', 'tree_full') for k in KS},
        'sensitivity_paired_tree_topk_vs_complete_k': {str(k): compare(rows, f'complete_k{k}', f'tree_top{k}') for k in KS},
        'usage': {'deepseek_counters_scope': 'Historical cumulative deepseek-v3 usage, differenced within each run; summing all deltas reproduces final counters.',
                  'cost_unit': 'CNY (元 from the historical accounting code; JSON has no currency key).',
                  'cost_basis': 'Archived cost estimates; 1/2 yuan per million prompt/completion tokens, respectively, consistent to 1e-12. No actual bills.',
                  'totals_by_arm': budgets,
                  'unavailable': {'provider_invoice': None, 'tree_recomputed_cost_at_prefix': None, 'per_role_billing': None, 'gpu_energy_cost': None, 'end_to_end_latency': None},
                  'timing': 'Generation and evaluation timing fields are zero placeholders in both archives; no measured speedup.',
                  'comparison': 'The same number of returned strings does not match compute. All tree top-k slices reuse the entire tree run and budget; not new width-k searches.'},
    }
    return summary, data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Recompute both files and compare bytes (read-only)')
    check = parser.parse_args().check
    summary, table = build()
    outputs = {'summary.json': (json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode(),
               'per_task.csv': table}
    release_files = ['README.md', 'answers.py', 'audit.py', 'identity.py', 'tree_archive.py',
                     'test_answers.py', 'test_audit.py', 'test_identity.py', 'test_tree_archive.py']
    digests = {name: sha((HERE/name).read_bytes()) for name in release_files}
    digests.update({name: sha(blob) for name, blob in outputs.items()})
    outputs['SHA256SUMS'] = ''.join(f'{digests[name]}  {name}\n' for name in sorted(digests)).encode()
    missing = []
    for name, blob in outputs.items():
        path = HERE / name
        if path.exists():
            require(path.read_bytes() == blob, name + ' differs; refusing to overwrite')
        else:
            require(not check, name + ' missing')
            missing.append(name)
    for name in missing:
        with (HERE/name).open('xb') as f:
            f.write(outputs[name])
    for name, blob in outputs.items():
        print(sha(blob), name)


if __name__ == '__main__':
    main()
