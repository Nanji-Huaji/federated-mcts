#!/usr/bin/env python3
"""Offline archival comparison, not inference. From repo root: python3 -ISB <this file>."""
import argparse
import csv
import hashlib
import io
import json
import math
import re
from collections import Counter
from decimal import Decimal
from fractions import Fraction
from itertools import accumulate
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
IDS = tuple(range(900, 930))
CLOUD, EDGE = 'deepseek-v3', 'qwen2.5-32b-instruct'
ARMS = ('cloud', 'edge_gen', 'warm')
ROLES = {
    'cloud': 'DeepSeek-V3 generation and value evaluation',
    'edge_gen': 'Qwen2.5-32B-Instruct generation, DeepSeek-V3 value evaluation',
    'warm': 'As edge_gen, except DeepSeek-V3 generates the root expansion',
}
BASE = ('0.7_propose_n_generate_sample_1_value_n_evaluate_sample_1_method_select_greedy_'
        'n_select_sample_10_start900_end930_smg_False_sme_False_check_True_rule_False_'
        'warm_{}_last_False_idx_0')
HASHES = {
    'cloud': ('748f4bb483905ac54bab2d881eae686c8b006a225bc4742d4e3fa5903f060260',
              'c184ea7d893031b2f765620c34d6436f5a670b72f5ca9c9e65605264799290e7'),
    'edge_gen': ('45d5feb19940b629522a3f1b351a8fcd76769a7e171fd4f29240d5b9b03da7a7',
                 'd7095b1410f97865c1f1af064d9a60da84ac91bcb2be84dc84efb2781bf6eed6'),
    'warm': ('ef0e9e0dad15be81f3f6ba194e58b7b7cb60f26341ac8c54a709a089afb9d1f5',
             'b8dd6257e2b3cce6591361ad0fcc0e98da8bb1cad6960fc61600dc672792d9d8'),
}
# Context files are hashed for attribution; no project code is executed.
SOURCES = {
    'dataset': ('src/tot/data/24/24.csv', 'b9f12b3e36d987a3c714c4cef17d89a137d7c59da26532fcfb93b4821d8111b5'),
    'solver_context': ('src/tot/methods/bfs.py', '0992ed817fa15f611378ab7e10bd85b012f04efe8c1576c1285bccf4df8569ed'),
    'accounting_context': ('src/tot/models.py', '7829d6992c2ef4da90c6d212cd072565488d005966d84ea069f1d0c13afec49d'),
}
for arm in ARMS:
    directory = 'logs/game24/tot/deepseek-v3' if arm == 'cloud' else 'logs/logs_archived'
    stem = directory + '/' + BASE.format(arm == 'warm')
    SOURCES[arm + '_raw'] = stem + '.json', HASHES[arm][0]
    SOURCES[arm + '_performance'] = stem + '_performance.json', HASHES[arm][1]


def require(ok, what):
    if not ok:
        raise ValueError(what)


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def inputs():
    blobs, index = {}, {}
    for key, (path, want) in SOURCES.items():
        b = (ROOT / path).read_bytes()
        got = digest(b)
        require(got == want, f'input SHA-256 mismatch: {path}: {got}')
        blobs[key] = b
        index[key] = {'path': path, 'sha256': got, 'bytes': len(b)}
    return blobs, index


# Decode JSON once. Remove at most ONE optional terminal LF; do not strip prose,
# fix a left-list, insert missing steps or split on a literal backslash-n.
def op_lines(y):
    return y.removesuffix('\n').split('\n')


def grammar(num):
    return re.compile(rf'({num}) ([+*/-]) ({num}) = ({num}) \(left: ({num}(?: {num})*)\)')


STEP = grammar(r'-?[0-9]+(?:/[1-9][0-9]*)?')
ARCHIVE_INT_STEP = grammar(r'[0-9]+')


def verify(puzzle, y):
    """Return one reason; only 'valid' is a success. Nothing consults the old labels."""
    state = [Fraction(t) for t in puzzle.split()]
    require(len(state) == 4, f'invalid dataset puzzle {puzzle!r}')
    lines = op_lines(y)
    if len(lines) != 3:
        return 'step_count'
    for line in lines:
        m = STEP.fullmatch(line)
        if not m:
            return 'format'
        a, op, b, claimed, remaining = m.groups()
        a, b, claimed = map(Fraction, (a, b, claimed))
        for value in (a, b):  # repeated values require distinct copies
            if value not in state:
                return 'unavailable_operand'
            state.remove(value)
        if op == '+':
            result = a + b
        elif op == '-':
            result = a - b
        elif op == '*':
            result = a * b
        else:
            if b == 0:
                return 'division_by_zero'
            result = a / b
        if result != claimed:
            return 'wrong_result'
        state.append(result)
        if Counter(map(Fraction, remaining.split(' '))) != Counter(state):
            return 'wrong_left_multiset'
    return 'valid' if state == [24] else 'not_24'


def pct(a, n):
    return round(100 * a / n, 6)


def wilson(hits, n):
    z = 1.959963984540054
    p, z2 = hits / n, z * z
    d = 1 + z2 / n
    c = (p + z2 / (2*n)) / d
    h = z * math.sqrt(p*(1-p)/n + z2/(4*n*n)) / d
    return [round(100 * max(0, c-h), 6), round(100 * min(1, c+h), 6)]


def paired_bootstrap(gain, loss, n):
    """Exact empirical paired-bootstrap quantiles, NOT exact coverage for a population."""
    same = n - gain - loss
    require(n > 0 and min(same, gain, loss) >= 0, 'invalid paired counts')
    mass = Counter()
    for g in range(n+1):
        for l in range(n-g+1):
            t = n-g-l
            mass[g-l] += (math.comb(n, g) * math.comb(n-g, l)
                          * gain**g * loss**l * same**t)
    total = n**n
    require(sum(mass.values()) == total, 'bootstrap mass mismatch')
    keys = sorted(mass)
    cdf = list(zip(keys, accumulate(mass[k] for k in keys)))
    return [pct(next(k for k, c in cdf if 40*c >= q*total), n) for q in (1, 39)]


def paired(rows, a, b, field):
    maps = [{r['idx']: r[field] for r in rows if r['arm'] == arm} for arm in (a, b)]
    require(maps[0].keys() == maps[1].keys() == set(IDS), 'unmatched task IDs')
    by_cell = {(u, v): [i for i in IDS if (maps[0][i], maps[1][i]) == (u, v)]
               for u in (1, 0) for v in (1, 0)}
    both, gain, loss, neither = [len(z) for z in by_cell.values()]
    return {'a': a, 'b': b, 'n': len(IDS), 'both': both, 'a_only': gain, 'b_only': loss,
            'neither': neither, 'a_only_idx': by_cell[1, 0], 'b_only_idx': by_cell[0, 1],
            'difference_a_minus_b_pp': pct(gain-loss, len(IDS)),
            'paired_bootstrap95_pp': paired_bootstrap(gain, loss, len(IDS))}


COUNTERS = ('prompt_tokens', 'completion_tokens', 'total_tokens', 'total_calls')


def check_usage(usage, previous, models, task, tag):
    """Adjacent deltas of monotone snapshots, not a sum over cumulative rows."""
    require(set(usage) == models, f'{tag}: wrong model keys')
    for role, model in [('cloud', CLOUD), ('edge', EDGE)]:
        present = model in usage
        task[role + '_counter_present'] = int(present)
        if present:
            cur = usage[model]
            require(set(cur) == {*COUNTERS, 'cost'}, f'{tag}: usage fields')
            require(all(type(cur[f]) == int and cur[f] >= 0 for f in COUNTERS), f'{tag}: counters')
            require(cur['total_tokens'] == cur['prompt_tokens'] + cur['completion_tokens'], f'{tag}: token sum')
            compatible = Decimal(cur['prompt_tokens'] + 2*cur['completion_tokens']) / 10**6
            require(abs(compatible - cur['cost']) < Decimal('1e-12'), f'{tag}: historic cost consistency')
            prev = previous.get(model, {})
            for f in COUNTERS:
                diff = cur[f] - prev.get(f, 0)
                require(diff >= 0, f'{tag}: non-monotonic {f}')
                task[f'{role}_{f}_increment'] = diff
            task[f'{role}_old_cost_cumulative'] = str(cur['cost'])
        else:
            for f in COUNTERS:
                task[f'{role}_{f}_increment'] = 0  # recorded usage, not a hardware measurement
            task[f'{role}_old_cost_cumulative'] = ''


def run_arm(arm, raw, perf, puzzles):
    require([r['idx'] for r in raw] == list(IDS), f'{arm}: expected one consecutive record per ID')
    models = {CLOUD} if arm == 'cloud' else {CLOUD, EDGE}
    require(set(perf) == {'acc', 'avg_sum', 'sm', 'llm', *models} and
            (perf['llm'], perf['sm']) == (CLOUD, EDGE), f'{arm}: performance schema')
    out, previous, reasons, line_counts = [], {}, Counter(), Counter()
    for record in raw:
        idx, steps = record['idx'], record['steps']
        puzzle = puzzles[idx]['Puzzles']
        tag = f'{arm}/{idx}'
        require([s['step'] for s in steps] == [0, 1, 2], tag + ': missing depth')
        parents = ['']
        for s in steps:
            require(s['x'] == puzzle and s['ys'] == parents, tag + ': input or frontier mismatch')
            pool, values = s['new_ys'], s['values']
            require(len(pool) == len(values), tag + ': unaligned value scores')
            order = sorted(range(len(pool)), key=values.__getitem__, reverse=True)
            parents = s['select_new_ys']
            require(parents == [pool[k] for k in order[:10]], tag + ': not stable greedy top 10')
        ys = record['ys']
        require(ys == parents and len(ys) == 10, tag + ': final selected output mismatch')
        old = [r['r'] for r in record['infos']]
        require(len(old) == len(ys) and all(type(v) is int and v in (0, 1) for v in old), tag + ': old labels')
        # Profile the actual final pool (includes selected); keep short carried chains, don't complete them.
        all_lines = [line for y in pool for line in op_lines(y)]
        require(all(ARCHIVE_INT_STEP.fullmatch(line) for line in all_lines), tag + ': unexpected archive syntax')
        line_counts['final_pool_lines'] += len(all_lines)
        line_counts['selected_lines'] += sum(len(op_lines(y)) for y in ys)
        line_counts['selected_with_terminal_lf'] += sum(y.endswith('\n') for y in ys)
        verdict = {y: verify(puzzle, y) for y in set(pool)}
        result = [int(verdict[y] == 'valid') for y in ys]
        pool_result = [int(verdict[y] == 'valid') for y in pool]
        reasons.update(verdict[y] for y in ys)
        max_value = max(values)
        top = [ok for ok, v in zip(pool_result, values) if v == max_value]
        ranks = lambda vs: ';'.join(str(j) for j, v in enumerate(vs, 1) if v)
        task = {'idx': idx, 'dataset_rank': idx+1, 'puzzle': puzzle, 'arm': arm,
                'retained_slots': len(ys), 'strict_hits': sum(result), 'strict_any': max(result),
                'strict_first': result[0], 'strict_first_valid_rank': result.index(1)+1 if any(result) else '',
                'strict_valid_ranks': ranks(result), 'legacy_hits': sum(old), 'legacy_any': max(old),
                'legacy_first': old[0], 'legacy_valid_ranks': ranks(old),
                'label_disagreement_slots': sum(a != b for a, b in zip(result, old)),
                'label_disagreement_ranks': ranks([a != b for a, b in zip(result, old)]),
                'final_pool_slots': len(pool), 'final_pool_valid_slots': sum(pool_result),
                'max_value_slots': len(top), 'max_value_valid_slots': sum(top)}
        check_usage(record['usage_so_far'], previous, models, task, tag)
        previous = record['usage_so_far']
        out.append(task)
    require(previous == {m: perf[m] for m in models}, f'{arm}: final cumulative snapshot vs performance')
    n = len(out)
    colsum = lambda key: sum(t[key] for t in out)
    for f, old_field in [('legacy_hits', 'avg_sum'), ('legacy_any', 'acc')]:
        require(abs(Decimal(colsum(f))/n - perf[old_field]) < Decimal('1e-12'), f'{arm}: {old_field}')
    use = {}
    for role, m in [('cloud', CLOUD), ('edge', EDGE)]:
        counters = {f: colsum(f'{role}_{f}_increment') for f in COUNTERS}
        if m in models:
            require(all(perf[m][f] == counters[f] for f in COUNTERS), f'{arm}: deltas vs totals')
        use[role] = {'model': m, 'counter_present': m in models, **counters,
                     'old_cost_field': str(perf[m]['cost']) if m in models else None}
    ranks = [r['strict_first_valid_rank'] for r in out if r['strict_any']]
    return out, {
        'historical_configuration_label': ROLES[arm], 'n_tasks': n,
        'strict_any_solved': colsum('strict_any'), 'strict_any_pct': pct(colsum('strict_any'), n),
        'strict_any_wilson95_pct': wilson(colsum('strict_any'), n),
        'strict_first_solved': colsum('strict_first'), 'strict_valid_selected_slots': colsum('strict_hits'),
        'strict_selected_slot_verdict_counts': dict(sorted(reasons.items())),
        'selected_slots': colsum('retained_slots'),
        'actual_integer_line_profile': dict(line_counts),
        'legacy': {'any_solved': colsum('legacy_any'), 'first_solved': colsum('legacy_first'),
                   'positive_slots': colsum('legacy_hits'), 'raw_acc_field': str(perf['acc']),
                   'raw_avg_sum_field': str(perf['avg_sum'])},
        'strict_vs_legacy': {'discordant_slots': colsum('label_disagreement_slots'),
                             'discordant_any_tasks': sum(r['legacy_any'] != r['strict_any'] for r in out),
                             'discordant_first_tasks': sum(r['legacy_first'] != r['strict_first'] for r in out)},
        'sensitivity': {
            'solved_in_first_k_retained': {str(k): sum(a <= k for a in ranks) for k in (1, 3, 5, 10)},
            'valid_final_pool_slots': colsum('final_pool_valid_slots'), 'final_pool_slots': colsum('final_pool_slots'),
            'final_pool_covered_tasks': sum(r['final_pool_valid_slots'] > 0 for r in out),
            'covered_in_pool_not_retained_idx': [r['idx'] for r in out if r['final_pool_valid_slots'] and not r['strict_any']],
            'fixed_pool_max_value_top1': {
                'multiple_max_slots_tasks': sum(r['max_value_slots'] > 1 for r in out),
                'min_solved_over_tie_orders': sum(r['max_value_valid_slots'] == r['max_value_slots'] for r in out),
                'max_solved_over_tie_orders': sum(r['max_value_valid_slots'] > 0 for r in out),
                'expected_solved_uniform_over_max_slots': round(float(sum(
                    Fraction(r['max_value_valid_slots'], r['max_value_slots']) for r in out)), 6),
            },
        }, 'historical_usage': use,
    }


def build():
    blobs, index = inputs()
    puzzles = list(csv.DictReader(io.StringIO(blobs['dataset'].decode('utf-8'))))
    for i in IDS:
        require(puzzles[i]['Rank'] == str(i+1) and re.fullmatch(r'[0-9]+(?: [0-9]+){3}', puzzles[i]['Puzzles']),
                f'bad puzzle at zero-based row {i}')
    subset = ''.join(f"{i}\t{i+1}\t{puzzles[i]['Puzzles']}\n" for i in IDS).encode('utf-8')
    rows, arms = [], {}
    for arm in ARMS:
        raw, perf = [json.loads(blobs[arm+'_'+k], parse_float=Decimal) for k in ('raw', 'performance')]
        r, arms[arm] = run_arm(arm, raw, perf, puzzles)
        rows.extend(r)
    rows.sort(key=lambda r: (r['idx'], ARMS.index(r['arm'])))
    pair_stats = {field: [paired(rows, a, b, field) for a, b in
                          [('edge_gen', 'cloud'), ('warm', 'cloud'), ('warm', 'edge_gen')]]
                  for field in ('strict_any', 'strict_first')}
    savings = {}
    base = arms['cloud']['historical_usage']['cloud']
    for a in ('edge_gen', 'warm'):
        cur = arms[a]['historical_usage']['cloud']
        old, alt = Decimal(base['old_cost_field']), Decimal(cur['old_cost_field'])
        savings[a] = {'cloud_cost_field_reduction_6dp': f'{old-alt:.6f}',
                      'cloud_cost_field_reduction_pct': float(pct(old-alt, old)),
                      'cloud_total_tokens_reduction': base['total_tokens']-cur['total_tokens'],
                      'cloud_total_calls_reduction': base['total_calls']-cur['total_calls']}
    summary = {
        'schema': 'archive_game24.paired.v1', 'evidence_type': '离线复判既有归档；无新模型生成',
        'inputs': index, 'dataset': {'csv_rows': len(puzzles), 'idx_zero_based_inclusive': [900, 929],
            'unique_tasks': len(IDS), 'cohort_sha256': digest(subset),
            'cohort_encoding': 'UTF-8, ascending idx; idx<TAB>Rank<TAB>Puzzles<LF>, no header'},
        'archived_configuration': {
            'filename_flags': {'temperature': 0.7, 'method_generate': 'propose', 'n_generate_sample': 1,
                               'method_evaluate': 'value', 'n_evaluate_sample': 1, 'method_select': 'greedy',
                               'n_select_sample': 10, 'check': True, 'rule': False, 'last': False},
            'observed_depth': 3, 'observed_final_width': 10,
            'note': '文件名标称配置；并非逐题实际生成数或相等的推理预算。',
        },
        'protocol': {
            'primary': '按题；末层保留 ys 中任一完整合法轨迹。每题十条，按原序；零成功仍计入分母。',
            'verifier': '三步；完整匹配运算行，从数据四数起按副本取操作数；精确有理数验等式和每步 left 多重集，只接受最终单例 24。不修复，不解题。归档全是非负整数写法。',
            'provenance': '组标签取自上轮归档对应关系、性能文件模型键及文件名；与旧 solve/speculative_solve 分工一致，不是逐请求路由证明。',
            'conditioning': '同题固定历史输出，不是同预算重跑、随机抽题或盲测。首条、前 k 条、末层候选池、同分重排均不改历史账单或前层搜索。',
            'paired_interval': '同题 A−B 百分点；从 30 个配对任务等概率有放回抽 30 个，精确枚举差值 +1/−1/0 的经验多项分布，取左连续逆 CDF 的 2.5% 和 97.5% 分位数（无 Monte Carlo）。描述性，不是总体保证。',
            'marginal_interval': '百分数，95% Wilson score，z=1.959963984540054；仅作这批连续题的描述。',
            'tie_diagnostic': '固定原末层 new_ys 和 values；只选择一个最高分槽位。最差/最好同分次序按题给界，期望按并列槽位均匀选一；不是新策略实测。',
            'usage': '每模型历史累计计数与末条及 performance 核对；CSV 为相邻累计差（首题以 0 为记账基线）。无边模型键记 0 个已记录 tokens/calls，原 cost 留空。',
            'cost': 'old_cost_field 是归档 cost 十进制文本，非实付。旧代码单位元，每百万 prompt/completion 取 1/2 与两模型字段吻合；只按云字段对全云算差，不相加。',
        },
        'arms': arms, 'paired': pair_stats, 'cloud_only_savings_vs_cloud': savings,
        'exclusions': ['稿中全云 90%/2.17 与此归档不符，未归为同一运行。',
                       '无逐请求/角色账单、独立种子重复、可比 GPU 时长/能耗/边际价格；不估总成本、加速、非劣效或标准分数搜索表现。',
                       '只复核留档（可能已被历史格式检查处理）的文本，不还原原始模型响应或未展开分支。'],
    }
    buf = io.StringIO(newline='')
    writer = csv.DictWriter(buf, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return summary, buf.getvalue().encode('utf-8')


def artifacts():
    summary, table = build()
    result = {'summary.json': (json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode(),
              'per_task.csv': table}
    hashes = {v['path']: v['sha256'] for v in summary['inputs'].values()}
    for name in ('audit.py', 'test_audit.py', 'README.md', *result):
        b = result[name] if name in result else (HERE/name).read_bytes()
        hashes[(HERE/name).relative_to(ROOT).as_posix()] = digest(b)
    result['SHA256SUMS'] = ''.join(f'{v}  {k}\n' for k, v in sorted(hashes.items())).encode()
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check', action='store_true', help='recompute and compare bytes; create nothing')
    check = p.parse_args().check
    result = artifacts()
    missing = []
    for name, b in result.items():
        path = HERE/name
        if path.exists():
            require(path.read_bytes() == b, f'{name} differs; refusing to overwrite')
        else:
            require(not check, f'{name} missing in --check mode')
            missing.append(name)
    for name in missing:
        with (HERE/name).open('xb') as f:
            f.write(result[name])
    print('OK: 30 paired tasks x 3 archive arms; ' + ', '.join(result))


if __name__ == '__main__':
    main()
