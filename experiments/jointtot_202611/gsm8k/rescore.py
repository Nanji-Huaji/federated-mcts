"""Rebuild this fixed retrospective pair comparison. No model/task runtime imports."""

import argparse
import json
import sys
from collections import Counter
from decimal import Decimal
from fractions import Fraction

sys.dont_write_bytecode = True
from archive import (ARMS, CHECKOUT, CURRENT_TASK, HERE, HIST, KEYS, N, OLD_TASK, STEM, Sources,
                     digest, example_evidence, load_sources, make_scorer, require, run_path, stdout_check, usage)
from extractor import extract, reference
from paired_stats import paired

FREEZE = {'PROTOCOL.md': '9acffdc4c3780e991e07f8b22edb9786910113ce85702c43295927ccbef4c143',
          'extractor.py': 'ca7ae95cf643a16b3e45c5d349cbd026b83ce952a1b5f0bd4deed8b2fdadf335',
          'test_extractor.py': '01805d9e2bf072374e406c63975c8409de2b4252b5fa6aea7f505e845c4dbe28'}
PANELS = ('archived', 'frozen', 'current_checkout')
LOCAL_TEXT = frozenset({'audit.json', 'candidates.jsonl', 'outcomes.jsonl', 'review.jsonl'})


def packed(obj, *, lines=False):
    # Decimal values (estimated currency and clock counters) are decimal strings to avoid float drift.
    def conv(x):
        if isinstance(x, Decimal):
            return format(x, 'f')
        raise TypeError(str(type(x)))
    kw = dict(ensure_ascii=False, sort_keys=True, default=conv, allow_nan=False)
    if lines:
        return ''.join(json.dumps(row, separators=(',', ':'), **kw) + '\n' for row in obj).encode()
    return (json.dumps(obj, indent=2, **kw) + '\n').encode()


def candidate_rows(i, arm, row, gold, legacy, current):
    seen = set()
    results, review = [], []
    for slot, (y, info) in enumerate(zip(row['ys'], row['infos'])):
        key = f'{i}/{arm}/{slot}'
        pred = extract(y)  # candidate only; gold enters at the equality below, never at extraction
        old_pred, now_pred = legacy[0](y), current[0](y)
        old_replay = legacy[1](i, y) if y not in seen else 0  # archived runner's duplicate-text suppression
        seen.add(y)
        require(old_replay == info['r'], f'{key}: historical scorer/runner replay != actual logged label')
        bits = {'archived': info['r'], 'frozen': int(pred['value'] == gold), 'current_checkout': current[1](i, y)}
        result = {'id': key, 'idx': i, 'arm': arm, 'slot0': slot, 'text': y,
                  'text_sha256': digest(y.encode()), 'frozen_parse': pred, 'gold_value': gold,
                  'legacy_extracted': old_pred, 'legacy_replay_r': old_replay, 'checkout_extracted': now_pred, 'r': bits}
        results.append(result)
        tags = []
        if pred['value'] is None:
            tags.append(pred['status'])
        if bits['frozen'] != bits['archived']:
            tags.append('old_vs_frozen')
            if pred['status'] == 'no_answer' and '####' not in y:
                tags.append('archived_last_number_without_declared_answer')
            if pred['status'] == 'malformed_declaration':
                tags.append('malformed_formal_marker_blocks_credit')
            if bits['frozen'] and ',' in y and old_pred != pred['value']:
                tags.append('see_grouped_number_in_marker_vs_legacy_extracted')
        if bits['current_checkout'] != bits['frozen']:
            tags.append('current_vs_frozen')
        if bits['current_checkout'] != bits['archived']:
            tags.append('current_vs_old')
        if tags:
            review.append({'id': key, 'idx': i, 'arm': arm, 'slot0': slot, 'tags': tags, 'r': bits,
                           'gold': gold, 'frozen_parse': pred, 'legacy_extracted': old_pred,
                           'text_tail': y.rstrip()[-190:]})
    return results, review


def item_arm(candidates):
    bits = {p: [c['r'][p] for c in candidates] for p in PANELS}
    return {'saved_candidates': len(candidates), 'unparseable': sum(c['frozen_parse']['value'] is None for c in candidates),
            'any': {p: int(any(b)) for p, b in bits.items()}, 'first': {p: b[0] for p, b in bits.items()},
            'number_correct': {p: sum(b) for p, b in bits.items()}}


def arm_summary(arm, items, candidates, perf):
    rows = [item['methods'][arm] for item in items]
    slots = [c for c in candidates if c['arm'] == arm]
    any_old = sum(r['any']['archived'] for r in rows)
    n_old = sum(c['r']['archived'] for c in slots)
    require(Decimal(any_old) / N == perf['acc'] and Decimal(n_old) / N == perf['avg_sum'], 'sidecar does not match actual archived labels')
    return {'saved_candidates': len(slots), 'histogram_beam_size': dict(Counter(r['saved_candidates'] for r in rows)),
            'number_correct_candidates': {p: sum(c['r'][p] for c in slots) for p in PANELS},
            'mean_within_item_fraction_correct': {p: float(sum(Fraction(r['number_correct'][p], r['saved_candidates']) for r in rows) / N) for p in PANELS},
            'primary_any_correct_questions': {p: sum(r['any'][p] for r in rows) for p in PANELS},
            'first_saved_slot_correct_questions': {p: sum(r['first'][p] for r in rows) for p in PANELS},
            'frozen_parse_status_counts': {s: sum(c['frozen_parse']['status'] == s for c in slots) for s in ('formal', 'answer_phrase', 'terminal_scalar', 'no_answer', 'malformed_declaration', 'conflict')},
            'all_slots_unparseable_indices': [i for i, r in enumerate(rows) if r['unparseable'] == r['saved_candidates']],
            'old0_frozen1_candidate_count': sum(c['r']['frozen'] > c['r']['archived'] for c in slots),
            'old1_frozen0_candidate_count': sum(c['r']['frozen'] < c['r']['archived'] for c in slots),
            'old0_frozen1_item_indices': [i for i, r in enumerate(rows) if r['any']['frozen'] > r['any']['archived']],
            'old1_frozen0_item_indices': [i for i, r in enumerate(rows) if r['any']['frozen'] < r['any']['archived']],
            'archived_sidecar': {k: perf[k] for k in ('acc', 'avg_sum', 'sm', 'llm')},
            'archived_labels_reproduced_by_historical_code': len(slots)}


def audit(src, data, rr, split, consoles, legacy):
    a, b = (rr[arm] for arm in ARMS)
    pilot = json.loads(src.get(run_path('tot', STEM.replace('end50_', 'end1_')) + '.json'))[0]
    split_idxs = {k: sorted(set(v) & set(range(N))) for k, v in split.items() if isinstance(v, list)}
    for path in ['merged_run.py', 'src/tot/methods/bfs.py', 'src/tot/models.py']:
        src.get(path, HIST)
    return {
        'archive_compatible_git_revision_not_runtime_attestation': HIST,
        'current_comparator_task_blob_matches_revision': CHECKOUT,
        'case_identity': {'index_set_each': '0..49 exactly once; original order', 'distinct_questions': N,
            'step_x_matches_source_exactly_by_arm': {arm: sum(len(t['steps']) for t in rr[arm]) for arm in ARMS},
            'source_row_count': len(data), 'entire_historical_and_both_worktree_jsonls_byte_identical': True,
            'full_answer_string_matches_between_available_source_copies': N,
            'full_answer_string_independently_logged_by_each_run': False,
            'interpretation': 'Both traces match exactly the source questions. Same archived/reference snapshot and complete gold assumed by index; no independent full-gold copy or hash per run. Some stdout true-scalar fragments and exact archived-score replay corroborate, but cannot prove full runtime reference text identity.'},
        'gold_and_decisions': {'selected_top3_stable_descending_logged_value_matches_all_steps_by_arm': {arm: 4*N for arm in ARMS},
            'gold_used_in_this_rescore': 'Post hoc equality only, never extraction or branch selection.',
            'historical_code': 'At 3cd916a, merged_run.py:46-52,66-77 calls solve before final test_output_modify; src/tot/tasks/gsm8k.py:66-98 reads the row answer there. src/tot/methods/bfs.py:409-466,468-564 uses 4 fixed steps, model values, and greedy selection; unchecked proposals dispatch at 153-181; get_values at 57-86. No per-item gold comparison found in that generation, evaluation, routing, pruning or stopping path. Early stopping of judge token generation is not solution stopping.',
            'limitation': 'Consistent source + all recorded choices reproducible without gold is not a runtime audit trail for prompts/value origins. No proof that arbitrary unlogged code/requests never saw gold. Do not attribute current GSM8KTask.is_success_state (uses gold) to these runs.',
            'near_identical_solved_demonstrations_in_reachable_historical_templates': example_evidence(src, data, rr),
            'template_caveat': 'Manual lower bound, not exhaustive near-duplicate search or proof of actual request bodies. Matching exemplar answers make an absolute blind-evaluation claim untenable.'},
        'independence': {'current_repository_split_indices': split_idxs,
            'current_repository_split_counts': {k: len(v) for k,v in split_idxs.items()},
            'ordered_run_start0_end50_pairs_per_arm': 1, 'archived_inference_idx_in_filename': 0,
            'index0_also_in_one_item_tot_archive': pilot['idx'] == 0 and pilot['steps'][0]['x'] == a[0]['steps'][0]['x'],
            'one_item_archive_same_final_beam_as_50_item_tot_idx0': pilot['ys'] == a[0]['ys'],
            'paired_items_with_any_exact_same_final_text_across_arms': sum(bool(set(t['ys']) & set(s['ys'])) for t, s in zip(a, b)),
            'paired_items_with_exact_same_steps_across_arms': sum(t['steps'] == s['steps'] for t, s in zip(a,b)),
            'within_arm_duplicate_final_slot_counts': {arm: sum(len(t['ys']) - len(set(t['ys'])) for t in rr[arm]) for arm in ARMS},
            'conclusion': 'Not independent replications: fixed prefix, previously seen item/examples, shared model/judge family and task, correlated beams. No independent seeds, request or provider version IDs, commitment to blind holdout, or repeat paired runs established. Distinct texts are not proof of statistical independence. Current train/val/test names are repository splits, not evidence of model training or of which split was used in 2025.'},
        'stdout_checks': {arm: stdout_check(consoles[arm], arm, rr[arm], legacy[0], data) for arm in ARMS},
        'resource_limitations': 'usage_so_far monotone by model; difference from previous record (initial zero assumed), sidecar matches final. Historical fallback rate: 1 yuan/M input, 2 yuan/M output for both aliases (src/tot/models.py:189-226). total_calls increments once per accounted API response (src/tot/models.py:110-143); retry attempts not separately itemized. Both client aliases list the same API base in stdout: do not infer on-device compute from a local_client label. No GPU hours/invoices or complete end-to-end timings. Runner kept one latency_dict by reference; tot did not instrument these components (all zero), speculative_solve repeats the final accumulated pair in every record. Do not sum repeated times or compare latency.',
    }


def calculate():
    for name, pinned in FREEZE.items():
        require(digest((HERE/name).read_bytes()) == pinned, f'locked pre-outcome policy changed: {name}')
    src = Sources()
    data, split, rr, perf, consoles = load_sources(src)
    legacy = make_scorer(src.get(OLD_TASK, HIST), OLD_TASK, data)
    current_src = src.get(CURRENT_TASK)
    require(current_src == src.get(CURRENT_TASK, CHECKOUT), 'checkout task changed vs pinned comparison')
    current = make_scorer(current_src, CURRENT_TASK, data)
    candidates, review, items = [], [], []
    for i in range(N):
        q, ref = data[i]['question'], data[i]['answer']
        gold = reference(ref)
        parts = [k for k, v in split.items() if isinstance(v, list) and i in v]
        require(len(parts) == 1, f'item {i} not in exactly one local split')
        item = {'idx': i, 'question': q, 'reference_answer': ref, 'reference_value': gold,
                'question_sha256': digest(q.encode()), 'reference_answer_sha256': digest(ref.encode()),
                'current_repository_split': parts[0], 'methods': {}}
        for arm in ARMS:
            slotrows, flagged = candidate_rows(i, arm, rr[arm][i], gold, legacy, current)
            candidates.extend(slotrows)
            review.extend(flagged)
            item['methods'][arm] = item_arm(slotrows)
        item['spec_minus_tot_any'] = {p: item['methods'][ARMS[1]]['any'][p] - item['methods'][ARMS[0]]['any'][p] for p in PANELS}
        items.append(item)
    paired_any = {p: paired(*([r['methods'][a]['any'][p] for r in items] for a in ARMS)) for p in PANELS}
    audit_result = audit(src, data, rr, split, consoles, legacy)
    raw_resource, resource_totals = {}, {}
    for a in ARMS:
        raw_resource[a], resource_totals[a] = usage(rr[a], perf[a])
    def diffs(d):
        return {k: d[ARMS[1]][k] - d[ARMS[0]][k] for k in KEYS}
    resources = []
    for i in range(N):
        rows = {a: raw_resource[a][i] for a in ARMS}
        resources.append({'idx': i, 'methods': {a: r['by_model_delta'] for a, r in rows.items()},
                          'all_models': {a: r['all_models_delta'] for a, r in rows.items()},
                          'spec_minus_tot_all_models': diffs({a: r['all_models_delta'] for a, r in rows.items()})})
    d = {a: val['all_models'] for a, val in resource_totals.items()}
    delta = diffs(d)
    reductions = {k: round(-100 * float(delta[k] / d['tot'][k]), 4) for k in KEYS}
    summary = {
        'task': 'GSM8K retrospective rescore of 50 paired archived items; no new inference',
        'primary': 'frozen.any: >=1 of the saved final ys (not best intermediate branch); no-answer/conflict included as incorrect',
        'reference_source': 'Full-text pair identity at the dataset level, not attested per-run runtime gold snapshots. See audit.json.',
        'protocol_v1_sha256': FREEZE, 'interval': '95% paired question percentile bootstrap, exact enumeration under empirical exchangeability; conditional on these runs, not over seeds or a random holdout',
        'paired_any': paired_any,
        'by_arm': {a: arm_summary(a, items, candidates, perf[a]) for a in ARMS},
        'resources_historical': {'totals': resource_totals, 'all_models_spec_minus_tot': delta,
            'all_models_reduction_percent_relative_to_tot': reductions, 'unit_of_cost': 'estimated yuan at repository fallback rates, not a bill',
            'timing': 'no comparable wall-time / item-latency measurement'},
        'independence': 'No credible independent blind replication. At the compatible historical commit, no dynamic per-item gold lookup found in the solver; some reachable prompt examples include answers to the evaluated tasks. One fixed run per arm; provenance and request-level telemetry incomplete.',
        'decision': 'Archival scoring and the frozen exact rule rank the arms in opposite directions on these same texts. The fixed-run paired interval includes zero. The speculative run logged fewer tokens/calls at a lower estimated fee, but the data do not establish a quality win, equivalent accuracy, total deployment cost advantage, or a latency advantage.',
        'files': {'items': 'outcomes.jsonl', 'all_candidates': 'candidates.jsonl', 'parse_failures_and_judgement_differences': 'review.jsonl', 'per_pair_resources': 'resources.jsonl', 'evidence': 'audit.json', 'source_hashes': 'source_manifest.json'},
    }
    outputs = {name: packed(obj, lines=True) for name, obj in [('outcomes.jsonl', items), ('candidates.jsonl', candidates), ('review.jsonl', review), ('resources.jsonl', resources)]}
    outputs.update({'summary.json': packed(summary), 'audit.json': packed(audit_result), 'source_manifest.json': packed(src.manifest)})
    local = ['README.md', 'PROTOCOL.md', 'extractor.py', 'paired_stats.py', 'archive.py', 'rescore.py', 'test_extractor.py', 'test_rescore.py']
    hashed = {**outputs, **{name: (HERE/name).read_bytes() for name in local}}
    outputs['SHA256SUMS'] = ''.join(f'{digest(hashed[name])}  {name}\n' for name in sorted(hashed)).encode()
    return outputs, summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check', action='store_true', help='recompute in memory and compare output bytes; do not write')
    args = p.parse_args()
    outputs, summary = calculate()
    if args.check:
        for name, body in outputs.items():
            # SHA256SUMS is also compared here. It covers every recomputed text
            # report even when no plaintext copy is distributed.
            if name in LOCAL_TEXT and not (HERE/name).exists():
                continue
            require((HERE/name).read_bytes() == body, f'recomputed bytes differ: {name}')
    else:
        for name, body in outputs.items():
            (HERE/name).write_bytes(body)
    print(('Verified' if args.check else 'Wrote'), len(outputs), 'deterministic files in', HERE.name)
    for p, v in summary['paired_any'].items():
        print(f"{p}: ToT={v['tot_correct']}/{N}, speculative_solve={v['speculative_solve_correct']}/{N}, "
              f"delta={v['spec_minus_tot_pp']:+g} pp, 95% {v['paired_percentile_95_pp']} pp")


if __name__ == '__main__':
    main()
