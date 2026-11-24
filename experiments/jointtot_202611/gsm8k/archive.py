"""Local-only evidence access, source alignment, trace and accounting checks."""

import ast
import hashlib
import json
import re
import subprocess
from collections import Counter
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
HIST = "3cd916a973b934bf93023f2d82d78fe0f5369b91"
CHECKOUT = "db346fc937ec4e611e63f483908fc00178acdb94"
ARMS = ("tot", "speculative_solve")
N = 50
DATA = "src/tot/data/gsm8k/gsm8k.jsonl"
DATA2 = "src/federated_mcts/data/gsm8k/gsm8k.jsonl"
OLD_TASK = "src/tot/tasks/gsm8k.py"
CURRENT_TASK = "src/federated_mcts/tasks/gsm8k.py"
PROMPT = "src/tot/prompts/gsm8k.py"
STEM = ("0.7_propose_n_generate_sample_1_value_n_evaluate_sample_2_method_select_greedy_"
        "n_select_sample_3_start0_end50_smg_False_sme_False_check_False_rule_False_"
        "warm_False_last_False_idx_0")
STDOUT = {"tot": "gsm8k_baseline_tot_log.log", "speculative_solve": "gsm8k_ours.log"}
SPLIT = "data/splits/gsm8k.json"


def require(condition, msg):
    if not condition:
        raise ValueError(msg)


def digest(b):
    return hashlib.sha256(b).hexdigest()


class Sources:
    """Hash raw bytes; never emit stdout lines (they contain client configuration)."""
    def __init__(self):
        self.manifest = {}

    def get(self, path, rev=None):
        if rev is None:
            key = path
            data = (ROOT / path).read_bytes()
        else:
            key = rev + ":" + path
            data = subprocess.check_output(["git", "--no-optional-locks", "-C", str(ROOT), "cat-file", "blob", key])
        self.manifest[key] = {"bytes": len(data), "sha256": digest(data)}
        return data


def run_path(arm, stem=STEM):
    return f"logs/gsm8k/{arm}/deepseek-v3/{stem}"


def load_sources(src):
    raw = src.get(DATA)
    require(raw == src.get(DATA2) == src.get(DATA, HIST), "reference snapshots differ")
    data = [json.loads(line) for line in raw.splitlines()]
    split = json.loads(src.get(SPLIT))
    require(len({row['question'] for row in data[:N]}) == N, "duplicate questions")
    rr, perf, consoles = {}, {}, {}
    for arm in ARMS:
        base = run_path(arm)
        rr[arm] = json.loads(src.get(base + ".json"), parse_float=Decimal)
        perf[arm] = json.loads(src.get(base + "_performance.json"), parse_float=Decimal)
        consoles[arm] = src.get(STDOUT[arm]).decode("utf-8")
        validate_records(rr[arm], data)
    return data, split, rr, perf, consoles


def validate_records(rr, data):
    # Validate order as well as the set: never align by list position alone or take differences across resets.
    require([r['idx'] for r in rr] == list(range(N)), "invalid index set/order")
    for r in rr:
        i = r['idx']
        require(len(r['ys']) == len(r['infos']) > 0, f"empty/misaligned final candidates idx {i}")
        require(all(isinstance(y, str) and e['r'] in (0, 1) for y, e in zip(r['ys'], r['infos'])), f"bad finals idx {i}")
        require([s['step'] for s in r['steps']] == list(range(4)), f"malformed step chain idx {i}")
        parent = [""]
        for st in r['steps']:
            require(st['x'] == data[i]['question'], f"question mismatch idx {i} step {st['step']}")
            require(st['ys'] == parent, f"broken step link idx {i}")
            require(len(st['new_ys']) == len(st['values']), f"score count idx {i}")
            order = sorted(range(len(st['values'])), key=lambda j: -st['values'][j])[:3]
            parent = [st['new_ys'][j] for j in order]
            require(parent == st['select_new_ys'], f"greedy selection not reconstructed idx {i}")
        require(parent == r['ys'], f"final ys != last selection idx {i}")


def make_scorer(src, origin, data):
    """Use only these two functions from hash-pinned task source. No task/model imports or instantiation."""
    top = ast.parse(src, filename=origin).body
    fn = next(n for n in top if isinstance(n, ast.FunctionDef) and n.name == "extract_answer")
    cls = next(n for n in top if isinstance(n, ast.ClassDef) and n.name == "GSM8KTask")
    test = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "test_output")
    env = {"re": re, "print": lambda *a, **k: None}
    exec(compile(ast.Module([fn, test], []), origin, "exec"), env)
    records = SimpleNamespace(data=data)
    def score(i, y):
        v = env["test_output"](records, i, y)
        return (v[0] if isinstance(v, tuple) else v)["r"]
    return env['extract_answer'], score


def stdout_check(text, arm, rr, old_extract, data):
    require("File name: ./" + run_path(arm) + ".json\n" in text, "stdout filename does not match")
    require(list(map(int, re.findall(r'^Task (\d+)$', text, re.M))) == list(range(N)), "stdout task ids do not match")
    expected, total, any_ct = [], 0, 0
    for row in rr:
        hits = sum(z['r'] for z in row['infos'])
        total += hits
        any_ct += bool(hits)
        expected.append((row['idx'], hits, total, any_ct))
    found = re.findall(r'^(\d+) sum\(accs\) (\d+) cnt_avg (\d+) cnt_any (\d+)', text, re.M)
    require([tuple(map(int, r)) for r in found] == expected, "stdout counts do not match")
    i = None
    golds, role_counts = [], Counter()
    for line in text.splitlines():
        m = re.fullmatch(r'Task (\d+)', line)
        if m:
            i = int(m[1])
        m = re.fullmatch(r"Failed to extract answers: pred='[^']*', true='([-0-9.,]+)'", line)
        if m:
            require(m[1] == old_extract(data[i]['answer']), "stdout terminal true value differs")
            golds.append({'idx': i, 'reported_numeric_true': m[1]})
        # Emit only explicit model aliases from matched step messages, never the client partials/credentials.
        if re.match(r'^Step \d+ of 4 in Task \d+, solve_client:', line):
            aliases = re.findall(r"model='(qwen2\.5-32b-instruct|deepseek-v3)'", line)
            require(len(aliases) == 2, "unexpected model aliases")
            role_counts[' / '.join(aliases)] += 1
    clients = [ast.literal_eval(line.partition('model: ')[2]) for line in text.splitlines() if line.startswith('model: {')]
    bases = {c['client_name']: c.get('api_base') for c in clients}
    require(bases.get('local_client') and bases.get('remote_client'), 'missing client configuration evidence')
    return {'filename_and_all_50_task_and_score_counters_match': True,
            'same_api_base_reported_for_local_and_remote_client_aliases': bases['local_client'] == bases['remote_client'],
            'terminal_true_scalar_fragments_only': golds, 'logged_step_solve_eval_alias_pairs': dict(role_counts)}


KEYS = ("prompt_tokens", "completion_tokens", "total_tokens", "total_calls", "cost")


def usage(rr, sidecar):
    """Resolve the archive's cumulative counters (original order), not per-row totals."""
    last_raw = rr[-1]['usage_so_far']
    require(last_raw == {k: v for k, v in sidecar.items() if k not in ('acc', 'avg_sum', 'sm', 'llm')}, "sidecar != last usage snapshot")
    prev = {}
    rows = []
    for row in rr:
        snap = {}
        for model, raw in row['usage_so_far'].items():
            require(set(raw) == set(KEYS), "unknown resource counters")
            require(raw['total_tokens'] == raw['completion_tokens'] + raw['prompt_tokens'], "total tokens != prompt + completion")
            # Archive used float arithmetic; report fee at 1e-6 yuan precision and cross-check the token price.
            counters = dict(raw, cost=Decimal(raw['cost']).quantize(Decimal('.000001')))
            require(counters['cost'] == Decimal(raw['prompt_tokens'] + 2 * raw['completion_tokens']) / 1_000_000,
                    "reported cost differs from historical fallback price")
            snap[model] = counters
        require(set(prev) <= set(snap), "a model disappeared from cumulative counters")
        delta = {m: {k: v - prev.get(m, {}).get(k, 0) for k, v in s.items()} for m, s in snap.items()}
        require(all(v >= 0 for s in delta.values() for v in s.values()), "nonmonotone usage")
        rows.append({'idx': row['idx'], 'by_model_delta': delta, 'all_models_delta': total_models(delta)})
        prev = snap
    combined = total_models(prev)
    require(combined == {k: sum(r['all_models_delta'][k] for r in rows) for k in KEYS}, "resource delta sum mismatch")
    times = [r['time_consumption'] for r in rr]
    require(all(t == times[-1] for t in times), "unexpected time semantics; re-audit before comparison")
    return rows, {'by_model': prev, 'all_models': combined, 'last_matches_performance': True,
                  'time_fields_identical_across_all_records': True, 'repeated_run_time_fields_s': times[-1],
                  'per_item_times_usable': False}


def total_models(d):
    return {k: sum(m[k] for m in d.values()) for k in KEYS}


def example_evidence(src, data, rr):
    """Three manually identified near-identical labelled examples, not an exhaustive paraphrase detector."""
    tree = ast.parse(src.get(PROMPT, HIST)).body
    prompts = {n.targets[0].id: ast.literal_eval(n.value) for n in tree if isinstance(n, ast.Assign)}
    examples = []
    for idx, tag, prefix in [(0, 'cot_prompt', "Janet's ducks"), (2, 'cot_prompt', 'Josh decides'),
                             (1, 'value_last_step_prompt', 'A robe takes')]:
        block = next(b for b in re.split(r'^Problem: ', prompts[tag], flags=re.M) if b.startswith(prefix))
        exemplar_q = block.splitlines()[0]
        numbers = re.findall(r'####\s*([\d,]+)', block)
        gold = data[idx]['answer'].rsplit('####', 1)[-1].strip()
        require(numbers == [gold], "historical demonstration no longer matches the selected reference")
        # On the historical unchecked proposal path, a parent with #### triggers a CoT prompt;
        # on the value path, a new candidate with #### triggers the final-step judge (possibly cached).
        key = 'ys' if tag == 'cot_prompt' else 'new_ys'
        examples.append({'idx': idx, 'template': tag, 'example_question': exemplar_q, 'example_answer': gold,
                         'byte_identical_question': exemplar_q == data[idx]['question'],
                         'matching_template_trigger_steps_if_pinned_code_by_arm': {
                             arm: [s['step'] for s in rr[arm][idx]['steps'] if any('####' in y for y in s[key])] for arm in ARMS}})
    return examples
