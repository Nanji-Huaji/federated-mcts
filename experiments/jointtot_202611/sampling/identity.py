"""Verify the ten historical Game24 sampling archives, without running old code.

Evidence has two different strengths. Complete response vectors bind the archives to
recorded runs exactly. Inputs are reconstructed, not observed: in the pinned tree,
do_exp.py:8-20,79-90 runs the unmodified baseline recipe ten times, renaming each
result. Its shell script selects game24/naive and range(0,148) with no data override;
merged_run.py:47-71 dispatches to get_task/naive_solve. The default Game24Task
reads Puzzles from 24_extracted.csv, and naive_solve passes get_input(i) into the
sampler. Neither a runtime checkout hash nor per-request input is in the log.
No answer is parsed for numbers; no model weights, serving revision or bill are
attested. Never display the log: it includes sensitive configuration. This module
returns only typed bindings and allowlisted metadata, and never writes artifacts.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import io
import json
import os
import re
import subprocess
from pathlib import Path, PurePosixPath


_COMMIT = '3cd916a973b934bf93023f2d82d78fe0f5369b91'
_STDOUT = 'b8cf0359d5216359b51e818ea724d44c259801b687c644dcc8c54bfbd999ccb3'
_DIR = 'logs/game24/naive/deepseek-v3'
_MODEL = 'deepseek-v3'
_LOCAL = 'qwen2.5-32b-instruct'
_TASKS = 148
_BATCHES = (
    '8a450947b06fd43399f44b1bfafef423b0262e99f75f3df82be659ac3f25aa22',
    '785cf3aa3c876da203305b5c169b907765541d54ba638d5d34fa014ab8b9054a',
    'e7ced8c4dae5e5b9572b590d718e1764bd3657d33a54f709af69114dd29167eb',
    '49f9228b0c054a293f4412dea50ec14d5c2021139add117a2525b51a24ee7e4a',
    '3fd8363e8cf799034cdee0efd8ab87d6f0485c8ad820e37cce63c2e408c6388a',
    '23739658af93c618838fb30ac8db042eb77f4bdbb56d33df1feb24b58c5b369d',
    'f79399d16113f730eadb8265fb0d861dea672c759b26f90c840a9d19fde023c0',
    '4472dd945cfc2f4bc61f36343f1f00140ed6d6d2b55c33f5b11658ce95406b7d',
    '8859ae8b0e154ee688f6a8c53455c63f3e0144c0d1edd4f4d426059ae4464168',
    '2eb1cc574dabcf39bcaa1807e6cc94aef7caf0b7a735f37e1e7d0243d2c37809',
)
_EXTRACT = 'src/tot/data/24/24_extracted.csv'
_ORIGINAL = 'src/tot/data/24/24.csv'
_SOURCES = {
    'do_exp.py': 'd4f770e42c8bdcc47dd80f19928996d44d4915f640f99aec8a77d4381c17ecc2',
    'scripts/game24/eval_deepseek_pre_baseline.sh': '0affa7730ff2dcdf26fa48650f32514c5beb48aea79916fc7ddffa65ca94e481',
    'merged_run.py': 'e7bd20f0b25401abb4e071d094820e2e8fcbca85e0ec4cebe85c5467b868a544',
    'src/tot/tasks/__init__.py': 'a25ef6dbeaa5a0e9a99354c050ceee468d36018df5db0b9e2ba732e7231296c6',
    'src/tot/tasks/game24.py': '17b03460db4d2aed7e54ebdbdef2b095ec2886d973484c298232e93c149584f9',
    'src/tot/methods/bfs.py': '707631f4b7a7f5c19f0f125c4a7944bcac8aa53f75caf9c1ef95d1fcd5f77bb7',
    'src/tot/prompts/game24.py': '97333c4fa70c1af261adb998aba785e83c5085c50debe3c8336fc43d1adb44f7',
    'src/tot/models.py': 'b180ac76e5ad41ef3a7300a316c5f67ead548df64519112924ef6766d47fa377',
    _EXTRACT: 'a5904d5a779bd413546955d7d2ef4ba4ed417d05855404086f7fd6940b01b4c9',
    _ORIGINAL: 'b9f12b3e36d987a3c714c4cef17d89a137d7c59da26532fcfb93b4821d8111b5',
}


class IdentityError(ValueError):
    """Fail closed: a fixed diagnostic, never evidence text or subprocess output."""


def _require(condition, diagnostic):
    if not condition:
        raise IdentityError(diagnostic)


def _pinned(data, expected, diagnostic):
    digest = hashlib.sha256(data).hexdigest()
    _require(digest == expected, diagnostic)
    return digest


def _history(root):
    # cat-file returns uninterpreted bytes (no checkout, textconv, LFS, or filters).
    # Do not inherit Git overrides, credentials, trace destinations or transports.
    env = {'PATH': os.defpath, 'LC_ALL': 'C', 'GIT_CONFIG_NOSYSTEM': '1',
           'GIT_CONFIG_GLOBAL': os.devnull, 'GIT_NO_REPLACE_OBJECTS': '1',
           'GIT_NO_LAZY_FETCH': '1', 'GIT_ALLOW_PROTOCOL': '',
           'GIT_OPTIONAL_LOCKS': '0', 'GIT_TERMINAL_PROMPT': '0'}
    sources, hashes = {}, {}
    for path, expected in _SOURCES.items():
        p = subprocess.run(['git', 'cat-file', 'blob', f'{_COMMIT}:{path}'],
                           cwd=root, env=env, stdin=subprocess.DEVNULL,
                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=10)
        _require(p.returncode == 0, 'local historical object unavailable')
        hashes[path] = _pinned(p.stdout, expected, 'historical digest mismatch')
        if path in ('do_exp.py', _EXTRACT, _ORIGINAL):
            sources[path] = p.stdout
    return sources, hashes


def _collector_target(producer):
    # Read the constant, do not import or execute the historical collector.
    names = [n.value.value for n in ast.parse(producer).body
             if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and
                t.id == 'log_file' for t in n.targets) and isinstance(n.value, ast.Constant)]
    _require(len(names) == 1 and isinstance(names[0], str), 'collector target missing')
    target = PurePosixPath(names[0])
    _require(target.parent == PurePosixPath(_DIR), 'collector directory mismatch')
    return target


def _cohort(sources):
    def rows(path):
        return list(csv.DictReader(io.StringIO(sources[path].decode('utf-8-sig'))))

    extracted, original = rows(_EXTRACT), rows(_ORIGINAL)
    by_rank = {int(r['Rank']): r['Puzzles'] for r in original}
    expected = list(range(100, 200)) + list(range(500, 600)) + list(range(900, 1000))
    cells = [r['Puzzles'] for r in extracted]
    _require(cells == [by_rank[i] for i in expected], 'historical cohort mismatch')
    _require(len(set(cells)) == len(cells) and all(re.fullmatch(r'[0-9]+(?: [0-9]+){3}', s)
             for s in cells), 'invalid historical cohort')
    return [tuple(map(int, s.split())) for s in cells[:_TASKS]]


_CALL = re.compile(r"functools\.partial\(<function gpt at 0x[0-9a-fA-F]+>, "
                   r"model='deepseek-v3', temperature=0\.7, [^\r\n]+\)")
_SCORE = re.compile(r'([0-9]+) sum\(accs\) ([01]) cnt_avg ([0-9]+) cnt_any ([0-9]+) *')


def _recorded_runs(stdout, stem):
    lines = stdout.decode('utf-8').split('\n')
    starts = [i for i, s in enumerate(lines) if s.startswith('File name: ')]
    declared = [i for i, s in enumerate(lines) if s.startswith('Running experiment ')]
    _require(len(starts) == len(declared) == len(_BATCHES), 'run count mismatch')
    _require(sum(s.startswith('Renamed ') for s in lines) == len(declared), 'rename count mismatch')
    # Parent collector output is buffered and appears after child output. Parse the
    # explicit (experiment, destination) pairs; never use identical idx_0 stems as IDs.
    renames = []
    for ordinal, pos in enumerate(declared):
        _require(lines[pos] == f'Running experiment {ordinal + 1}/10...', 'collector order mismatch')
        m = re.fullmatch(r'Renamed (.+) to (.+)', lines[pos + 1])
        _require(m and PurePosixPath(m[1]) == stem, 'rename source mismatch')
        dest = PurePosixPath(m[2])
        _require(dest.parent == stem.parent and re.fullmatch(r'[0-9]\.json', dest.name),
                 'rename destination mismatch')
        renames.append(int(dest.stem))

    runs = []
    for start, end in zip(starts, starts[1:] + [len(lines)]):
        _require(PurePosixPath(lines[start].removeprefix('File name: ')) == stem,
                 'run output mismatch')
        _require(lines[start - 1] == f'Local backend: {_LOCAL}, Remote backend: {_MODEL}',
                 'backend label mismatch')
        block = lines[start + 1:end]
        positions = [i for i, s in enumerate(block) if s.startswith('Task ')]
        _require(len(positions) == _TASKS, 'task count mismatch')
        _require(sum(s.startswith('functools.partial(') for s in block) == _TASKS,
                 'generation binding count mismatch')
        for line, model in zip(block[:2], (_LOCAL, _MODEL)):
            _require(line.startswith('model: {') and line.endswith("'model': '" + model + "'}"),
                     'configuration label mismatch')
        rows, cumulative = [], 0
        for idx, pos in enumerate(positions):
            _require(block[pos] == f'Task {idx}', 'logged task order mismatch')
            call = pos + 1 + int(block[pos + 1].startswith('Warning:'))
            _require(_CALL.fullmatch(block[call]), 'requested model mismatch')
            payload = block[call + 1]
            _require(re.fullmatch(r'-+', block[call + 2]) and block[call + 3] == 'ys  ' + payload,
                     'logged answer copies disagree')
            # Deliberately compare response representations, not extracted expressions,
            # guesses about inputs, truncated text, or a per-task majority vote.
            m = _SCORE.fullmatch(block[call + 4])
            _require(m and int(m[1]) == idx, 'logged score index mismatch')
            flag = int(m[2])
            cumulative += flag
            _require(int(m[3]) == int(m[4]) == cumulative, 'logged score totals mismatch')
            rows.append((idx, payload, flag))
        totals = [s.removeprefix('Token consumption: ') for s in block if s.startswith('Token consumption: ')]
        _require(len(totals) == 1, 'run accounting summary missing')
        runs.append((rows, ast.literal_eval(totals[0])))
    return runs, renames


def _batch(data):
    records = json.loads(data)
    _require(type(records) is list and len(records) == _TASKS, 'batch count mismatch')
    rows = []
    for idx, r in enumerate(records):
        _require(type(r['idx']) is int and r['idx'] == idx, 'batch index mismatch')
        ys, infos, usage = r['ys'], r['infos'], r['usage_so_far']
        _require(type(ys) is list and len(ys) == 1 and type(ys[0]) is str and
                 type(infos) is list and len(infos) == 1, 'invalid batch answer record')
        flag = infos[0]['r']
        _require(type(flag) is int and flag in (0, 1), 'invalid archived score')
        _require(set(usage) == {_MODEL} and type(usage[_MODEL]['total_calls']) is int and
                 usage[_MODEL]['total_calls'] == idx + 1, 'per-task model accounting mismatch')
        rows.append((idx, repr(ys), flag))
    return rows, records[-1]['usage_so_far']


def _load(root):
    stdout = (root / 'exp_log.log').read_bytes()
    stdout_digest = _pinned(stdout, _STDOUT, 'stdout digest mismatch')
    sources, history_hashes = _history(root)
    puzzles = _cohort(sources)
    runs, renames = _recorded_runs(stdout, _collector_target(sources['do_exp.py']))
    batches, batch_hashes = [], {}
    for digit, expected in enumerate(_BATCHES):
        data = (root / _DIR / f'{digit}.json').read_bytes()
        batch_hashes[str(digit)] = _pinned(data, expected, 'batch digest mismatch')
        batches.append(_batch(data))

    run_to_batch = []
    for (rows, usage), renamed in zip(runs, renames):
        # Require uniqueness on the full ordered (idx, exact response, evaluator flag)
        # vector ALONE, then cross-check independent rename and accounting evidence.
        matches = [b for b, (other, _) in enumerate(batches) if other == rows]
        _require(len(matches) == 1, 'no unique full-run response match')
        b = matches[0]
        _require(renamed == b and usage == batches[b][1], 'run corroboration mismatch')
        run_to_batch.append(b)
    _require(sorted(run_to_batch) == list(range(len(_BATCHES))), 'non-bijective run match')
    mapping = {(b, idx): puzzles[idx] for b in run_to_batch for idx in range(_TASKS)}
    cohort_bytes = ''.join(' '.join(map(str, p)) + '\n' for p in puzzles).encode('ascii')
    return mapping, {
        'stdout_sha256': stdout_digest, 'batch_sha256': batch_hashes,
        'historical_commit': _COMMIT, 'historical_source_sha256': history_hashes,
        'cohort_sha256': hashlib.sha256(cohort_bytes).hexdigest(),
        'original_rank_ranges_inclusive': [[100, 199], [500, 547]],
        'run_to_batch': run_to_batch, 'rows_matched': len(mapping),
        'identity_basis': 'historical_recipe_reconstruction',
        'run_binding': 'unique_ordered_exact_responses_indices_and_flags_plus_rename_and_usage',
        'prompt_inputs_logged': False, 'execution_commit_logged': False,
        'requested_model': _MODEL, 'per_task_model_bindings': sum(len(rows) for rows, _ in runs),
        'configured_local_model': _LOCAL, 'local_generation_bindings': 0,
        'model_evidence': 'configured_remote_and_generation_callable_and_usage_label',
        'server_model_attested': False,
    }


def load_identity(root: Path | str) -> tuple[dict[tuple[int, int], tuple[int, ...]], dict]:
    """Return {(batch digit, local idx): ordered puzzle ints}, safe provenance metadata.

    Requires this offline archive in the repository root and local pinned Git objects.
    Missing, changed, ambiguous or contradictory evidence raises IdentityError; no
    fallback to today's CSV, raw responses, filenames, or idx-only joins.
    """
    try:
        return _load(Path(root))
    except IdentityError as error:
        diagnostic = str(error)  # Only our fixed diagnostic codes reach this branch.
    except (OSError, ValueError, SyntaxError, KeyError, IndexError, TypeError,
            csv.Error, subprocess.SubprocessError):
        diagnostic = 'required evidence absent or malformed'
    # Raise *outside* both handlers. The public exception must not keep an inner
    # traceback (including its frame locals) or an exception context with raw bytes.
    raise IdentityError(diagnostic)
