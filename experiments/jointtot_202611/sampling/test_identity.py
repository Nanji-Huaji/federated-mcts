"""Offline acceptance tests for the archive identity interface; no fixtures copied."""

import hashlib
import json
import sys
import traceback
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import identity
from identity import IdentityError, load_identity


LOG = ROOT / 'exp_log.log'
BATCH = ROOT / 'logs/game24/naive/deepseek-v3'


@contextmanager
def alternate_snapshot(changes):
    # Simulate a second frozen archive entirely in memory. Re-pin ONLY the changed
    # artifacts so subsequent failures test the evidence chain, not just checksums.
    # No pin override is exposed to callers, and no raw log or answer is printed.
    original_read = Path.read_bytes

    def read(path):
        data = original_read(path)
        return changes[path](data) if path in changes else data

    batches = list(identity._BATCHES)
    stdout = identity._STDOUT
    for path in changes:
        digest = hashlib.sha256(read(path)).hexdigest()
        if path == LOG:
            stdout = digest
        else:
            batches[int(path.stem)] = digest
    with patch.object(Path, 'read_bytes', read), patch.object(identity, '_STDOUT', stdout), \
            patch.object(identity, '_BATCHES', tuple(batches)):
        yield


class IdentityTests(unittest.TestCase):
    def test_every_batch_binds_to_the_independent_historical_cohort(self):
        mapping, meta = load_identity(ROOT)
        self.assertEqual(len(mapping), 1480)
        for batch in range(10):
            rows = [mapping[batch, idx] for idx in range(148)]
            self.assertTrue(all(type(row) is tuple and len(row) == 4 and
                                all(type(n) is int for n in row) for row in rows))
            canon = ''.join(' '.join(map(str, row)) + '\n' for row in rows).encode('ascii')
            # Independent reference: original CSV ranks 100..199 then 500..547;
            # not the first 148 original CSV rows or numbers parsed from answers.
            self.assertEqual(hashlib.sha256(canon).hexdigest(),
                             '1dd303a52cd3486f75b0ba4f29d4de16164437a1bc4611d6768041fcd5cb3857')
        self.assertEqual(meta['run_to_batch'], list(range(10)))
        self.assertEqual(meta['rows_matched'], 1480)
        self.assertEqual(meta['requested_model'], 'deepseek-v3')
        self.assertEqual(meta['configured_local_model'], 'qwen2.5-32b-instruct')
        self.assertEqual(meta['local_generation_bindings'], 0)
        self.assertIs(meta['prompt_inputs_logged'], False)
        self.assertIs(meta['execution_commit_logged'], False)
        self.assertIs(meta['server_model_attested'], False)

    def test_changed_stdout_exposes_no_evidence_even_in_traceback_locals(self):
        # Synthetic bytes only; no historical diagnostic ever becomes a test fixture.
        marker = 'sentinel-that-must-not-appear'
        with patch.object(Path, 'read_bytes', return_value=marker.encode()):
            try:
                load_identity(ROOT)
            except IdentityError as error:
                self.assertEqual(str(error), 'stdout digest mismatch')
                self.assertIsNone(error.__context__)
                stack = traceback.TracebackException.from_exception(error, capture_locals=True).stack
                internal = [v for f in stack if f.filename == identity.__file__
                            for v in (f.locals or {}).values()]
                self.assertFalse(any(marker in v for v in internal))
            else:
                self.fail('modified stdout must be rejected')

    def test_unaccounted_extra_generation_binding_cannot_share_a_task_block(self):
        def extra_binding(data):
            records = data.split(b'\n')
            pos = next(i for i, r in enumerate(records) if r.startswith(b'functools.partial('))
            records.insert(pos + 6, records[pos])
            return b'\n'.join(records)

        with alternate_snapshot({LOG: extra_binding}), self.assertRaisesRegex(
                IdentityError, '^generation binding count mismatch$'):
            load_identity(ROOT)

    def test_even_a_single_trailing_space_in_one_late_answer_breaks_the_match(self):
        def changed(data):
            rows = json.loads(data)
            rows[-1]['ys'][0] += ' '
            return json.dumps(rows).encode()

        with alternate_snapshot({BATCH / '9.json': changed}), self.assertRaisesRegex(
                IdentityError, '^no unique full-run response match$'):
            load_identity(ROOT)

    def test_duplicate_candidates_cannot_be_disambiguated_by_ordinal_or_renames(self):
        with alternate_snapshot({BATCH / '1.json': lambda _: (BATCH / '0.json').read_bytes()}), \
                self.assertRaisesRegex(IdentityError, '^no unique full-run response match$'):
            load_identity(ROOT)

    def test_transposed_complete_batches_contradict_the_independent_rename_trace(self):
        raw = Path.read_bytes  # Capture before substituting virtual paths.
        changes = {BATCH / '4.json': lambda _: raw(BATCH / '5.json'),
                   BATCH / '5.json': lambda _: raw(BATCH / '4.json')}
        with alternate_snapshot(changes), self.assertRaisesRegex(
                IdentityError, '^run corroboration mismatch$'):
            load_identity(ROOT)

    def test_type_confused_index_must_not_compare_equal_to_an_integer(self):
        def changed(data):
            rows = json.loads(data)
            rows[1]['idx'] = True
            return json.dumps(rows).encode()

        with alternate_snapshot({BATCH / '2.json': changed}), self.assertRaisesRegex(
                IdentityError, '^batch index mismatch$'):
            load_identity(ROOT)

    def test_matching_answers_do_not_override_a_different_requested_model(self):
        def changed(data):
            return data.replace(b"model='deepseek-v3'", b"model='other'", 1)

        with alternate_snapshot({LOG: changed}), self.assertRaisesRegex(
                IdentityError, '^requested model mismatch$'):
            load_identity(ROOT)

    def test_missing_git_objects_do_not_fall_back_to_current_csv(self):
        import subprocess
        unavailable = subprocess.CompletedProcess([], 1, b'', b'sentinel')
        with patch.object(subprocess, 'run', return_value=unavailable), self.assertRaisesRegex(
                IdentityError, '^local historical object unavailable$'):
            load_identity(ROOT)


if __name__ == '__main__':
    unittest.main()
