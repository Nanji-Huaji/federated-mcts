"""Public configuration/manifest contracts, with no client or API activity."""
from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from federated_mcts.config import DecisionSpec, load_model_config
from federated_mcts.evaluation.game24_b0_manifest_loader import derive_fingerprints_from_manifest_dict
from federated_mcts.evaluation.game24_manifest import Game24Manifest
from federated_mcts.evaluation.game24_runner_args import parse_args
from federated_mcts.experiments.runner import execute, prepare


class TestDecisionIdentity(unittest.TestCase):
    @staticmethod
    def manifest(spec=None):
        spec = spec or DecisionSpec()
        return Game24Manifest(
            task="text", task_ids=[2, 0, 1], task_split_name=None,
            decision_policy="local_validity", oracle_audit=False,
            seed=spec.seed or 0, temperature=spec.temperature,
            method_generate=spec.method_generate, method_evaluate=spec.method_evaluate,
            method_select=spec.method_select, n_generate_sample=spec.n_generate_sample,
            n_evaluate_sample=spec.n_evaluate_sample, n_select_sample=spec.n_select_sample,
            game24_exact_prune=spec.game24_exact_prune, assign_strategy=spec.assign_strategy,
            schema_version=2, decision_spec=spec, runner_path="fmcts-run")

    def test_all_consumed_options_participate_in_identity(self):
        original = self.manifest()
        mutations = {
            "solve_method": "federated_solve", "search_policy": "diverse",
            "check_format": True, "eval_rule": True, "warm_start": True,
            "slm_generate": True, "slm_eval": True, "last_lm": True,
            "diverse_joint_rank": False, "diversity_weight": .1,
            "beam_expand": 3, "dqn_token_budget": 1., "dqn_epsilon": .2,
            "dqn_checkpoint_sha256": "a" * 64, "rate_card_version": "other",
            "mock_client": True,
        }
        for field, value in mutations.items():
            with self.subTest(field=field):
                manifest = self.manifest(replace(DecisionSpec(), **{field: value}))
                self.assertNotEqual(manifest.run_fingerprint, original.run_fingerprint)
                self.assertNotEqual(manifest.decision_policy_fingerprint,
                                    original.decision_policy_fingerprint)
                self.assertEqual(derive_fingerprints_from_manifest_dict(json.loads(manifest.to_json())),
                                 (manifest.decision_policy_fingerprint, manifest.run_fingerprint))

    def test_output_locations_are_not_decision_identity(self):
        before = DecisionSpec(dqn_checkpoint_sha256="a" * 64)
        after = replace(before, dqn_checkpoint="different/path", dqn_jsonl="trace.jsonl")
        self.assertEqual(before.identity(), after.identity())
        with self.assertRaises(FrozenInstanceError):
            before.seed = 9

    def test_checkpoint_identity_follows_bytes_not_filename(self):
        with tempfile.TemporaryDirectory() as temp:
            first, second = Path(temp) / "first", Path(temp) / "second"
            first.write_bytes(b"weights")
            second.write_bytes(b"weights")
            a = DecisionSpec.from_args(parse_args(["--task", "text", "--dqn_checkpoint", str(first)]))
            b = DecisionSpec.from_args(parse_args(["--task", "text", "--dqn_checkpoint", str(second)]))
            self.assertEqual(a.identity(), b.identity())
            second.write_bytes(b"different")
            c = DecisionSpec.from_args(parse_args(["--task", "text", "--dqn_checkpoint", str(second)]))
            self.assertNotEqual(a.identity(), c.identity())
            self.assertEqual(c.dqn_checkpoint_sha256, hashlib.sha256(b"different").hexdigest())

    def test_invalid_or_nonfinite_identity_is_rejected(self):
        for field, value in (("seed", True), ("seed", "2"), ("temperature", float("nan")),
                             ("diversity_weight", float("inf")), ("prompt_sample", []),
                             ("method_evaluate", "unknown"), ("check_format", 1),
                             ("dqn_checkpoint_sha256", "invalid"), ("n_select_sample", 0)):
            with self.subTest(field=field, value=value):
                data = DecisionSpec().identity()
                data[field] = value
                with self.assertRaises(ValueError):
                    DecisionSpec.validate_identity(data)
        missing = DecisionSpec().identity()
        del missing["check_format"]
        with self.assertRaises(ValueError):
            DecisionSpec.validate_identity(missing)

    def test_legacy_codec_round_trips_without_v2_fields(self):
        legacy = replace(self.manifest(), schema_version=1, decision_spec=None,
                         runner_path="scripts/merged_run.py")
        raw = json.loads(legacy.to_json())
        self.assertNotIn("decision_spec", raw)
        self.assertEqual(raw["schema_version"], 1)
        self.assertEqual(derive_fingerprints_from_manifest_dict(raw),
                         (legacy.decision_policy_fingerprint, legacy.run_fingerprint))
        self.assertNotEqual(legacy.run_fingerprint, self.manifest().run_fingerprint)


class TestPreparedRun(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cwd = os.getcwd()
        os.chdir(self.tmp.name)
        self.config = [{"client_name": "remote_client", "model": "gpt-4o"}]
        Path("models.json").write_text(json.dumps(self.config))

    def tearDown(self):
        os.chdir(self.cwd)
        self.tmp.cleanup()

    def args(self, *extra):
        return parse_args(["--task", "text", "--task_start_index", "0",
                           "--task_end_index", "1", "--model_config", "models.json", *extra])

    def test_prepare_is_side_effect_free_and_snapshots_config_once(self):
        import builtins
        real_open = builtins.open
        reads = []
        def observed_open(path, *args, **kwargs):
            if str(path) == "models.json":
                reads.append(path)
            return real_open(path, *args, **kwargs)
        state = random.getstate()
        with patch("builtins.open", side_effect=observed_open), patch(
                "federated_mcts.models.clients.create_clients") as factory:
            plan = prepare(self.args("--seed", "17"))
            factory.assert_not_called()
        self.assertEqual(len(reads), 1)
        self.assertEqual(random.getstate(), state)
        self.assertFalse(Path("logs").exists())
        snapshot = plan.setup.model_configs
        self.assertEqual(snapshot[0]["max_tokens"], 1000)
        self.assertEqual(snapshot[0]["temperature"], plan.decision.temperature)
        Path("models.json").write_text("[]")
        snapshot[0]["model"] = "mutated"
        self.assertEqual(plan.setup.model_configs[0]["model"], "gpt-4o")

    def test_changed_checkpoint_fails_before_clients_or_run_artifacts(self):
        checkpoint = Path("policy.pt")
        checkpoint.write_bytes(b"initial-policy")
        plan = prepare(self.args("--search_policy", "dqn", "--dqn_checkpoint", str(checkpoint)))
        checkpoint.write_bytes(b"replaced-policy")
        with patch("federated_mcts.models.clients.create_clients") as factory:
            with self.assertRaisesRegex(ValueError, "checkpoint changed"):
                execute(plan)
            factory.assert_not_called()
        self.assertFalse(Path("logs").exists())

    def test_checkpoint_loader_validates_the_deserialized_bytes(self):
        from federated_mcts.core.dqn.checkpoint import CheckpointConfigurationError, load_checkpoint
        checkpoint = Path("policy.pt")
        original = b"initial-policy"
        checkpoint.write_bytes(original)
        digest = hashlib.sha256(original).hexdigest()
        checkpoint.write_bytes(b"replaced-policy")
        kwargs = dict(state_dim=16, action_count=8, hidden_sizes=(64, 64), expected_sha256=digest)
        with patch("federated_mcts.core.dqn.checkpoint.torch.load") as load:
            with self.assertRaisesRegex(CheckpointConfigurationError, "checkpoint changed"):
                load_checkpoint(str(checkpoint), **kwargs)
            load.assert_not_called()
        checkpoint.write_bytes(original)
        payload = {"metadata": {"state_dim": 16, "action_count": 8, "hidden_sizes": [64, 64]},
                   "model": {}}
        with patch("federated_mcts.core.dqn.checkpoint.torch.load", return_value=payload) as load:
            load_checkpoint(str(checkpoint), **kwargs)
        self.assertEqual(load.call_args.args[0].getvalue(), original)
        self.assertTrue(load.call_args.kwargs["weights_only"])

    def test_resume_and_output_index_do_not_change_run_identity(self):
        first = prepare(self.args())
        resumed = prepare(self.args("--resume"))
        relocated = prepare(self.args("--inference_idx", "9"))
        self.assertEqual(first.manifest.run_fingerprint, resumed.manifest.run_fingerprint)
        self.assertEqual(first.manifest.run_fingerprint, relocated.manifest.run_fingerprint)
        self.assertNotEqual(first.setup.file_name, relocated.setup.file_name)

    def test_unknown_rate_fails_before_clients_or_artifacts(self):
        self.config[0]["model"] = "unpriced-model"
        Path("models.json").write_text(json.dumps(self.config))
        with patch("federated_mcts.models.clients.create_clients") as factory:
            with self.assertRaises(ValueError):
                prepare(self.args())
            factory.assert_not_called()
        self.assertFalse(Path("logs").exists())

    def test_credentials_do_not_change_identity_but_defaults_do(self):
        first = prepare(self.args())
        self.config[0]["api_key"] = "do-not-persist-this"
        Path("models.json").write_text(json.dumps(self.config))
        second = prepare(self.args())
        self.assertEqual(first.manifest.run_fingerprint, second.manifest.run_fingerprint)
        self.assertNotIn("do-not-persist-this", second.manifest.to_json())
        self.config[0]["max_tokens"] = 256
        Path("models.json").write_text(json.dumps(self.config))
        third = prepare(self.args())
        self.assertNotEqual(first.manifest.run_fingerprint, third.manifest.run_fingerprint)

    def test_invalid_model_config_is_rejected(self):
        for config in ([], [{"model": "gpt-4o", "client_name": []}], self.config * 2,
                       [{**self.config[0], "max_tokens": True}],
                       [{**self.config[0], "temperature": float("nan")} ]):
            with self.subTest(config=config):
                Path("models.json").write_text(json.dumps(config))
                with self.assertRaises(ValueError):
                    load_model_config("models.json", temperature=.7)


if __name__ == "__main__":
    unittest.main()
