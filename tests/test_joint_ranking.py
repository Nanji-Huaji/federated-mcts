import sys
import unittest

sys.path.insert(0, "src")

from federated_mcts.core.joint_ranking import evaluate_ranked_candidates, parse_ranked_scores


class _Args:
    eval_rule = False
    n_evaluate_sample = 1


class _Task:
    value_cache = {}

    @staticmethod
    def canonical_state_key(x, y):
        return y.split(":", 1)[0]

    @staticmethod
    def joint_rank_prompt_wrap(x, candidates):
        return "[JOINT_RANK]" + "|".join(str(i) for i in range(len(candidates)))

    @staticmethod
    def value_prompt_wrap(x, y):
        return y

    @staticmethod
    def value_outputs_unwrap(x, y, outputs):
        return 0.25

    @staticmethod
    def pre_value_check(y, eval_rule):
        return 0, False

    @staticmethod
    def is_success_state(x, y):
        return False


class _Client:
    def __init__(self, response):
        self.response = response
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return [self.response]


class TestJointRankingParser(unittest.TestCase):
    def test_parses_complete_id_score_payload(self):
        payload = '{"ranking":[{"id":1,"score":0.9},{"id":0,"score":0.4}]}'

        scores = parse_ranked_scores(payload, candidate_count=2)

        self.assertEqual(scores, [0.4, 0.9])

    def test_rejects_duplicate_or_missing_ids(self):
        payload = '{"ranking":[{"id":0,"score":0.9},{"id":0,"score":0.4}]}'

        self.assertIsNone(parse_ranked_scores(payload, candidate_count=2))


class TestJointRankingEvaluation(unittest.TestCase):
    def test_one_joint_call_scores_all_uncached_states(self):
        client = _Client('{"ranking":[{"id":0,"score":0.8},{"id":1,"score":0.3}]}')
        cache = {}

        scores = evaluate_ranked_candidates(
            _Args(), _Task(), "x", ["a:path1", "b:path2"], client,
            evaluator_id="remote", cache=cache, joint_rank=True,
        )

        self.assertEqual(scores, [0.8, 0.3])
        self.assertEqual(client.calls, 1)
        self.assertEqual(len(cache), 2)

    def test_transposition_cache_skips_equivalent_state(self):
        client = _Client('{"ranking":[{"id":0,"score":0.7}]}')
        cache = {("_Task", "remote", "a"): 0.6}

        scores = evaluate_ranked_candidates(
            _Args(), _Task(), "x", ["a:new-path"], client,
            evaluator_id="remote", cache=cache, joint_rank=True,
        )

        self.assertEqual(scores, [0.6])
        self.assertEqual(client.calls, 0)

    def test_invalid_joint_payload_falls_back_to_value_evaluation(self):
        client = _Client("not-json")

        scores = evaluate_ranked_candidates(
            _Args(), _Task(), "x", ["a:path1", "b:path2"], client,
            evaluator_id="remote", cache={}, joint_rank=True,
        )

        self.assertEqual(scores, [0.25, 0.25])
        self.assertEqual(client.calls, 3)


if __name__ == "__main__":
    unittest.main()
