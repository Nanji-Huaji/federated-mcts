"""Contract tests for API client reuse and single-retry-layer defects.

PRODUCTION DEFECTS:
1. _create_client() returns a new openai.OpenAI(max_retries=5) on every
   call — no caching by (base_url, api_key).
2. _create_completion has a @backoff.on_exception decorator that
adds a second retry layer atop the OpenAI client's built-in retries.

Expected contract:
- Clients are reused (singleton per endpoint/key pair).
- Exactly one retry mechanism is active.
"""

import sys
import unittest

sys.path.insert(0, "src")

from federated_mcts.models.api_client import _create_client, _create_completion


class TestApiClientCache(unittest.TestCase):

    def test_same_client_returned_for_equal_endpoint_and_key(self):
        c1 = _create_client(api_base="http://a:8000/v1", api_key="k1")
        c2 = _create_client(api_base="http://a:8000/v1", api_key="k1")
        self.assertIs(
            c1, c2,
            "DEFECT: _create_client returns a new OpenAI instance every call. "
            "Expected: reuse clients for equal (base_url, api_key).",
        )

    def test_different_clients_for_different_endpoints(self):
        c1 = _create_client(api_base="http://a:8000/v1", api_key="k1")
        c2 = _create_client(api_base="http://b:8000/v1", api_key="k1")
        self.assertIsNot(c1, c2)
        self.assertNotEqual(c1.base_url, c2.base_url)

    def test_different_clients_for_different_keys(self):
        c1 = _create_client(api_base="http://a:8000/v1", api_key="k1")
        c2 = _create_client(api_base="http://a:8000/v1", api_key="k2")
        self.assertIsNot(c1, c2)


class TestSingleRetryLayer(unittest.TestCase):

    def test_backoff_decorator_absent(self):
        has_wrapper = hasattr(_create_completion, "__wrapped__")
        self.assertFalse(
            has_wrapper,
            "DEFECT: @backoff.on_exception decorator adds a second retry "
            "layer on top of OpenAI(max_retries=5).  "
            "Expected: undecorated; only the OpenAI client retries.",
        )


if __name__ == "__main__":
    unittest.main()
