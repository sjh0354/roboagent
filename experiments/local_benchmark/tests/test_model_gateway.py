import json
import urllib.error

from experiments.local_benchmark.model import ModelGateway


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


def test_transport_retry_is_bounded_and_audited(tmp_path, monkeypatch):
    calls = {"count": 0}

    def fake_urlopen(_request, timeout):
        assert timeout == 4.0
        calls["count"] += 1
        if calls["count"] < 3:
            raise urllib.error.URLError("transient eof")
        return _Response({"choices": [{"message": {"content": "{}"}}]})

    monkeypatch.setattr("experiments.local_benchmark.model.urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("experiments.local_benchmark.model.time.sleep", lambda _seconds: None)
    gateway = ModelGateway.__new__(ModelGateway)
    gateway.base_url = "https://example.invalid/v1"
    gateway.api_key = "redacted"
    gateway.log_path = tmp_path / "calls.jsonl"
    gateway.timeout_seconds = 4.0
    gateway.call_count = 0
    gateway.role_counts = {}
    body, _elapsed = gateway._post_json({"model": "kimi-k3"}, "online_monitor")
    assert body["choices"][0]["message"]["content"] == "{}"
    assert calls["count"] == 3
    failures = [json.loads(line) for line in gateway.log_path.read_text().splitlines()]
    assert [row["attempt"] for row in failures] == [1, 2]
    assert all(row["api_call_attempt"] and row["retryable"] for row in failures)
