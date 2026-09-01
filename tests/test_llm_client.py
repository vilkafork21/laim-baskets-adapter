"""Клиент: проба response_format с фолбэком, repair-петля, счётчики."""
from __future__ import annotations

import json


from laim_basket.config import LlmConfig
from laim_basket.llm import client as client_module

SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}},
          "required": ["ok"], "additionalProperties": False}


class _Response:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


def _ok(content):
    return _Response(200, {"choices": [{"message": {"content": content}}]})


def _client(tmp_path):
    config = LlmConfig(url="http://gw/api", model="glm-5.2", api_key="123")
    return client_module.LlmClient(config, tmp_path / "debug")


def test_response_format_probe_falls_back_on_400(tmp_path, monkeypatch):
    bodies = []

    def fake_post(url, headers=None, json=None, timeout=None):
        bodies.append(dict(json))  # снимок: клиент мутирует body при фолбэке
        if "response_format" in json:
            return _Response(400, {"error": "unknown field response_format"})
        return _ok('{"ok": true}')

    monkeypatch.setattr(client_module.requests, "post", fake_post)
    client = _client(tmp_path)
    result = client_module.request_structured(
        client, [{"role": "user", "content": "?"}], SCHEMA, "probe")
    assert result == {"ok": True}
    assert "response_format" in bodies[0] and "response_format" not in bodies[1]
    assert client.structured_output is False
    client.chat([{"role": "user", "content": "?"}], "next", response_schema=SCHEMA)
    assert "response_format" not in bodies[2]


def test_response_format_success_sets_flag(tmp_path, monkeypatch):
    monkeypatch.setattr(
        client_module.requests, "post",
        lambda url, headers=None, json=None, timeout=None: _ok('{"ok": true}'))
    client = _client(tmp_path)
    client_module.request_structured(
        client, [{"role": "user", "content": "?"}], SCHEMA, "task")
    assert client.structured_output is True


def test_repair_turn_sends_error_text_and_counts(tmp_path, monkeypatch):
    answers = iter(["не json", '{"ok": true}'])
    requests_seen = []

    def fake_post(url, headers=None, json=None, timeout=None):
        requests_seen.append(json)
        return _ok(next(answers))

    monkeypatch.setattr(client_module.requests, "post", fake_post)
    client = _client(tmp_path)
    result = client_module.request_structured(
        client, [{"role": "user", "content": "?"}], SCHEMA, "task")
    assert result == {"ok": True}
    assert client.repair_turns == 1 and client.calls == 2
    repair_message = requests_seen[1]["messages"][-1]["content"]
    assert "отклонён" in repair_message


def test_think_block_is_stripped(tmp_path, monkeypatch):
    monkeypatch.setattr(
        client_module.requests, "post",
        lambda url, headers=None, json=None, timeout=None:
        _ok('<think>шум</think>{"ok": true}'))
    client = _client(tmp_path)
    assert client_module.request_structured(
        client, [{"role": "user", "content": "?"}], SCHEMA, "t") == {"ok": True}
