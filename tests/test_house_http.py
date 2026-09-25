from __future__ import annotations

import json
import os
from unittest.mock import patch

import t2_forecaster.house_overlay as h


class _Resp:
    def __init__(self, status, payload):
        self.status = status
        self._payload = payload
    def read(self, limit):
        return self._payload


class _Conn:
    response = None
    captured = {}
    def __init__(self, host, port, timeout):
        self.captured["connect"] = (host, port, timeout)
    def request(self, method, target, body, headers):
        self.captured["request"] = (method, target, body, headers)
    def getresponse(self):
        return self.response
    def close(self):
        self.captured["closed"] = True


def _env():
    return {
        "MODEL_ENDPOINT": "http://model.local",
        "MODEL_NAME": "house-model",
        "MODEL_TOKEN": "token123",
        "http_proxy": "http://user:pass@proxy.local:8080",
    }


def _envelope(content, finish="stop", reasoning=None):
    msg = {"content": content}
    if reasoning is not None:
        msg["reasoning_content"] = reasoning
    return json.dumps({"choices": [{"finish_reason": finish, "message": msg}]}).encode()


def test_house_route_and_payload_are_platform_compatible():
    _Conn.captured = {}
    _Conn.response = _Resp(200, _envelope('{"claims":[]}'))
    with patch.dict(os.environ, _env(), clear=True), patch.object(h.http.client, "HTTPConnection", _Conn):
        got = h._house("hello")
    assert got == {"claims": []}
    method, target, body, headers = _Conn.captured["request"]
    payload = json.loads(body)
    assert method == "POST"
    assert target == "http://model.local/v1/chat/completions"
    assert headers["Authorization"] == "Bearer token123"
    assert headers["Proxy-Authorization"].startswith("Basic ")
    assert payload["model"] == "house-model"
    assert payload["max_tokens"] == 1200
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert _Conn.captured["closed"] is True


def test_house_403_is_noop():
    _Conn.response = _Resp(403, b"{}")
    with patch.dict(os.environ, _env(), clear=True), patch.object(h.http.client, "HTTPConnection", _Conn):
        assert h._house("hello") is None


def test_house_truncated_reply_is_noop():
    _Conn.response = _Resp(200, _envelope('{"claims":[]}', finish="length"))
    with patch.dict(os.environ, _env(), clear=True), patch.object(h.http.client, "HTTPConnection", _Conn):
        assert h._house("hello") is None


def test_house_prose_wrapped_json_is_noop():
    _Conn.response = _Resp(200, _envelope('Here is the answer: {"claims":[]}'))
    with patch.dict(os.environ, _env(), clear=True), patch.object(h.http.client, "HTTPConnection", _Conn):
        assert h._house("hello") is None
