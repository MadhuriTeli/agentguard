"""Tests for Client.

These never touch the network: anthropic.Anthropic() doesn't make any
requests at construction time, so we build a real client and then swap out
`_client.messages.create` with a controllable stub. This tests our own
retry/backoff/error-normalization logic, not Anthropic's API itself.
"""

from __future__ import annotations

from typing import ClassVar

import anthropic
import pytest

from agentguard.judges.client import Client, LLMClientError


def _make_client(
    *,
    max_retries: int = 3,
    retry_base_delay_seconds: float = 0.0,
) -> Client:
    return Client(
        max_retries=max_retries,
        retry_base_delay_seconds=retry_base_delay_seconds,
    )


def _text_response(text: str):
    class _Block:
        type = "text"

        def __init__(self, t):
            self.text = t

    class _Response:
        content: ClassVar[list] = [_Block(text)]

    return _Response()


def _status_error(status_code: int, message: str = "boom"):
    request = anthropic._base_client.httpx2.Request(
        "POST",
        "https://api.anthropic.com/v1/messages",
    )
    response = anthropic._base_client.httpx2.Response(
        status_code,
        request=request,
    )

    return anthropic.APIStatusError(message, response=response, body=None)


def _connection_error(message: str = "connection reset"):
    request = anthropic._base_client.httpx2.Request(
        "POST",
        "https://api.anthropic.com/v1/messages",
    )

    return anthropic.APIConnectionError(message=message, request=request)


def test_successful_call_returns_text_on_first_try():
    client = _make_client()
    call_count = {"n": 0}

    def fake_create(**kwargs):
        call_count["n"] += 1
        return _text_response('{"verdict": "pass", "reason": "ok"}')

    client._client.messages.create = fake_create

    result = client.complete("grade this")

    assert result == '{"verdict": "pass", "reason": "ok"}'
    assert call_count["n"] == 1


def test_transient_rate_limit_is_retried_then_succeeds():
    client = _make_client()
    calls = {"n": 0}

    def fake_create(**kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            raise _status_error(429, "rate limited")
        return _text_response("recovered")

    client._client.messages.create = fake_create

    result = client.complete("grade this")

    assert result == "recovered"
    assert calls["n"] == 3


def test_non_transient_error_is_not_retried():
    client = _make_client()
    calls = {"n": 0}

    def fake_create(**kwargs):
        calls["n"] += 1
        raise _status_error(401, "invalid api key")

    client._client.messages.create = fake_create

    with pytest.raises(LLMClientError, match="non-retryable"):
        client.complete("grade this")

    # a 401 will never succeed on retry — must fail fast, not burn attempts
    assert calls["n"] == 1


def test_exhausting_retries_raises_llm_client_error():
    client = _make_client(max_retries=2)
    calls = {"n": 0}

    def fake_create(**kwargs):
        calls["n"] += 1
        raise _status_error(503, "overloaded")

    client._client.messages.create = fake_create

    with pytest.raises(LLMClientError):
        client.complete("grade this")

    assert calls["n"] == 2


def test_connection_error_is_retried():
    client = _make_client()
    calls = {"n": 0}

    def fake_create(**kwargs):
        calls["n"] += 1
        if calls["n"] < 2:
            raise _connection_error()
        return _text_response("recovered")

    client._client.messages.create = fake_create

    assert client.complete("grade this") == "recovered"
    assert calls["n"] == 2


def test_empty_content_raises_llm_client_error():
    client = _make_client()

    class _EmptyResponse:
        content: ClassVar[list] = []

    client._client.messages.create = lambda **kwargs: _EmptyResponse()

    with pytest.raises(LLMClientError, match="no text content"):
        client.complete("grade this")


def test_system_prompt_and_task_prompt_are_passed_separately():
    client = _make_client()
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return _text_response('{"verdict": "pass", "reason": "ok"}')

    client._client.messages.create = fake_create
    client.complete("this is the task-specific prompt")

    assert captured["system"] == client.system_prompt
    assert captured["messages"] == [{"role": "user", "content": "this is the task-specific prompt"}]
