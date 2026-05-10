from types import SimpleNamespace

import pytest

from opentelemetry.instrumentation.vertexai import event_emitter as ee
from opentelemetry.instrumentation.vertexai.event_models import ChoiceEvent, MessageEvent


class FakeLogger:
    def __init__(self):
        self.records = []

    def emit(self, record):
        self.records.append(record)


def test_parse_vertex_finish_reason_branches():
    assert ee._parse_vertex_finish_reason(None) == "unknown"
    assert ee._parse_vertex_finish_reason(1) == "stop"
    assert ee._parse_vertex_finish_reason(SimpleNamespace(value=2)) == "max_tokens"
    assert ee._parse_vertex_finish_reason(999) == "unknown"


def test_emit_prompt_events_string_and_list(monkeypatch):
    logger = FakeLogger()
    monkeypatch.setattr(ee, "should_emit_events", lambda: True)
    monkeypatch.setattr(ee, "should_send_prompts", lambda: True)

    ee.emit_prompt_events(["hello", ["world", "again"]], logger)

    assert len(logger.records) == 1
    record = logger.records[0]
    assert record.event_name == "gen_ai.user.message"
    assert record.body["content"] == "hello\nworld\nagain\n"


def test_emit_response_events_string(monkeypatch):
    logger = FakeLogger()
    monkeypatch.setattr(ee, "should_emit_events", lambda: True)
    monkeypatch.setattr(ee, "should_send_prompts", lambda: True)

    ee.emit_response_events("assistant reply", logger)

    assert len(logger.records) == 1
    record = logger.records[0]
    assert record.event_name == "gen_ai.choice"
    assert record.body["finish_reason"] == "unknown"
    assert record.body["message"]["content"] == "assistant reply"


def test_emit_response_events_generation_response(monkeypatch):
    logger = FakeLogger()
    monkeypatch.setattr(ee, "should_emit_events", lambda: True)
    monkeypatch.setattr(ee, "should_send_prompts", lambda: True)

    class FakeGenerationResponse:
        pass

    monkeypatch.setattr(ee, "GenerationResponse", FakeGenerationResponse)

    response = FakeGenerationResponse()
    response.candidates = [
        SimpleNamespace(index=0, text="first", finish_reason=1),
        SimpleNamespace(index=1, text="second", finish_reason=SimpleNamespace(value=2)),
    ]

    ee.emit_response_events(response, logger)

    assert len(logger.records) == 2
    assert logger.records[0].body["index"] == 0
    assert logger.records[0].body["finish_reason"] == "stop"
    assert logger.records[1].body["index"] == 1
    assert logger.records[1].body["finish_reason"] == "max_tokens"


def test_emit_event_returns_when_disabled_or_logger_missing(monkeypatch):
    monkeypatch.setattr(ee, "should_emit_events", lambda: False)
    logger = FakeLogger()
    ee.emit_event(MessageEvent(content="x"), logger)
    assert logger.records == []

    monkeypatch.setattr(ee, "should_emit_events", lambda: True)
    ee.emit_event(MessageEvent(content="x"), None)


def test_emit_event_unsupported_type_raises(monkeypatch):
    monkeypatch.setattr(ee, "should_emit_events", lambda: True)
    logger = FakeLogger()
    with pytest.raises(TypeError):
        ee.emit_event(object(), logger)


def test_emit_message_event_role_and_tool_calls_handling(monkeypatch):
    logger = FakeLogger()
    monkeypatch.setattr(ee, "should_emit_events", lambda: True)
    monkeypatch.setattr(ee, "should_send_prompts", lambda: True)

    event = MessageEvent(
        content="hello",
        role="assistant",
        tool_calls=[{"id": "1", "type": "function", "function": {"function_name": "f", "arguments": {"a": 1}}}],
    )
    ee.emit_event(event, logger)

    record = logger.records[0]
    assert record.event_name == "gen_ai.assistant.message"
    assert "role" not in record.body
    assert "tool_calls" in record.body


def test_emit_message_event_redacts_content_when_prompts_disabled(monkeypatch):
    logger = FakeLogger()
    monkeypatch.setattr(ee, "should_emit_events", lambda: True)
    monkeypatch.setattr(ee, "should_send_prompts", lambda: False)

    event = MessageEvent(
        content="secret",
        role="assistant",
        tool_calls=[{"id": "1", "type": "function", "function": {"function_name": "f", "arguments": {"x": "y"}}}],
    )
    ee.emit_event(event, logger)
    body = logger.records[0].body
    assert "content" not in body
    assert "arguments" not in body["tool_calls"][0]["function"]


def test_emit_choice_event_redacts_content_and_assistant_role(monkeypatch):
    logger = FakeLogger()
    monkeypatch.setattr(ee, "should_emit_events", lambda: True)
    monkeypatch.setattr(ee, "should_send_prompts", lambda: False)

    event = ChoiceEvent(
        index=0,
        message={"content": "hidden", "role": "assistant"},
        finish_reason="stop",
        tool_calls=[{"id": "1", "type": "function", "function": {"function_name": "f", "arguments": {"x": "y"}}}],
    )
    ee.emit_event(event, logger)
    body = logger.records[0].body
    assert "role" not in body["message"]
    assert "content" not in body["message"]
    assert "arguments" not in body["tool_calls"][0]["function"]
