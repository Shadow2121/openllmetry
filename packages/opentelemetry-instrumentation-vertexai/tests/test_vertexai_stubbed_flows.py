from types import SimpleNamespace

import pytest

import opentelemetry.instrumentation.vertexai as vi


class FakeSpan:
    def __init__(self):
        self.ended = False
        self.attrs = {}
        self.status = None

    def is_recording(self):
        return True

    def set_attribute(self, key, value):
        self.attrs[key] = value

    def set_status(self, status):
        self.status = status

    def end(self):
        self.ended = True


class FakeTracer:
    def __init__(self, span):
        self.span = span
        self.name = None

    def start_span(self, name, kind, attributes):
        self.name = name
        return self.span


def _mk_response(text="hello"):
    return SimpleNamespace(
        usage_metadata=SimpleNamespace(total_token_count=10, candidates_token_count=6, prompt_token_count=4),
        candidates=[SimpleNamespace(text=text)],
    )


def _mk_stream():
    yield SimpleNamespace(text="hello ", usage_metadata=None)
    yield SimpleNamespace(
        text="world",
        usage_metadata=SimpleNamespace(total_token_count=11, candidates_token_count=7, prompt_token_count=4),
    )


async def _mk_astream():
    yield SimpleNamespace(text="hello ", usage_metadata=None)
    yield SimpleNamespace(
        text="world",
        usage_metadata=SimpleNamespace(total_token_count=11, candidates_token_count=7, prompt_token_count=4),
    )


def test_stubbed_bison_predict_and_chat_sync(monkeypatch):
    monkeypatch.setattr(vi.context_api, "get_value", lambda key: False)
    monkeypatch.setattr(vi, "should_emit_events", lambda: False)

    span = FakeSpan()
    tracer = FakeTracer(span)

    predict_out = vi._wrap(tracer, None, {"span_name": "vertexai.predict"})(
        lambda *args, **kwargs: _mk_response("predict output"),
        SimpleNamespace(_model_id="text-bison@001"),
        ("Give me questions",),
        {"prompt": "Give me questions", "max_output_tokens": 256, "top_p": 0.8, "top_k": 40},
    )
    assert tracer.name == "vertexai.predict"
    assert predict_out.candidates[0].text == "predict output"

    chat_out = vi._wrap(tracer, None, {"span_name": "vertexai.send_message"})(
        lambda *args, **kwargs: _mk_response("chat output"),
        SimpleNamespace(_model_id="chat-bison@001"),
        ("How many planets?",),
        {"max_output_tokens": 256, "top_p": 0.95, "top_k": 40},
    )
    assert tracer.name == "vertexai.send_message"
    assert chat_out.candidates[0].text == "chat output"
    assert span.ended is True


@pytest.mark.asyncio
async def test_stubbed_bison_predict_async_and_stream_async(monkeypatch):
    monkeypatch.setattr(vi.context_api, "get_value", lambda key: False)
    monkeypatch.setattr(vi, "should_emit_events", lambda: False)
    span = FakeSpan()
    tracer = FakeTracer(span)

    async def predict_async(*args, **kwargs):
        return _mk_response("predict async output")

    async_out = await vi._awrap(tracer, None, {"span_name": "vertexai.predict_async"})(
        predict_async,
        SimpleNamespace(_model_id="text-bison@001"),
        ("Give me questions",),
        {"prompt": "Give me questions"},
    )
    assert tracer.name == "vertexai.predict_async"
    assert async_out.candidates[0].text == "predict async output"

    async def stream_async(*args, **kwargs):
        return _mk_astream()

    stream = await vi._awrap(tracer, None, {"span_name": "vertexai.predict_streaming_async"})(
        stream_async,
        SimpleNamespace(_model_id="text-bison@001"),
        ("Give me questions",),
        {"prompt": "Give me questions"},
    )
    parts = []
    async for item in stream:
        parts.append(item.text)
    assert "".join(parts) == "hello world"


def test_stubbed_bison_stream_and_chat_stream_sync(monkeypatch):
    monkeypatch.setattr(vi.context_api, "get_value", lambda key: False)
    monkeypatch.setattr(vi, "should_emit_events", lambda: False)
    span = FakeSpan()
    tracer = FakeTracer(span)

    predict_stream = vi._wrap(tracer, None, {"span_name": "vertexai.predict_streaming"})(
        lambda *args, **kwargs: _mk_stream(),
        SimpleNamespace(_model_id="text-bison"),
        (),
        {"prompt": "Give me questions"},
    )
    assert "".join([item.text for item in predict_stream]) == "hello world"

    chat_stream = vi._wrap(tracer, None, {"span_name": "vertexai.send_message_streaming"})(
        lambda *args, **kwargs: _mk_stream(),
        SimpleNamespace(_model_id="chat-bison@001"),
        (),
        {"message": "How many planets?"},
    )
    assert "".join([item.text for item in chat_stream]) == "hello world"


def test_stubbed_gemini_generate_content_event_mode(monkeypatch):
    monkeypatch.setattr(vi.context_api, "get_value", lambda key: False)
    monkeypatch.setattr(vi, "should_emit_events", lambda: True)
    calls = {"prompt": 0, "response": 0}
    monkeypatch.setattr(vi, "emit_prompt_events", lambda args, logger: calls.__setitem__("prompt", calls["prompt"] + 1))
    monkeypatch.setattr(vi, "emit_response_events", lambda response, logger: calls.__setitem__("response", calls["response"] + 1))

    span = FakeSpan()
    tracer = FakeTracer(span)
    out = vi._wrap(tracer, object(), {"span_name": "vertexai.generate_content"})(
        lambda *args, **kwargs: _mk_response("gemini output"),
        SimpleNamespace(_model_name="publishers/google/models/gemini-2.0-flash-lite"),
        (["what is shown in this image?"],),
        {},
    )
    assert out.candidates[0].text == "gemini output"
    assert calls["prompt"] == 1
    assert calls["response"] == 1
