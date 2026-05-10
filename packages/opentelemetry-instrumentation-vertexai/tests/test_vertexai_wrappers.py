from types import SimpleNamespace

import pytest

import opentelemetry.instrumentation.vertexai as vi


class FakeSpan:
    def __init__(self):
        self.status = None
        self.ended = False
        self.attributes = {}

    def is_recording(self):
        return True

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def set_status(self, status):
        self.status = status

    def end(self):
        self.ended = True


class FakeTracer:
    def __init__(self, span):
        self.span = span

    def start_span(self, name, kind, attributes):
        self.span.start_name = name
        self.span.start_kind = kind
        self.span.start_attributes = attributes
        return self.span


def _sync_generator():
    yield SimpleNamespace(text="A", usage_metadata=None)
    yield SimpleNamespace(
        text="B",
        usage_metadata=SimpleNamespace(total_token_count=5, candidates_token_count=3, prompt_token_count=2),
    )


async def _async_generator():
    yield SimpleNamespace(text="A", usage_metadata=None)
    yield SimpleNamespace(
        text="B",
        usage_metadata=SimpleNamespace(total_token_count=5, candidates_token_count=3, prompt_token_count=2),
    )


def test_wrap_sync_non_streaming(monkeypatch):
    span = FakeSpan()
    tracer = FakeTracer(span)
    instance = SimpleNamespace(_model_id="text-bison@001")
    response = SimpleNamespace(
        usage_metadata=SimpleNamespace(total_token_count=10, candidates_token_count=6, prompt_token_count=4),
        candidates=[SimpleNamespace(text="done")],
    )

    monkeypatch.setattr(vi.context_api, "get_value", lambda key: False)
    monkeypatch.setattr(vi, "should_emit_events", lambda: False)

    wrapped = vi._wrap(tracer, None, {"span_name": "vertexai.predict"})(
        lambda *args, **kwargs: response,
        instance,
        ("hello",),
        {"prompt": "hello"},
    )
    assert wrapped is response
    assert span.ended is True
    assert span.start_name == "vertexai.predict"


def test_wrap_sync_streaming_generator(monkeypatch):
    span = FakeSpan()
    tracer = FakeTracer(span)
    instance = SimpleNamespace(_model_id="text-bison@001")

    monkeypatch.setattr(vi.context_api, "get_value", lambda key: False)
    monkeypatch.setattr(vi, "should_emit_events", lambda: False)

    wrapped = vi._wrap(tracer, None, {"span_name": "vertexai.predict_streaming"})(
        lambda *args, **kwargs: _sync_generator(),
        instance,
        ("hello",),
        {"prompt": "hello"},
    )
    parts = [chunk.text for chunk in wrapped]
    assert parts == ["A", "B"]
    assert span.ended is True


@pytest.mark.asyncio
async def test_awrap_async_non_streaming(monkeypatch):
    span = FakeSpan()
    tracer = FakeTracer(span)
    instance = SimpleNamespace(_model_name="publishers/google/models/gemini-2.0-flash-lite")
    response = SimpleNamespace(
        usage_metadata=SimpleNamespace(total_token_count=10, candidates_token_count=6, prompt_token_count=4),
        candidates=[SimpleNamespace(text="done")],
    )

    monkeypatch.setattr(vi.context_api, "get_value", lambda key: False)
    monkeypatch.setattr(vi, "should_emit_events", lambda: False)

    async def wrapped(*args, **kwargs):
        return response

    out = await vi._awrap(tracer, None, {"span_name": "vertexai.generate_content_async"})(
        wrapped, instance, ("hello",), {"prompt": "hello"}
    )
    assert out is response
    assert span.ended is True


@pytest.mark.asyncio
async def test_awrap_async_streaming_generator(monkeypatch):
    span = FakeSpan()
    tracer = FakeTracer(span)
    instance = SimpleNamespace(_model_id="text-bison@001")

    monkeypatch.setattr(vi.context_api, "get_value", lambda key: False)
    monkeypatch.setattr(vi, "should_emit_events", lambda: False)

    async def wrapped(*args, **kwargs):
        return _async_generator()

    stream = await vi._awrap(tracer, None, {"span_name": "vertexai.predict_streaming_async"})(
        wrapped, instance, ("hello",), {"prompt": "hello"}
    )
    parts = []
    async for chunk in stream:
        parts.append(chunk.text)
    assert parts == ["A", "B"]
    assert span.ended is True


def test_wrap_suppression_keys_skip_instrumentation(monkeypatch):
    span = FakeSpan()
    tracer = FakeTracer(span)
    instance = SimpleNamespace(_model_id="text-bison@001")

    monkeypatch.setattr(
        vi.context_api,
        "get_value",
        lambda key: key in (vi._SUPPRESS_INSTRUMENTATION_KEY, vi.SUPPRESS_LANGUAGE_MODEL_INSTRUMENTATION_KEY),
    )
    monkeypatch.setattr(vi, "should_emit_events", lambda: False)

    result = vi._wrap(tracer, None, {"span_name": "vertexai.predict"})(
        lambda *args, **kwargs: "raw",
        instance,
        ("hello",),
        {"prompt": "hello"},
    )
    assert result == "raw"
    assert span.ended is False


def test_model_resolution_with_nested_model(monkeypatch):
    span = FakeSpan()
    tracer = FakeTracer(span)
    instance = SimpleNamespace(
        _model=SimpleNamespace(_model_name="publishers/google/models/gemini-2.5-flash", _model_id="fallback-model")
    )
    response = SimpleNamespace(
        usage_metadata=SimpleNamespace(total_token_count=1, candidates_token_count=1, prompt_token_count=0),
        candidates=[SimpleNamespace(text="ok")],
    )

    monkeypatch.setattr(vi.context_api, "get_value", lambda key: False)
    monkeypatch.setattr(vi, "should_emit_events", lambda: False)

    vi._wrap(tracer, None, {"span_name": "vertexai.send_message"})(
        lambda *args, **kwargs: response,
        instance,
        ("hello",),
        {"prompt": "hello"},
    )
    assert span.ended is True


def test_instrument_and_uninstrument_call_all_wrapped_methods(monkeypatch):
    wrapped_calls = []
    unwrap_calls = []

    monkeypatch.setattr(vi, "should_emit_events", lambda: False)
    monkeypatch.setattr(vi, "wrap_function_wrapper", lambda package, method, wrapper: wrapped_calls.append((package, method)))
    monkeypatch.setattr(vi, "unwrap", lambda target, method: unwrap_calls.append((target, method)))

    instrumentor = vi.VertexAIInstrumentor()
    instrumentor._instrument()
    instrumentor._uninstrument()

    assert len(wrapped_calls) == len(vi.WRAPPED_METHODS)
    assert len(unwrap_calls) == len(vi.WRAPPED_METHODS)
