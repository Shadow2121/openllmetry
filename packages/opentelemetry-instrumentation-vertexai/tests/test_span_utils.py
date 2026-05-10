import asyncio
import json
from types import SimpleNamespace

import pytest

from opentelemetry.instrumentation.vertexai import span_utils as su


class FakeSpan:
    def __init__(self, recording=True):
        self._recording = recording
        self.attributes = {}
        self.context = SimpleNamespace(trace_id=123, span_id=456)

    def is_recording(self):
        return self._recording

    def set_attribute(self, key, value):
        self.attributes[key] = value


def test_set_span_attribute_ignores_none_and_empty():
    span = FakeSpan()
    su._set_span_attribute(span, "k1", None)
    su._set_span_attribute(span, "k2", "")
    su._set_span_attribute(span, "k3", "v3")
    assert "k1" not in span.attributes
    assert "k2" not in span.attributes
    assert span.attributes["k3"] == "v3"


def test_is_base64_image_part_branches():
    good = SimpleNamespace(
        mime_type="image/jpeg",
        inline_data=SimpleNamespace(data=b"abc"),
    )
    bad = SimpleNamespace(mime_type="text/plain", inline_data=SimpleNamespace(data=b"abc"))
    missing = SimpleNamespace(mime_type="image/jpeg")
    assert su._is_base64_image_part(good) is True
    assert su._is_base64_image_part(bad) is False
    assert su._is_base64_image_part(missing) is False


@pytest.mark.asyncio
async def test_process_image_part_async_success_and_disabled(monkeypatch):
    item = SimpleNamespace(
        mime_type="image/png",
        inline_data=SimpleNamespace(data=b"xyz"),
    )

    monkeypatch.setattr(su.Config, "upload_base64_image", None)
    assert await su._process_image_part(item, 1, 2, 3) is None

    async def fake_upload(trace_id, span_id, image_name, base64_string):
        assert trace_id == "1"
        assert span_id == "2"
        assert image_name == "content_3.png"
        assert base64_string
        return "https://example/image"

    monkeypatch.setattr(su.Config, "upload_base64_image", fake_upload)
    payload = await su._process_image_part(item, 1, 2, 3)
    assert payload == {"type": "image_url", "image_url": {"url": "https://example/image"}}


def test_process_image_part_sync_success(monkeypatch):
    item = SimpleNamespace(
        mime_type="image/jpeg",
        inline_data=SimpleNamespace(data=b"xyz"),
    )

    async def fake_upload(trace_id, span_id, image_name, base64_string):
        assert image_name == "content_0.jpeg"
        return "https://example/sync-image"

    monkeypatch.setattr(su.Config, "upload_base64_image", fake_upload)
    payload = su._process_image_part_sync(item, 1, 2, 0)
    assert payload == {"type": "image_url", "image_url": {"url": "https://example/sync-image"}}


def test_run_async_with_running_loop_branch(monkeypatch):
    calls = {"started": False, "joined": False}

    class FakeLoop:
        def is_running(self):
            return True

    class FakeThread:
        def __init__(self, target):
            self.target = target

        def start(self):
            calls["started"] = True

        def join(self):
            calls["joined"] = True

    monkeypatch.setattr(su.asyncio, "get_running_loop", lambda: FakeLoop())
    monkeypatch.setattr(su.threading, "Thread", FakeThread)

    async def coro():
        return None

    coro_obj = coro()
    su.run_async(coro_obj)
    coro_obj.close()
    assert calls["started"] is True
    assert calls["joined"] is True


@pytest.mark.asyncio
async def test_set_input_attributes_async_string_and_list(monkeypatch):
    monkeypatch.setattr(su, "should_send_prompts", lambda: True)
    span = FakeSpan()
    await su.set_input_attributes(span, ["hello", ["world"]])
    assert any(key.endswith(".role") for key in span.attributes.keys())
    assert any(key.endswith(".content") for key in span.attributes.keys())


@pytest.mark.asyncio
async def test_set_input_attributes_async_noop_paths(monkeypatch):
    monkeypatch.setattr(su, "should_send_prompts", lambda: False)
    span = FakeSpan(recording=True)
    await su.set_input_attributes(span, ["hello"])
    assert span.attributes == {}

    monkeypatch.setattr(su, "should_send_prompts", lambda: True)
    span_not_recording = FakeSpan(recording=False)
    await su.set_input_attributes(span_not_recording, ["hello"])
    assert span_not_recording.attributes == {}


def test_set_input_attributes_sync_paths(monkeypatch):
    monkeypatch.setattr(su, "should_send_prompts", lambda: True)
    span = FakeSpan()
    su.set_input_attributes_sync(span, ["hello"])
    assert any(key.endswith(".role") for key in span.attributes.keys())

    monkeypatch.setattr(su, "should_send_prompts", lambda: False)
    span2 = FakeSpan()
    su.set_input_attributes_sync(span2, ["hello"])
    assert span2.attributes == {}


def test_set_model_input_attributes_and_response_paths(monkeypatch):
    span = FakeSpan()
    kwargs = {
        "prompt": "p",
        "temperature": 0.4,
        "max_output_tokens": 42,
        "top_p": 0.9,
        "top_k": 20,
        "presence_penalty": 0.1,
        "frequency_penalty": 0.2,
    }
    su.set_model_input_attributes(span, kwargs, "model-1")
    assert span.attributes
    assert any("request.model" in k for k in span.attributes)

    monkeypatch.setattr(su, "should_send_prompts", lambda: True)
    su.set_response_attributes(span, "model-1", "done")
    assert any("completion.0.content" in k for k in span.attributes)

    usage = SimpleNamespace(total_token_count=11, candidates_token_count=7, prompt_token_count=4)
    su.set_model_response_attributes(span, "model-1", usage)
    assert any("usage.input_tokens" in k for k in span.attributes)


def test_setters_noop_when_not_recording(monkeypatch):
    span = FakeSpan(recording=False)
    monkeypatch.setattr(su, "should_send_prompts", lambda: True)
    su.set_model_input_attributes(span, {"prompt": "x"}, "m")
    su.set_response_attributes(span, "m", "y")
    su.set_model_response_attributes(span, "m", None)
    assert span.attributes == {}
