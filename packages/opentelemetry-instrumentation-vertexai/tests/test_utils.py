from opentelemetry.instrumentation.vertexai import config
from opentelemetry.instrumentation.vertexai import event_models
from opentelemetry.instrumentation.vertexai import utils


def test_should_send_prompts_env_and_override(monkeypatch):
    monkeypatch.delenv(utils.TRACELOOP_TRACE_CONTENT, raising=False)
    monkeypatch.setattr(utils.context_api, "get_value", lambda key: False)
    assert utils.should_send_prompts() is True

    monkeypatch.setenv(utils.TRACELOOP_TRACE_CONTENT, "False")
    monkeypatch.setattr(utils.context_api, "get_value", lambda key: False)
    assert utils.should_send_prompts() is False

    monkeypatch.setenv(utils.TRACELOOP_TRACE_CONTENT, "False")
    monkeypatch.setattr(utils.context_api, "get_value", lambda key: True)
    assert utils.should_send_prompts() is True


def test_should_emit_events_tracks_config():
    original = config.Config.use_legacy_attributes
    try:
        config.Config.use_legacy_attributes = True
        assert utils.should_emit_events() is False
        config.Config.use_legacy_attributes = False
        assert utils.should_emit_events() is True
    finally:
        config.Config.use_legacy_attributes = original


def test_dont_throw_success_and_exception_path():
    called = {"count": 0}

    def ok(x):
        return x + 1

    def boom():
        raise ValueError("x")

    wrapped_ok = utils.dont_throw(ok)
    assert wrapped_ok(1) == 2

    def exception_logger(exc):
        called["count"] += 1

    original = config.Config.exception_logger
    try:
        config.Config.exception_logger = exception_logger
        wrapped_boom = utils.dont_throw(boom)
        assert wrapped_boom() is None
        assert called["count"] == 1
    finally:
        config.Config.exception_logger = original


def test_config_and_event_model_defaults():
    assert config.Config.use_legacy_attributes is True
    assert callable(config.Config.upload_base64_image)

    msg = event_models.MessageEvent(content="hi")
    assert msg.role == "user"
    assert msg.tool_calls is None

    choice = event_models.ChoiceEvent(index=0, message={"content": "x", "role": "assistant"})
    assert choice.finish_reason == "unknown"
    assert choice.tool_calls is None
